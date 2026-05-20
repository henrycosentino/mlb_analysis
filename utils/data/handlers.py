import gc
import pytz
import time
import random
import requests
import threading
import mlbstatsapi
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import deque
from datetime import datetime
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, as_completed


# --- Kalshi API Downloader ---
class KalshiHistorical:
    def __init__(
        self, 
        read_limit: int = 500, 
        max_workers: int = 5
    ) -> None:
        
        self.read_limit = read_limit
        self.max_workers = max_workers

        self._rate_lock = threading.Lock()
        self.call_timestamps = deque()

        self.rest_url = "https://api.elections.kalshi.com/trade-api/v2"

        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=max_workers,
            pool_maxsize=max_workers * 2,
            max_retries=requests.adapters.Retry(
                total=5,                     
                backoff_factor=1.0,   
                status_forcelist=[429, 500, 502, 503, 504],
                respect_retry_after_header=True 
            )
        )
        self.session.mount("https://", adapter)

    def _rate_limit(self):
        """
        Thread-safe rate limiting. Blocks the calling thread if the rolling
        60-second window is at capacity, then registers the call.
        """
        
        with self._rate_lock:
            while True:
                current_time = time.time()

                while self.call_timestamps and current_time - self.call_timestamps[0] >= 60:
                    self.call_timestamps.popleft()

                if len(self.call_timestamps) < self.read_limit:
                    self.call_timestamps.append(current_time)
                    return

                sleep_time = 60 - (current_time - self.call_timestamps[0]) + 0.05
                self._rate_lock.release()
                try:
                    time.sleep(max(sleep_time, 0.05))
                finally:
                    self._rate_lock.acquire()

    def get_markets(
        self, 
        params: dict = {}, 
        historical: bool = False
    ) -> list:
        """
        Gets all Kalshi markets data

        Arguments:
            params: dictionary for API query filtering
            historical: use `True` for markets settled before the recent cutoff
        """

        all_markets = []
        cursor = None
        endpoint = f"{self.rest_url}/historical/markets" if historical else f"{self.rest_url}/markets"

        with tqdm(desc="Getting Kalshi Markets Data", unit=" pages") as pbar:
            while True:
                p = dict(params)
                if cursor:
                    p["cursor"] = cursor

                self._rate_limit()
                response = self.session.get(endpoint, params=p, timeout=10)
                response.raise_for_status()
                data = response.json()

                if 'markets' not in data:
                    raise KeyError("Markets data key error.")
                all_markets.extend(data['markets'])
                pbar.update(1)
                pbar.set_postfix({"total markets": len(all_markets)})

                cursor = data.get('cursor')
                if not cursor:
                    break

        return all_markets

    def search_for_tickers(
        self, 
        market_data: list, 
        keywords: list = ["LAA"], 
        drop_duplicates: bool = True
    ) -> list:
        """
        Searches Kalshi historic market data by ticker for keyword(s).
        Returns the matching tickers.

        Arguments:
            market_data: a list of Kalshi market data
            keywords: a list of keywords to search ticker for
            drop_duplicates: remove duplicate tickers
        """

        tickers = []
        for market in tqdm(market_data, desc="Getting Kalshi Market Tickers"):
            ticker = market.get('ticker')
            if not ticker:
                continue

            if keywords:
                if any(k in ticker for k in keywords): tickers.append(ticker)
            else:
                tickers.append(ticker)

        if drop_duplicates:
            tickers = list(set(tickers))

        return tickers

    def _fetch_all_trades_for_ticker(
        self, 
        ticker: str, 
        endpoint: str
    ) -> tuple[str, list]:
        """
        Fetches all paginated trades for a single ticker.
        Returns (ticker, trades_list). Designed to be called from a thread pool.
        """

        time.sleep(random.uniform(0.05, 0.3))
        trades = []
        params = {"ticker": ticker, "limit": 1000}
        cursor = None

        while True:
            if cursor:
                params["cursor"] = cursor

            self._rate_limit()
            response = self.session.get(
                self.rest_url+endpoint, params=params, timeout=10
            )
            response.raise_for_status()
            data = response.json()

            if 'trades' not in data:
                raise KeyError(f"Trades data key error for ticker '{ticker}'.")
            trades.extend(data['trades'])

            cursor = data.get('cursor')
            if not cursor:
                break

        return ticker, trades

    def get_trades(
        self, 
        tickers: list, 
        endpoint: str
    ) -> dict:
        """
        Gets historic trade data for one or more tickers, fetching concurrently.

        Arguments:
            tickers: a list of tickers in Kalshi format
            endpoint: Kalshi REST API endpoint 
        """
        all_trades = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._fetch_all_trades_for_ticker, ticker, endpoint): ticker
                for ticker in tickers
            }

            with tqdm(total=len(futures), desc=f"Getting Kalshi Trades") as pbar:
                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        ticker, trades = future.result()
                        all_trades[ticker] = trades
                    except Exception as e:
                        print(f"\nError fetching '{ticker}': {e}")
                        all_trades[ticker] = None
                    pbar.update(1)

        return all_trades
    
    def download_game_trades(
        self,
        game_trades: dict,
        file_name: str,
        dir_path: str,
        compression: str
    ) -> None:
        """
        Downloads Kalshi game trade data as parquet file

        Arguments:
            game_trades: dictionary containing all Kalshi trades
            file_name: the parquet file name
            dir_path: the directories path where the file will be stored
            compression: parquet compression method
        """

        trades_list = []
        for ticker, trade_list in game_trades.items():
            trades_list.extend(trade_list)

        kalshi_path = Path(dir_path) / file_name

        if Path(kalshi_path).exists():
            print(f"'{file_name}' already exists in system, skipping download.")
            return
        
        kalshi_path.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(trades_list)
        df.to_parquet(kalshi_path, index=False, compression=compression)
        print(f"'{file_name}' download complete.")
        

# --- mlbstatsapi Downloader ---
class MLBData:
    def __init__(
        self,
        team: str,
        start_dt: str,
        end_dt: str
    ) -> None:

        self.team = team
        self.start_dt = start_dt
        self.end_dt = end_dt

        self.mlb_api = mlbstatsapi.Mlb()

        self.schedule = self.mlb_api.get_schedule(
            start_date = self.start_dt, 
            end_date = self.end_dt
        )

    def get_data(self) -> tuple[dict, dict]:
        """
        Get home run and score data for all scheduled games.
        """
        if not self.schedule.dates:
            print("No games found in schedule")
            return {}, {}

        hr_data_dict = {}
        score_data_dict = {}
        for date in tqdm(self.schedule.dates, desc=f"Getting mlbstatsapi Home Run & Score Data for {self.team}"): # Iterate through all the dates along the game schedule

            for game in date.games: # For that specific date, iterate through the games played
                home_team = game.teams.home.team.name
                away_team = game.teams.away.team.name
                game_id = game.gamepk

                if self.team in home_team or self.team in away_team: # Check if the team we want data for is playing in the game
                    try:
                        game_pbp = self.mlb_api.get_game_play_by_play(game_id)
                    except Exception as e:
                        print(f"Error getting play by play data for {game_id}: {e}")
                        continue

                    # Get home run hitting data (time stamp and team who hit the home run)
                    home_runs = [play for play in game_pbp.allplays if play.result.event == 'Home Run']

                    # Get home run data for each home run hit in the game
                    hr_data_ls = []
                    for hr in home_runs:
                        if hasattr(hr, 'playendtime') and hr.playendtime:
                            batting_team = hr.about.halfinning
                
                            if batting_team == 'top':
                                hr_team = away_team
                            else:
                                hr_team = home_team

                            runners_on_base = sum(1 for runner in hr.runners 
                            if hasattr(runner.movement, 'originbase') 
                            and runner.movement.originbase is not None
                            and runner.details.eventtype == 'home_run')
                                
                            hr_data_ls.append({
                                "team": hr_team,
                                "time": hr.playendtime,
                                "runners_on_base": runners_on_base,
                                "inning": hr.about.inning
                                })

                    key = f"{game_id}_{date.date}"
                    hr_data_dict[key] = hr_data_ls

                    # Get game score data (home team score and away team score)
                    plays = game_pbp.allplays

                    score_dict = {}
                    for play in plays:

                        home_score = play.result.homescore
                        away_score = play.result.awayscore
                        timestamp = play.playendtime

                        score_dict[timestamp] = {
                            home_team: home_score,
                            away_team: away_score
                            }
                            
                    score_data_dict[key] = score_dict

        return hr_data_dict, score_data_dict
    
    def download_score_data(
        self,
        score_data_dict: dict,
        file_name: str,
        dir_path: str,
        compression: str
    ) -> None:
        """
        Downloads mlbstatsapi game score data as parquet file

        Arguments:
            score_data_dict: dictionary containing all score data
            file_name: the file name
            dir_path: the directories path where the file will be stored
            compression: parquet compression method
        """

        score_data = []
        for game_id, timestamps in score_data_dict.items():
            for timestamp, scores in timestamps.items():
                for team, score in scores.items():
                    score_data.append({
                        'game_id': game_id,
                        'timestamp': timestamp,
                        'team': team,
                        'score': score
                    })

        score_path = dir_path + file_name

        if Path(score_path).exists():
            print(f"'{file_name}' already exists in system, skipping download.")
            return None
        else:
            df = pd.DataFrame(score_data)
            df.to_parquet(score_path, index=False, compression=compression)
            print(f"'{file_name}' download complete.")
            return None

    def download_homerun_data(
        self,
        hr_data_dict: dict,
        file_name: str,
        dir_path: str,
        compression: str
    ) -> None:
        """
        Downloads mlbstatsapi game home run data as parquet file

        Arguments:
            hr_data_dict: dictionary containing all home run data
            file_name: the file name
            dir_path: the directories path where the file will be stored
            compression: parquet compression method
        """

        hr_data = []
        for game_id, timestamps in hr_data_dict.items():
            for entry in timestamps:
                hr_data.append({
                    'game_id': game_id,
                    'team': entry['team'],
                    'time': entry['time'],
                    'runners_on_base': entry['runners_on_base'],
                    'inning': entry['inning']
                })

        homerun_path = dir_path + file_name

        if Path(homerun_path).exists():
            print(f"'{file_name}' already exists in system, skipping download.")
            return None
        else:
            df = pd.DataFrame(hr_data)
            df.to_parquet(homerun_path, index=False, compression=compression)
            print(f"'{file_name}' download complete.")
            return None
        

# --- Data Processing ---
class DataProcessing:
    def __init__(
        self, 
        team: str,
        trade_path: str, 
        score_path: str,
        homerun_path: str,
        tz: str = 'America/Los_Angeles',
    ):
        # Variable parameters
        self.team = team
        self.trade_path = trade_path
        self.score_path = score_path
        self.homerun_path = homerun_path
        self.tz = pytz.timezone(tz)

        # Invariable attributes
        self.team_map = {
            "LAA" : "Los Angeles Angels",
            "LAD" : "Los Angeles Dodgers",
            "SF"  : "San Francisco Giants",
            "ATH" : "Athletics",
            "CWS" : "Chicago White Sox",
            "CHC" : "Chicago Cubs",
            }
        self.tickers = self._get_tickers()
        self.teamFull = self._get_teamFull()
        
    def _get_teamFull(self) -> str:
        if self.team in self.team_map:
            return self.team_map[self.team]
        else:
            raise KeyError(f"Invalid 'team'. Available teams: {list(self.team_map.keys())}")

    def _get_tickers(self) -> set:
        """
        Extract unique tickers from trade data.
        """

        trade_df = pd.read_parquet(self.trade_path)
        
        if 'ticker' not in trade_df.columns:
            raise KeyError("'ticker' column not found in trades data.")
        
        return set(trade_df['ticker'].unique())
        
    def _ticker(
        self, 
        tickers: set, 
        target_df: pd.DataFrame
    ) -> list:
        """
        Match dates from target_df to tickers based on embedded date strings.

        Arguments:
            tickers: set of tickers
            target_df: homerun or score data frames
        """

        if 'time_pst' not in target_df.columns:
            raise ValueError("DataFrame must have a 'time_pst' column.")
        
        ticker_ls = []

        for time in target_df['time_pst']:
            time_date = time.date()
            found_ticker = False 
            
            for ticker in tickers:
                date_str = ticker.split('-')[1][:7]
                year = int('20' + date_str[:2])
                month_str = date_str[2:5]
                day = int(date_str[5:7])
                ticker_date = datetime.strptime(
                    f"{year}-{month_str}-{day}", 
                    '%Y-%b-%d'
                ).date()
                
                if time_date == ticker_date:
                    ticker_ls.append(ticker)
                    found_ticker = True
                    break 
            
            if not found_ticker:
                ticker_ls.append(None)

        return ticker_ls

    def get_trade_df(self) -> pd.DataFrame:
        """
        Load and process trade data
        """

        trade_df = pd.read_parquet(self.trade_path)

        # Validate required columns
        required_cols = ['ticker', 'created_time', 'yes_price_dollars', 'count_fp']
        missing_cols = [col for col in required_cols if col not in trade_df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}.")
        
        # Convert timestamp
        trade_df['time_pst'] = pd.to_datetime(
            trade_df['created_time'], 
            format='ISO8601',
            utc=True
        ).dt.tz_convert(self.tz)

        # Rename count column
        trade_df = trade_df.rename(columns={'count_fp': 'trade_count'})

        # Convert to numeric
        trade_df['yes_price_dollars'] = pd.to_numeric(trade_df['yes_price_dollars'], errors='coerce')
        trade_df['trade_count'] = pd.to_numeric(trade_df['trade_count'], errors='coerce')
        
        trade_df = trade_df.sort_values('time_pst').reset_index(drop=True)
        
        return trade_df[['ticker', 'time_pst', 'yes_price_dollars', 'trade_count']]
    
    def get_score_df(self) -> pd.DataFrame:
        """
        Load and process score data, calculating score differential
        """

        score_df = pd.read_parquet(self.score_path)
        
        # Validate required columns
        required_cols = ['team', 'game_id', 'timestamp', 'score']
        missing_cols = [col for col in required_cols if col not in score_df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}.")
        
        score_df['time_pst'] = pd.to_datetime(
            score_df['timestamp'], 
            utc=True
        ).dt.tz_convert(self.tz)

        # Separate home team and opponent scores
        opp_score_df = score_df[score_df['team'] != self.teamFull][
            ['game_id', 'time_pst', 'score']
        ].rename(columns={'score': 'opp_score'}
        ).reset_index(drop=True)
        
        ht_score_df = score_df[score_df['team'] == self.teamFull][
            ['game_id', 'time_pst', 'score']
        ].rename(columns={'score': f'{self.team.lower()}_score'}
        ).reset_index(drop=True)

        # Merge
        score_df = pd.merge(
            ht_score_df, 
            opp_score_df, 
            how='outer', 
            on=['time_pst', 'game_id']
        )
        
        # Fill any missing values and cast as int type
        score_df[f'{self.team.lower()}_score'] = score_df[f'{self.team.lower()}_score'].fillna(0).astype(int)
        score_df['opp_score'] = score_df['opp_score'].fillna(0).astype(int)
        
        score_df = score_df.drop(columns=['game_id'])

        # Add ticker column
        score_df.insert(0, 'ticker', self._ticker(self.tickers, score_df))

        # Drop rows with no matching Kalshi market
        score_df = score_df.dropna(subset=['ticker']).reset_index(drop=True)

        return score_df
    
    def get_homerun_df(self) -> pd.DataFrame:
        """
        Load and process homerun data.
        """
        
        homerun_df = pd.read_parquet(self.homerun_path)

        # Validate required columns
        required_cols = ['time', 'runners_on_base', 'inning', 'team']
        missing_cols = [col for col in required_cols if col not in homerun_df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")
        
        homerun_df['time_pst'] = pd.to_datetime(
            homerun_df['time'], 
            utc=True
        ).dt.tz_convert(self.tz)

        # Create home run indicators
        homerun_df['homerun_dummy'] = 1
        
        team_col = f'{self.team.lower()}_homerun_dummy'
        homerun_df[team_col] = (
            homerun_df['team'] == self.teamFull
        ).astype(int)

        homerun_df['opp_homerun_dummy'] = (
            homerun_df[team_col] == 0
        ).astype(int)

        output_cols = [
            'time_pst', 'homerun_dummy', team_col, 
            'opp_homerun_dummy', 'runners_on_base', 
            'inning'
        ]
        
        homerun_df = homerun_df[output_cols]

        # Add ticker column
        homerun_df.insert(0, 'ticker', self._ticker(self.tickers, homerun_df))

        # Drop rows with no matching Kalshi market
        homerun_df = homerun_df.dropna(subset=['ticker']).reset_index(drop=True)

        return homerun_df
  

# --- Feature Engineering ---
class FeatureEngineering(DataProcessing):
    def __init__(self,
        team: str,
        trade_path: str,
        score_path: str,
        homerun_path: str,
        tz: str = 'America/Los_Angeles'
    ):
        super().__init__(
            team=team,
            trade_path=trade_path,
            score_path=score_path,
            homerun_path=homerun_path,
            tz=tz
        )
        
        # Processed data frames
        self.trade_df = self.get_trade_df()
        self.score_df = self.get_score_df()
        self.homerun_df = self.get_homerun_df()

    def _opponent_dummy(
        self,
        homerun_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Adds the home teams opponent feature, creating several dummy variable columns.
        """

        ticker_ls = list(homerun_df['ticker'])
        opponent_ls = []

        for t in ticker_ls:

            if t is None or pd.isna(t):
                opponent_ls.append(None)
                continue

            parts = t.split('-')
            if len(parts) < 3:
                opponent_ls.append(None)
                print(f"Unexpected ticker format: '{t}'")
                continue

            # Middle section e.g. '26MAY181840BOSATL' or '25APR18LADTEX2'
            ticker_slice = parts[1]
            if ticker_slice[-1].isdigit():  # strip doubleheader suffix
                ticker_slice = ticker_slice[:-1]

            window_len = len(self.team)

            # Slide window to find self.team in ticker_slice
            ht_start_idx = None
            for i in range(len(ticker_slice) - window_len + 1):
                if ticker_slice[i: i + window_len] == self.team:
                    ht_start_idx = i
                    break

            if ht_start_idx is None:
                opponent_ls.append(None)
                print(f"Team '{self.team}' not found in ticker: '{t}'")
                continue

            ht_end_idx = ht_start_idx + window_len

            # Case 1: opponent is to the right of self.team
            if ht_end_idx < len(ticker_slice):
                opponent = ticker_slice[ht_end_idx:]

            # Case 2: opponent is to the left of self.team
            else:
                prefix = ticker_slice[5:ht_start_idx]
                opponent = prefix.lstrip('0123456789')

            # Validate — opponent should be 2-3 uppercase letters
            if not opponent.isalpha() or not opponent.isupper() or len(opponent) not in (2, 3):
                opponent_ls.append(None)
                print(f"Could not parse opponent from ticker: '{t}' (got '{opponent}')")
                continue

            opponent_ls.append(opponent)

        if len(opponent_ls) != len(ticker_ls):
            raise ValueError("Error calculating opponent team feature for home run data frame.")

        homerun_df['opponent'] = opponent_ls
        homerun_df = pd.get_dummies(homerun_df, columns=['opponent'], drop_first=True, dtype=int)

        return homerun_df

    def _score_delta(
        self,
        score_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculates the score delta, and absolute score delta features.
        """

        score_df['score_delta'] = (
            score_df[f'{self.team.lower()}_score'] - 
            score_df['opp_score']
        )

        score_df['score_delta_abs'] = score_df['score_delta'].abs()

        return score_df
    
    def _resample(
        self,
        trade_df: pd.DataFrame,
        resample_type: str = 'median'
    ) -> pd.DataFrame:
        """
        Helper function that resamples the price feature in trades data frame.
        Essentially, it removes the effect of order book sweeps by 'whales'.

        Args:
            resample_type: how the resample is defined.
        """
    
        resample_types = ['median', 'mean', 'min', 'max', 'first', 'last']
        if resample_type not in resample_types:
            raise ValueError(f"'resample_type' must be one of: {resample_types}.")
        
        agg_funcs = {
            'yes_price_dollars' : resample_type,
            'trade_count'       : 'sum'
        }

        trade_df = (
            trade_df.groupby(['ticker', 'time_pst'], as_index=False)
            .agg(agg_funcs)
        )

        trade_df = trade_df.sort_values('time_pst').reset_index(drop=True)

        return trade_df
    
    def _rolling_vol(
        self,
        trade_df_sorted: pd.DataFrame,
        window_params: tuple = (2, 50),
        window_types: list = ['time', 'trade'],
        chg_type: str = 'px_chg'
    )-> dict:
        """
        Calculates rolling volatility feature for a given window.

        Arguments:
            trade_df_sorted: the trades data frame sorted by time
            window_params: the starting and ending index of the windows size, inclusive
            window_type: can be set to either time or trade count
            chg_type: 'pct_px_chg' or 'px_chg'
        """

        if not (isinstance(window_params, (tuple, list)) and len(window_params) == 2):
            raise TypeError("window_params must be a tuple or list of 2 integers.")

        if any(not isinstance(x, int) or x <= 0 for x in window_params):
            raise ValueError(f"All window_params must be positive integers: {window_params}.")

        if window_params[0] > window_params[1]:
            raise ValueError(f"Start window ({window_params[0]}) cannot be greater than end window ({window_params[1]}).")

        allowed_types = {'time', 'trade'}
        if not set(window_types).issubset(allowed_types) or not window_types:
            raise ValueError(f"window_types must be a list containing 'time', 'trade', or both. Got: {window_types}.")

        if ('time' in window_types) and ('time_pst' not in trade_df_sorted.columns):
            raise ValueError(f"Column 'time_pst' required for time-based rolling volatility. Found: {trade_df_sorted.columns.tolist()}.")
        
        if 'yes_price_dollars' not in trade_df_sorted.columns:
            raise ValueError(f"Column 'yes_price_dollars' required for time-based rolling volatility. Found: {trade_df_sorted.columns.tolist()}.")
        
        if chg_type == 'pct_px_chg':
            trade_df_sorted[chg_type] = trade_df_sorted.groupby('ticker')['yes_price_dollars'].pct_change().fillna(0)
        else:
            trade_df_sorted[chg_type] = trade_df_sorted.groupby('ticker')['yes_price_dollars'].diff().fillna(0) 
        
        def _time(t, df, chg_type):
            """Calculates rolling standard deviation, with window sizes based on time (minutes)."""
            key_name = f"rolling_std_{t}mins"
            new_col = (
                df.groupby('ticker', group_keys=False)[[chg_type, 'time_pst']] 
                .rolling(
                    window=f'{t}min',
                    on='time_pst',
                    min_periods=2
                )[chg_type]
                .std()
                .values
            )
            return key_name, new_col

        def _trade(t, df, chg_type):
            """Calculates rolling standard deviation, with window sizes based on the trade count."""
            key_name = f"rolling_std_{t}trades" 
            new_col = (
                df.groupby('ticker', group_keys=False)[chg_type]
                .rolling(
                    window=t,
                    min_periods=2
                )
                .std()
                .values
            )
            return key_name, new_col

        new_cols = {}
        for t in range(window_params[0], window_params[1] + 1):

            if 'time' in window_types and 'trade' in window_types:
                time_key, time_col = _time(t, trade_df_sorted, chg_type)
                trade_key, trade_col = _trade(t, trade_df_sorted, chg_type)
                new_cols[time_key] = time_col
                new_cols[trade_key] = trade_col

            elif 'time' in window_types and 'trade' not in window_types:
                time_key, time_col = _time(t, trade_df_sorted, chg_type)
                new_cols[time_key] = time_col

            else:
                trade_key, trade_col = _trade(t, trade_df_sorted, chg_type)
                new_cols[trade_key] = trade_col

        trade_df_sorted = pd.concat([trade_df_sorted, pd.DataFrame(new_cols, index=trade_df_sorted.index)], axis=1)

        return trade_df_sorted
    
    def _merge_frames(
        self,
        resample_type: str = 'median',
        vol_window_params: tuple = (2, 50),
        tolerance: str = '3min',
        chg_type: str = 'px_chg'
    ) -> pd.DataFrame:
        """
        Merges home run, trades, and score data frames using a 'backward' method to find the nearest trade/score by timestamp.

        Arguments:
            resample_type: 'median', 'mean', 'min', 'max', 'first', 'last', resampling price data to counter the effect of order book sweeps
            vol_window_params: (starting window size, ending window size), inclusive
            tolerance: the maximum number of minutes pd.merge_asof() can look 'backward' to find the nearest trade
            chg_type: 'pct_px_chg' or 'px_chg'
        """
        
        # Sort data frames
        homerun_df_sorted = self.homerun_df.copy().sort_values('time_pst').reset_index(drop=True)
        trade_df_sorted = self.trade_df.copy().sort_values('time_pst').reset_index(drop=True)

        # Resample trades data frame
        trade_df_sorted = self._resample(trade_df=trade_df_sorted, resample_type=resample_type)
        
        # Create opponent team dummy variable feature
        homerun_df_sorted = self._opponent_dummy(homerun_df=homerun_df_sorted)
        
        # Create rolling volatility feature by time and trade count window types
        trade_df_sorted = self._rolling_vol(
            trade_df_sorted=trade_df_sorted,
            window_params=vol_window_params,
            window_types=['time', 'trade'],
            chg_type=chg_type
        )
        
        # Merge home run and trades data frames
        homerun_event_df = pd.merge_asof(
            homerun_df_sorted,
            trade_df_sorted,
            on='time_pst',
            by='ticker',
            direction='backward',
            tolerance=pd.Timedelta(tolerance)
        )
        
        # Create score delta feature and sort score data frame
        score_df = self._score_delta(self.score_df.copy())
        score_df_sorted = score_df.sort_values('time_pst').reset_index(drop=True)

        # Lag score data to align correctly with home run data
        lagged_score_cols = [f'{self.team.lower()}_score', 'opp_score', 'score_delta','score_delta_abs']
        score_df_sorted[lagged_score_cols] = score_df_sorted.groupby('ticker')[lagged_score_cols].shift(1).bfill()
        score_df_sorted = score_df_sorted.sort_values('time_pst').reset_index(drop=True)
        
        # Merge home run events and score frames
        homerun_events_w_score_df = pd.merge_asof(
            homerun_event_df,
            score_df_sorted,
            on='time_pst',
            by='ticker',
            direction='backward'
        )
        
        # Merge all data together
        final_df = pd.merge(trade_df_sorted, homerun_events_w_score_df, how='outer')
        
        # Fill dummy variables
        final_df[f'{self.team.lower()}_homerun_dummy'] = final_df[f'{self.team.lower()}_homerun_dummy'].fillna(0).astype(int)
        final_df['opp_homerun_dummy'] = final_df['opp_homerun_dummy'].fillna(0).astype(int)
        final_df['homerun_dummy'] = final_df['homerun_dummy'].fillna(0).astype(int)

        # Fill score enteries
        score_cols = [f'{self.team.lower()}_score', 'opp_score', 'score_delta', 'score_delta_abs']
        final_df[score_cols] = (
            final_df.groupby('ticker')[score_cols]
            .ffill()
            .fillna(0)
        )

        final_df = final_df.dropna(subset=['yes_price_dollars']).reset_index(drop=True)
        
        return final_df
    
    def get_generic_df(
        self,
        resample_type: str = 'median',
        vol_window_params: tuple = (2, 50),
        merge_tolerance: str = '3min',
        trade_tolerance: str = '3min',
        chg_type: str = 'px_chg'
    ) -> pd.DataFrame:
        """
        Engineers price reaction applying the generic price change method.

        Arguments:
            resample_type: 'median', 'mean', 'min', 'max', 'first', 'last', resampling price data to counter the effect of order book sweeps
            vol_window_params: (starting window size, ending window size), inclusive
            merge_tolerance: the maximum number of minutes pd.merge_asof() can look 'backward' to find the nearest trade
            trade_tolerance: the maximum number of minutes the next trade can be away from the home run trade time
            chg_type: 'pct_px_chg' or 'px_chg'
        """
        
        if chg_type not in ['pct_px_chg', 'px_chg']:
            raise ValueError(f"Argument 'chg_type' must be 'pct_px_chg' or 'px_chg', not '{chg_type}'.")

        # Merge home run, trades, and score data frames
        df = self._merge_frames(
            resample_type=resample_type,
            vol_window_params=vol_window_params,
            tolerance=merge_tolerance,
            chg_type=chg_type
        )

        # Create temporary columns for mask 
        df['_next_price'] = df['yes_price_dollars'].shift(-1)
        df['_next_time'] = df['time_pst'].shift(-1)
        df['_next_ticker'] = df['ticker'].shift(-1)
        df['_next_homerun'] = df['homerun_dummy'].shift(-1)

        # Mask for valid home run rows
        time_diff = pd.to_datetime(df['_next_time']) - pd.to_datetime(df['time_pst'])
        valid = (
            (df['homerun_dummy'] == 1) &                   # A home run occured
            (time_diff < pd.Timedelta(trade_tolerance)) &  # Apply a trade tolerance
            (df['_next_ticker'] == df['ticker']) &         # Use the right game
            (df['_next_homerun'] != 1)                     # Make sure a home run didn't occur in the next trade
        )

        if chg_type == 'pct_px_chg':
            df[f'g_{chg_type}'] = np.where(valid, (df['_next_price'] - df['yes_price_dollars']) / df['yes_price_dollars'], np.nan)
        else:
            df[f'g_{chg_type}'] = np.where(valid, df['_next_price'] - df['yes_price_dollars'], np.nan)

        # Drop temporary columns
        df = df.drop(columns=['_next_price', '_next_time', '_next_ticker', '_next_homerun'])

        # Filter for home runs
        final_df = df[df['homerun_dummy'] == 1].reset_index(drop=True)

        return final_df
    
    def get_forward_df(
        self,
        resample_type: str = 'median',
        vol_window_params: tuple = (2, 50),
        merge_tolerance: str = '3min',
        window_params: tuple = (1, 6),
        bin_params: tuple = (1, 6),
        chg_type: str = 'pct_px_chg'
    ) -> pd.DataFrame:
        """
        Engineers price reaction applying the forward binned mean price change method for different window/bin combinations.

        Arguments:
            resample_type: 'median', 'mean', 'min', 'max', 'first', 'last', resampling price data to counter the effect of order book sweeps
            vol_window_params: (starting window size, ending window size), inclusive
            merge_tolerance: the maximum number of minutes pd.merge_asof() can look 'backward' to find the nearest trade
            window_params: (starting window size, ending window size), inclusive
            bin_params: (starting number of trades per bin, ending number of trades per bin), inclusive
            chg_type: 'pct_px_chg' or 'px_chg'
        """

        if chg_type not in ['pct_px_chg', 'px_chg']:
            raise ValueError(f"Argument 'chg_type' must be 'pct_px_chg' or 'px_chg', not '{chg_type}'.")

        # Merge home run, trades, and score data frames
        df = self._merge_frames(
            resample_type=resample_type,
            vol_window_params=vol_window_params,
            tolerance=merge_tolerance,
            chg_type=chg_type
        )

        # Pre-allocate result arrays
        n = len(df)
        result_arrays = {
            f'f_{chg_type}_w{w}_b{b}': np.full(n, np.nan)
            for w in range(window_params[0], window_params[1] + 1)
            for b in range(bin_params[0], bin_params[1] + 1)
        }

        price_arr = df['yes_price_dollars'].to_numpy(dtype=np.float64)
        homerun_arr = df['homerun_dummy'].to_numpy(dtype=np.int8)
        ticker_arr = df['ticker'].to_numpy()

        homerun_indices = df.index[df['homerun_dummy'] == 1].tolist()

        for idx in tqdm(homerun_indices, desc="Forward Binned Method"):
            homerun_px = price_arr[idx]
            ticker = ticker_arr[idx]

            for w in range(window_params[0], window_params[1] + 1):
                for b in range(bin_params[0], bin_params[1] + 1):
                    start_idx = idx + w
                    end_idx   = idx + w + b - 1

                    if end_idx >= n:
                        continue

                    # No second home run in the entire region
                    if homerun_arr[idx : end_idx + 1].sum() > 1:
                        continue

                    # Same ticker throughout the forward bin
                    if (ticker_arr[start_idx : end_idx + 1] != ticker).any():
                        continue

                    avg_px = price_arr[start_idx : end_idx + 1].mean()

                    if chg_type == 'pct_px_chg':
                        result_arrays[f'f_pct_px_chg_w{w}_b{b}'][idx] = (avg_px - homerun_px) / homerun_px
                    else:
                        result_arrays[f'f_px_chg_w{w}_b{b}'][idx] = avg_px - homerun_px

        # Attach columns
        new_cols_df = pd.DataFrame(result_arrays, index=df.index)
        df = pd.concat([df, new_cols_df], axis=1)

        # Filter for home runs
        final_df = df[df['homerun_dummy'] == 1].reset_index(drop=True)

        return final_df

    def get_double_df(
        self,
        resample_type: str = 'median',
        vol_window_params: tuple = (2, 50),
        merge_tolerance: str = '3min',
        window_params: tuple = (1, 6),
        bin_params: tuple = (1, 6),
        chg_type: str = 'px_chg'
    ) -> pd.DataFrame:
        """
        Engineers price reaction applying the double binned mean price change method for several window/bin combinations.

        Args:
            resample_type: 'median', 'mean', 'min', 'max', 'first', 'last', resampling price data to counter the effect of order book sweeps
            vol_window_params: (starting window size, ending window size), inclusive.
            merge_tolerance: The maximum number of minutes pd.merge_asof() can look 'backward' to find the nearest trade.
            window_params: (starting window size, ending window size), inclusive.
            bin_params: (starting number of trades per bin, ending number of trades per bin), inclusive.
            chg_type: 'pct_px_chg' or 'px_chg'.
        """
        if chg_type not in ['pct_px_chg', 'px_chg']:
            raise ValueError(f"Argument 'chg_type' must be 'pct_px_chg' or 'px_chg', not '{chg_type}'.")

        df = self._merge_frames(
            resample_type=resample_type,
            vol_window_params=vol_window_params,
            tolerance=merge_tolerance,
            chg_type=chg_type
        )

        homerun_indices = df.index[df['homerun_dummy'] == 1].to_numpy()
        if len(homerun_indices) == 0:
            return df[df['homerun_dummy'] == 1].reset_index(drop=True)

        n = len(df)
        price_arr = df['yes_price_dollars'].to_numpy(dtype=np.float64)
        homerun_arr = df['homerun_dummy'].to_numpy(dtype=np.int8)
        ticker_arr = df['ticker'].to_numpy()   
        
        rolling_vol_cols = [c for c in df.columns if 'rolling_' in c]
        vol_arrs = {c: df[c].to_numpy(dtype=np.float64) for c in rolling_vol_cols}
        
        w_range = range(window_params[0], window_params[1] + 1)
        b_range = range(bin_params[0], bin_params[1] + 1)

        processed_rows = []

        for idx in tqdm(homerun_indices, desc="Double Binned Method (Optimized)"):
            ticker = ticker_arr[idx]
            row_results = {'_original_idx': idx}

            for w in w_range:
                for b in b_range:
                    b1s = idx - w - b + 1   # before-bin start
                    b1e = idx - w           # before-bin end
                    b2s = idx + w           # after-bin start
                    b2e = idx + w + b - 1   # after-bin end

                    if b1s < 0 or b2e >= n:
                        continue

                    if homerun_arr[b1s : b2e + 1].sum() > 1:
                        continue

                    if (ticker_arr[b1s : b2e + 1] != ticker).any():
                        continue

                    avg_before = price_arr[b1s : b1e + 1].mean()
                    avg_after  = price_arr[b2s : b2e + 1].mean()

                    row_results[f'd_yes_price_w{w}_b{b}'] = avg_before

                    if chg_type == 'pct_px_chg':
                        row_results[f'd_pct_px_chg_w{w}_b{b}'] = (avg_after - avg_before) / avg_before
                    else:
                        row_results[f'd_px_chg_w{w}_b{b}'] = avg_after - avg_before

                    for col in rolling_vol_cols:
                        row_results[f'd_{col}_w{w}_b{b}'] = vol_arrs[col][b1s]

            processed_rows.append(row_results)

        features_df = pd.DataFrame(processed_rows).set_index('_original_idx')

        df_hr_only = df.loc[homerun_indices].copy()
        
        df_hr_only = df_hr_only.drop(columns=rolling_vol_cols)

        final_df = df_hr_only.join(features_df, how='left')

        return final_df
    
    def run_data_diagnostics(self):
        """
        Runs preliminary data diagnostics on the preprocessed data frames: trade_df, score_df, and homerun_df.
        """

        trade_df = self.trade_df.copy()
        score_df = self.score_df.copy()
        homerun_df = self.homerun_df.copy()

        print("="*10, "DATA FRAME DIAGNOSTICS FOR PROCESSED DATA", "="*10)

        print("     Trade Data Frame Information")
        print(trade_df.info())
        print("     Trade Data Frame Column Information")
        for col in trade_df.columns:
            nan_count = trade_df[col].isna().sum()
            nan_pct = trade_df[col].isna().mean() * 100
            print(" "*10, "Column: ", col)
            print(" "*15, f"Total NaN: {nan_count} / {len(trade_df[col])}")
            print(" "*15, f"Percent NaN: {nan_pct:.1f}%")
        print()

        print("     Score Data Frame Information")
        print(score_df.info())
        print("     Score Data Frame Column Information")
        for col in score_df.columns:
            nan_count = score_df[col].isna().sum()
            nan_pct = score_df[col].isna().mean() * 100
            print(" "*10, "Column: ", col)
            print(" "*15, f"Total NaN: {nan_count} / {len(score_df[col])}")
            print(" "*15, f"Percent NaN: {nan_pct:.1f}%")
        print()

        print("     Home Run Data Frame Information")
        print(homerun_df.info())
        print("     Home Run Data Frame Column Information")
        for col in homerun_df.columns:
            nan_count = homerun_df[col].isna().sum()
            nan_pct = homerun_df[col].isna().mean() * 100
            print(" "*10, "Column: ", col)
            print(" "*15, f"Total NaN: {nan_count} / {len(homerun_df[col])}")
            print(" "*15, f"Percent NaN: {nan_pct:.1f}%")
        print()

    def run_fe_diagnostics(
        self,
        generic_df: pd.DataFrame,
        forward_df: pd.DataFrame,
        double_df: pd.DataFrame
        ):
        """
        Runs data diagnostics on the feature engineered data frames: generic_df, forward_df, and double_df.
        """

        print()
        print("="*10, "DATA FRAME DIAGNOSTICS FOR FEATURE ENGINEERED DATA", "="*10)

        print()
        print("Generic Price Change Data Frame Column Information")
        for col in generic_df.columns:
            nan_count = generic_df[col].isna().sum()
            if nan_count == 0:
                continue
                
            nan_pct = generic_df[col].isna().mean() * 100
            print("     ", col)
            print("     ", f"Total NaN Values: {nan_count} / {len(generic_df[col])}")
            print("     ", f"Percent NaN Values: {nan_pct:.2f}%")
            print()

        print()
        print("Forward Binned Mean Price Change Data Frame Column Information")
        for col in forward_df.columns:
            nan_count = forward_df[col].isna().sum()
            if nan_count == 0:
                continue
                
            nan_pct = forward_df[col].isna().mean() * 100
            print("     ", col)
            print("     ", f"Total NaN Values: {nan_count} / {len(forward_df[col])}")
            print("     ", f"Percent NaN Values: {nan_pct:.2f}%")
            print()

        print()
        print("Double Binned Mean Price Change Data Frame Column Information")
        for col in double_df.columns:
            nan_count = double_df[col].isna().sum()
            if nan_count == 0:
                continue
                
            nan_pct = double_df[col].isna().mean() * 100
            print("     ", col)
            print("     ", f"Total NaN Values: {nan_count} / {len(double_df[col])}")
            print("     ", f"Percent NaN Values: {nan_pct:.2f}%")
            print()

    def download_data(
        self,
        data: pd.DataFrame,
        file_name: str,
        dir_path: str,
        compression: str
    ) -> None:
        """
        Downloads pandas DataFrame to Parquet file.
            
        Args:
            data: pandas DataFrame to save.
            file_name: Name of the parquet file.
            dir_path: Directory path where file will be saved.
        """
            
        file_path = dir_path + file_name
            
        if Path(file_path).exists():
            print(f"'{file_name}' already exists, skipping download.")
            return None
            
        try:
            dir = Path(dir_path)
            dir.mkdir(parents=True, exist_ok=True)
                
            data.to_parquet(file_path, index=False, compression=compression)
            print(f"'{file_name}' download complete.")
            return None
                
        except PermissionError:
            print(f"Error: No permission to write to {file_path}")
            raise
        except OSError as e:
            print(f"Error: Could not write file due to OS error: {e}")
            raise
        except Exception as e:
            print(f"Error: Unexpected error while saving {file_name}: {e}")
            raise


def data_pipeline(
    team_map: dict = {'LAA': 'Angels'},
    min_ts: int = 1742281200,
    resample_type: str = 'median',
    vol_window_params: tuple = (2, 50),
    merge_tolerance: str = '3min',
    trade_tolerance: str = '3min',
    window_params: tuple = (1, 6),
    bin_params: tuple = (1, 6),
    chg_type: str = 'px_chg',
):
    
    # Teams for Kalshi query
    teams = team_map.keys()

    kh = KalshiHistorical(read_limit=500, max_workers=5)

    # Get Kalshi markets
    recent_params = {
        "limit"             : 1000,
        "min_settled_ts"    : min_ts,
        "status"            : "settled",
        "series_ticker"     : "KXMLBGAME"
    }
    historical_params = {
        "limit"             : 1000,
        "series_ticker"     : "KXMLBGAME"
    }
    recent_market_data = kh.get_markets(
        params = recent_params,
        historical = False
    )
    historical_market_data = kh.get_markets(
        params = historical_params,
        historical = True
    )

    valid_historical_markets = []
    for market in historical_market_data:
        close_time_str = market.get("close_time")
        if close_time_str:
            close_time_unix = int(datetime.fromisoformat(close_time_str).timestamp())
            if close_time_unix >= min_ts:
                valid_historical_markets.append(market)

    historical_market_data = valid_historical_markets

    # Search for Kalshi tickers
    recent_tickers = None
    if recent_market_data:
        recent_tickers = kh.search_for_tickers(
            market_data = recent_market_data,
            keywords = teams,
            drop_duplicates = True
        )
    historical_tickers = None
    if historical_market_data:
        historical_tickers = kh.search_for_tickers(
            market_data = historical_market_data,
            keywords = teams,
            drop_duplicates = True
        )
    
    for team in teams:
        team_lower = team.lower()
        
        # --- Download Kalshi & mlbstatsapi Data ---
        raw_data_paths = {
            "raw_trade_data"   : f"../data/raw/{team_lower}_kalshi_trade_data.parquet",
            "raw_homerun_data" : f"../data/raw/{team_lower}_homerun_data.parquet",
            "raw_score_data"   : f"../data/raw/{team_lower}_score_data.parquet"
        }
        
        # Check if any raw data files are missing
        if not all(Path(file_path).exists() for file_path in raw_data_paths.values()):
            print()
            print(f"Downloading and processing Kalshi and mlbstatsapi data for {team}...")

            # Filter tickers by favoring 'yes' for team
            team_recent_tickers = None
            if recent_tickers:
                team_recent_tickers = [t for t in recent_tickers if t.split('-')[-1] == team]
            team_historical_tickers = None
            if historical_tickers:
                team_historical_tickers = [t for t in historical_tickers if t.split('-')[-1] == team]

            # Get Kalshi trade data
            recent_trade_data = {}
            if team_recent_tickers:
                recent_trade_data = kh.get_trades(
                    tickers = team_recent_tickers,
                    endpoint = '/markets/trades'
                )
            historical_trade_data = {}
            if team_historical_tickers:
                historical_trade_data = kh.get_trades(
                    tickers = team_historical_tickers,
                    endpoint = '/historical/trades'
                )
            trade_data = recent_trade_data | historical_trade_data
            
            # Download Kalshi trade data locally
            print()
            print("Downloading Kalshi trade data locally...")
            if trade_data:
                kh.download_game_trades(
                    game_trades = trade_data,
                    file_name = f"{team_lower}_kalshi_trade_data.parquet",
                    dir_path = "../data/raw/",
                    compression = 'gzip'
                )
            else:
                print(f"No trade data found for {team}. Continuing to next team...")
                continue

            # Get and validate dates for mlbstatsapi
            start_dt_str = None
            end_dt_str = None
            for k, v in trade_data.items():
                if not v: continue
                for trade in v:
                    created_time = trade.get('created_time')
                    if not created_time:
                        continue

                    if start_dt_str is None or created_time < start_dt_str:
                        start_dt_str = created_time
                    if end_dt_str is None or created_time > end_dt_str:
                        end_dt_str = created_time

            if not start_dt_str or not end_dt_str:
                print("No trade data found to determine date range.")
                return

            # 'YYYY-MM-DD' from the full ISO timestamp
            start_dt = start_dt_str[:10]
            end_dt = end_dt_str[:10]

            print(f"\nAuto-detected date range from Kalshi trades: {start_dt} to {end_dt}")
                    
            mlb = MLBData(
                team = team_map[team],
                start_dt = start_dt,
                end_dt = end_dt
            )

            # Get mlbstatsapi data
            hr_data_dict, score_data_dict = mlb.get_data()

            # Download home run data locally
            print("\nDownloading mlbstatsapi home run data locally...")
            mlb.download_homerun_data(
                hr_data_dict = hr_data_dict,
                file_name = f"{team_lower}_homerun_data.parquet",
                dir_path = "../data/raw/",
                compression = 'gzip'
            )

            # Download score data locally
            print("\nDownloading mlbstatsapi score data locally...")
            mlb.download_score_data(
                score_data_dict = score_data_dict,
                file_name = f"{team_lower}_score_data.parquet",
                dir_path = "../data/raw/",
                compression = 'gzip'
            )
        else:
            print(f"\nRaw data files already exist, skipping download for {team}.")

        # --- Feature Engineering ---
        processed_data_paths = {
            "generic_data" : f"../data/processed/{team_lower}_generic_data.parquet",
            "forward_data" : f"../data/processed/{team_lower}_forward_data.parquet",
            "double_data"  : f"../data/processed/{team_lower}_double_data.parquet"
        }
        
        # Check if any processed data files are missing
        if not all(Path(file_path).exists() for file_path in processed_data_paths.values()):
            print(f"\nRunning feature engineering for {team}...")
            fe = FeatureEngineering(
                trade_path = raw_data_paths["raw_trade_data"],
                score_path = raw_data_paths["raw_score_data"],
                homerun_path = raw_data_paths["raw_homerun_data"],
                team = team
            )

            # Processed data diagnostics
            fe.run_data_diagnostics()

            # Feature engineering using generic percent price change method
            print("\nFeature engineering generic percent price change method...")
            generic_df = fe.get_generic_df(
                resample_type=resample_type,
                vol_window_params=vol_window_params,
                merge_tolerance=merge_tolerance,
                trade_tolerance=trade_tolerance,
                chg_type=chg_type
            )

            # Feature engineering using forward binned mean percent price change method
            print("\nFeature engineering forward binned mean percent price change method...")
            forward_df = fe.get_forward_df(
                resample_type=resample_type,
                vol_window_params=vol_window_params,
                merge_tolerance=merge_tolerance,
                window_params=window_params,
                bin_params=bin_params,
                chg_type=chg_type
            )

            # Feature engineering using double binned mean percent price change method
            print("\nFeature engineering double binned mean percent price change method...")
            double_df = fe.get_double_df(
                resample_type=resample_type,
                vol_window_params=vol_window_params,
                merge_tolerance=merge_tolerance,
                window_params=window_params,
                bin_params=bin_params,
                chg_type=chg_type
            )

            # Feature engineered data diagnosticss
            fe.run_fe_diagnostics(generic_df=generic_df, forward_df=forward_df, double_df=double_df)

            # Download data locally
            print("\nDownloading generic data locally...")
            fe.download_data(
                data = generic_df,
                file_name = f"{team_lower}_generic_data.parquet",
                dir_path = "../data/processed/",
                compression = 'gzip'
            )
            del generic_df


            print("\nDownloading forward binned mean data locally...")
            fe.download_data(
                data = forward_df,
                file_name = f"{team_lower}_forward_data.parquet",
                dir_path = "../data/processed/",
                compression = 'gzip'
            )
            del forward_df


            print("\nDownloading double binned mean data locally...")
            fe.download_data(
                data = double_df,
                file_name = f"{team_lower}_double_data.parquet",
                dir_path = "../data/processed/",
                compression = 'gzip'
            )

            del double_df
            del fe.trade_df, fe.score_df, fe.homerun_df
            del fe
            gc.collect()
  
        else:
            print()
            print(f"Feature engineered data files already exist, skipping feature engineering for {team}.")