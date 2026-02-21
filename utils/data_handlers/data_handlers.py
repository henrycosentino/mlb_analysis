import pytz
import time
import requests
import mlbstatsapi
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from joblib import Parallel, delayed


# --- Kalshi API Downloader ---
class KalshiData:
    def __init__(self):
        self.markets_endpoint = "https://api.elections.kalshi.com/trade-api/v2/markets" 
        self.trades_endpoint = "https://api.elections.kalshi.com/trade-api/v2/markets/trades"
        
        self.series_tickers = {
            "mlb": "KXMLBGAME",
            "nfl": "KXNFLGAME",
            "nba": "KXNBAGAME"
            }
        
        self.sports_teams = {
            "mlb": [
                'SEA', 'TOR', 'LAD', 'MIL', 'PHI', 'CHC',
                'NYY', 'DET', 'CIN', 'BOS', 'CLE', 'SD',
                'ATH', 'ATL', 'BAL', 'TEX', 'WSH', 'SF',
                'AZ',  'MIA', 'TB',  'HOU', 'LAA', 'KC',
                'STL', 'MIN', 'PIT', 'CWS', 'COL', 'NYM'
            ],
            "nfl": [
                "MIN", "PIT"
            ],
            "nba": [
                "ATL", "BOS", "BKN", "CHA", "CHI", 
                "CLE", "DAL", "DEN", "DET", "GSW", 
                "HOU", "IND", "LAC", "LAL", "MEM", 
                "MIA", "MIL", "MIN", "NOP", "NYK", 
                "OKC", "ORL", "PHI", "PHX", "POR", 
                "SAC", "SAS", "TOR", "UTA", "WAS"
            ]
        }
        
        self.cities = {
            "LAA": "Los Angeles A",
            "SF": "San Francisco",
            "COL": "Colorado",
            "HOU": "Houston",
            "SEA": "Seattle",
            "TOR": "Toronto",
            "LAD": "Los Angeles D",
            "MIL": "Milwaukee",
            "PHI": "Philadelphia",
            "CHC": "Chicago C",
            "NYY": "New York Y",
            "DET": "Detroit",
            "CIN": "Cincinnati",
            "BOS": "Boston",
            "CLE": "Cleveland",
            "SD": "San Diego",
            "ATH": "A's",
            "ATL": "Atlanta",
            "BAL": "Baltimore",
            "TEX": "Texas",
            "WSH": "Washington",
            "AZ": "Arizona",
            "MIA": "Miami",
            "TB": "Tampa Bay",
            "KC": "Kansas City",
            "STL": "St. Louis",
            "MIN": "Minnesota",
            "PIT": "Pittsburgh",
            "CWS": "Chicago WS",
            "NYM": "New York M"
        }

    def date_to_timestamp(
        self, 
        date_string: str
    ) -> int:
        """
        Internal helper method that converts a user-friendly date string to Unix timestamp.
        """

        try:
            dt = datetime.strptime(date_string, "%m-%d-%Y")
        except ValueError:
            dt = datetime.strptime(date_string, "%Y-%m-%d")
            
        return int(dt.timestamp())


    def get_sports_markets(
        self, 
        sport: str = "mlb", 
        min_close_date: str = None,
        max_close_date: str = None,
        limit: int = 2
    ) -> list:
        """
        Gets all Kalshi's market data for a specific sports series.
        """
        
        params = {}

        if min_close_date:
            params['min_close_ts'] = self.date_to_timestamp(min_close_date)

        if max_close_date:
            params['max_close_ts'] = self.date_to_timestamp(max_close_date) + 86399

        if sport and sport in self.series_tickers.keys():
            params['series_ticker'] = self.series_tickers[sport]
        else:
            raise KeyError("No series sport key found.")

        pbar = tqdm(desc=f"Getting Kalshi {sport.upper()} Markets")
        all_markets = []
        cursor = None
        while limit > 0:
            if cursor:
                params["cursor"] = cursor
            
            response = requests.get(self.markets_endpoint, params=params, timeout=10)
            response.raise_for_status()
            time.sleep(0.5)
            data = response.json()

            try:
                all_markets.extend(data['markets'])
            except KeyError:
                raise KeyError("Markets data key error.")
            
            pbar.update(1)
            pbar.set_postfix_str(f"{len(all_markets)} markets fetched")
            limit -= 1

            try:
                cursor = data['cursor']
            except KeyError:
                raise KeyError("Pagination key error.")
            
            if not cursor:
                break
        pbar.close()
                
        return all_markets


    def get_game_tickers(
        self, 
        sport: str = "mlb", 
        team: str = "LAA", 
        team_opt: str = None, 
        min_close_date: str = None,
        max_close_date: str = None,
        limit: int = 2
    ) -> list:
        """
        Searches all Kashli's market data for a specific sport and returns that sports game tickers, along with the yes team.
        """
        
        teams_to_check = [team]
        if team_opt:
            teams_to_check.append(team_opt)

        if sport not in self.sports_teams:
            raise KeyError(f"Sport '{sport}' not found. Available sports: {list(self.sports_teams.keys())}")

        invalid_teams = [t for t in teams_to_check if t not in self.sports_teams[sport]]
        if invalid_teams:
            raise KeyError(f"Teams {invalid_teams} not found for sport '{sport}'.")
        
        sports_market_data = self.get_sports_markets(
            sport=sport, 
            min_close_date=min_close_date,
            max_close_date=max_close_date,
            limit=limit
            )

        game_tickers = []
        for market in tqdm(sports_market_data, desc=f"Getting Kalshi Game Tickers for {team}"):
            if all(t in market['ticker'] for t in teams_to_check):
                try:
                    ticker = market['ticker']
                except KeyError:
                    raise KeyError("Kalshi markets ticker data key error.")
                
                try:
                    yes_favor = market['yes_sub_title']
                except KeyError:
                    raise KeyError("Kalshi markets key error.")
                
                game_tickers.append((ticker, yes_favor))
                
        return game_tickers
    

    def _yes_team(
        self, 
        team: str = 'LAA', 
        game_tickers: list = None
    ) -> list:
        """
        Internal helper method that sorts for a specific market outcome for the tickers returned by search_sports_markets().
        """
        
        if team in self.cities.keys():
            city = self.cities[team]
            target_team_ls = []
            for data in game_tickers:
                if city == data[1]:
                    target_team_ls.append(data[0])

            return target_team_ls
        
        else:
            raise ValueError("Team must be a predefined team.")
        
    
    def get_game_trades(
        self, 
        game_tickers: list = None,
        sport: str = "mlb", 
        team: str = "LAA",
        team_opt: str = None, 
        min_close_date: str = None,
        max_close_date: str = None,
        limit: int = 2
    ) -> dict:
        """
        Gets a set of trade price data for one or more games.

        Arguments:
            game_tickers: a list of tickers in Kalshi format
            sport: the specific sport of interest
            team: the specific team of interest
            team_opt: the specific second team of interest (for a specific matchup)
            limit: the number of cursor pagination that will occur in get_sports_markets()
        """

        if game_tickers is None:
            game_tickers = self.get_sports_markets(
                sport=sport, 
                team=team, 
                team_opt=team_opt, 
                min_close_date=min_close_date,
                max_close_date=max_close_date,
                limit=limit
            )

        game_tickers = self._yes_team(
            team = team, 
            game_tickers = game_tickers
        )
        
        count = 0
        all_games_trades = {}
        for ticker in tqdm(game_tickers, desc=f"Getting Kalshi Game Trades Data for {team}"):
                
            game_trades = []
            params = {"ticker": ticker}
            cursor = None
            
            while True:
                    
                if cursor:
                    params["cursor"] = cursor

                response = requests.get(self.trades_endpoint, params=params, timeout=10)
                response.raise_for_status()
                game_data = response.json()

                try:
                    game_trades.extend(game_data['trades'])
                except KeyError:
                    raise KeyError("Trades data key error.")
                
                try:
                    cursor = game_data['cursor']
                except KeyError:
                    raise KeyError("Pagination key error.")
            
                if not cursor:
                    break
            
            count += 1

            all_games_trades[ticker] = game_trades
                        
        return all_games_trades
    
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
        """

        trades_list = []
        for ticker, trade_list in game_trades.items():
            trades_list.extend(trade_list)

        kalshi_path = dir_path + file_name

        if Path(kalshi_path).exists():
            print(f"'{file_name}' already exists in system, skipping download.")
            return None
        else:
            df = pd.DataFrame(trades_list)
            df.to_parquet(kalshi_path, index=False, compression=compression)
            print(f"'{file_name}' download complete.")
            return None
        

# --- mlbstatsapi Downloader ---
class MLBData:
    def __init__(
        self,
        team: str,
        start_dt: str,
        end_dt: str
    ):

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
        self.trade_path = trade_path
        self.score_path = score_path
        self.homerun_path = homerun_path
        self.team = team
        self.tz = pytz.timezone(tz)

        # Invariable attributes
        self.team_map = {
            "LAA": "Los Angeles Angels",
            "LAD": "Los Angeles Dodgers"
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
        """

        if 'time_pst' not in target_df.columns:
            raise ValueError("DataFrame must have a 'time_pst' column.")
        
        ticker_ls = []

        for time in target_df['time_pst']:
            time_date = time.date()
            found_ticker = False 
            
            for ticker in tickers:
                if len(ticker) < 17:
                    raise ValueError(f"Ticker format unexpected: {ticker}")
                # Parse date from ticker format (assuming format like "XXXXXXXXXX24DEC15XXX")
                date_str = ticker[10:17]
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
        Load and process trade data.
        """

        trade_df = pd.read_parquet(self.trade_path)

        # Validate required columns
        required_cols = ['ticker', 'created_time', 'price', 'count']
        missing_cols = [col for col in required_cols if col not in trade_df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}.")
        
        # Convert timestamp
        trade_df['time_pst'] = pd.to_datetime(
            trade_df['created_time'], 
            utc=True
        ).dt.tz_convert(self.tz)

        # Rename count column
        trade_df = trade_df.rename(columns={'count': 'trade_count'})
        
        trade_df = trade_df.sort_values('time_pst').reset_index(drop=True)
        
        return trade_df[['ticker', 'time_pst', 'price', 'trade_count']]
    
    def get_score_df(self) -> pd.DataFrame:
        """
        Load and process score data, calculating score differential.
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

        # Feature engineered data frames
        self.generic_df = None
        self.forward_df = None
        self.double_df = None

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

            if len(t) == 28:  # Ticker of the form "KXMLBGAME-25APR18LADTEX2-LAD" (doubleheader)
                team_one = t[17:20]
                team_two = t[20:23]
                if team_one == self.team:
                    opponent_ls.append(team_two)
                else:
                    opponent_ls.append(team_one)

            elif len(t) == 27: # Ticker of the form "KXMLBGAME-25APR16LAATEX-LAA"
                team_one = t[17:20]
                team_two = t[20:23]
                if team_one == self.team:
                    opponent_ls.append(team_two)
                else:
                    opponent_ls.append(team_one)

            elif len(t) == 26: # Ticker of the form "KXMLBGAME-25AUG05TBLAA-LAA" or "KXMLBGAME-25AUG05LAATB-LAA"
                split_one = [t[17:20], t[20:22]] # Team sub-string of the form 'LAATB'
                split_two = [t[17:19], t[19:22]] # Team sub-string of the form 'TBLAA'

                # Scans for a team sub-string of the form 'LAATB'
                if self.team in split_one:
                    opponent_ls.append(split_one[1])
                # Scans for a team sub-string of the form 'TBLAA'
                else:
                    opponent_ls.append(split_two[0])
            else:
                opponent_ls.append(None)
                print(f"Undefined handling for ticker '{t}' of length: {len(t)}")

        if (len(opponent_ls) != len(ticker_ls)):
            raise ValueError("Error calculating opponent team feature for home run data frame.")

        homerun_df['opponent'] = opponent_ls

        # Creates numeric dummies based on which opposing team hit a home run
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
        Essentially, it removes the effect of Order Book Sweeps by 'whales'.

        Args:
            resample_type: The way the resample is defined.

        """
    
        resample_types = ['median', 'mean', 'min', 'max', 'first', 'last']
        if resample_type not in resample_types:
            raise ValueError(f"'resample_type' must be one of: {resample_types}.")
        
        agg_funcs = {
            'price': resample_type,
            'trade_count': 'sum'
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
        window_types: list = ['time', 'trade']
    )-> dict:
        """
        Calculates rolling volatility feature for a given window.

        Args:
            trade_df_sorted: The trades data frame sorted by time.
            window_params: The starting and ending index of the windows size, inclusive.
            window_type: Can be set to either time or trade count.
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
        
        if 'price' not in trade_df_sorted.columns:
            raise ValueError(f"Column 'price' required for time-based rolling volatility. Found: {trade_df_sorted.columns.tolist()}.")
        
        trade_df_sorted['pct_px_chg'] = trade_df_sorted.groupby('ticker')['price'].pct_change().fillna(0)
        
        def _time(t, df):
            """Calculates rolling standard deviation, with window sizes based on time (minutes)."""
            key_name = f"rolling_std_{t}mins"
            new_col = (
                df.groupby('ticker', group_keys=False)[['pct_px_chg', 'time_pst']] 
                .rolling(
                    window=f'{t}min',
                    on='time_pst',
                    min_periods=2
                )['pct_px_chg']
                .std()
                .values
            )
            return key_name, new_col

        def _trade(t, df):
            """Calculates rolling standard deviation, with window sizes based on the trade count."""
            key_name = f"rolling_std_{t}trades" 
            new_col = (
                df.groupby('ticker', group_keys=False)['pct_px_chg']
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
                time_key, time_col = _time(t, trade_df_sorted)
                trade_key, trade_col = _trade(t, trade_df_sorted)
                new_cols[time_key] = time_col
                new_cols[trade_key] = trade_col

            elif 'time' in window_types and 'trade' not in window_types:
                time_key, time_col = _time(t, trade_df_sorted)
                new_cols[time_key] = time_col

            else:
                trade_key, trade_col = _trade(t, trade_df_sorted)
                new_cols[trade_key] = trade_col

        trade_df_sorted = pd.concat([trade_df_sorted, pd.DataFrame(new_cols, index=trade_df_sorted.index)], axis=1)

        return trade_df_sorted
    
    def _merge_frames(
        self,
        resample_type: str = 'median',
        vol_window_params: tuple = (2, 50),
        tolerance: str = '3min',
    ) -> pd.DataFrame:
        """
        Merges home run, trades, and score data frames using a 'backward' method to find the nearest trade/score by timestamp.

        Args:
            vol_window_params: (starting window size, ending window size), inclusive
            tolerance: The maximum number of minutes pd.merge_asof() can look 'backward' to find the nearest trade.
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
            window_types=['time', 'trade']
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
        
        # Sort score data frame and create score delta feature
        score_df_sorted = self._score_delta(self.score_df.copy())
        score_df_sorted = score_df_sorted.sort_values('time_pst').reset_index(drop=True)

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

        final_df = final_df.dropna(subset=['price']).reset_index(drop=True)
        
        return final_df
    
    def get_generic_df(
        self,
        resample_type: str = 'median',
        vol_window_params: tuple = (2, 50),
        merge_tolerance: str = '3min',
        trade_tolerance: str = '3min',
        chg_type: str = 'pct_px_chg'
    ) -> pd.DataFrame:
        """
        Engineers price reaction applying the generic price change method.

        Args:
            vol_window_params: (starting window size, ending window size), inclusive
            merge_tolerance: The maximum number of minutes pd.merge_asof() can look 'backward' to find the nearest trade.
            trade_tolerance: The maximum number of minutes the next trade can be away from the home run trade time.
            chg_type: 'pct_px_chg' or 'px_chg'.
        """
        
        if chg_type not in ['pct_px_chg', 'px_chg']:
            raise ValueError(f"Argument 'chg_type' must be 'pct_px_chg' or 'px_chg', not '{chg_type}'.")

        # Merge home run, trades, and score data frames
        df = self._merge_frames(
            resample_type=resample_type,
            vol_window_params=vol_window_params,
            tolerance=merge_tolerance
        )

        # Create temporary columns for mask 
        df['_next_price'] = df['price'].shift(-1)
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
            df[f'g_{chg_type}'] = np.where(valid, (df['_next_price'] - df['price']) / df['price'], np.nan)
        else:
            df[f'g_{chg_type}'] = np.where(valid, df['_next_price'] - df['price'], np.nan)

        # Drop temporary columns
        df = df.drop(columns=['_next_price', '_next_time', '_next_ticker', '_next_homerun'])

        self.generic_df = df

        return df
    
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
        Engineers price reaction applying the forward binned mean price change method for several window/bin combinations.

        Args:
            vol_window_params: (starting window size, ending window size), inclusive.
            merge_tolerance: The maximum number of minutes pd.merge_asof() can look 'backward' to find the nearest trade.
            window_params: (starting window size, ending window size), inclusive.
            bin_params: (starting number of trades per bin, ending number of trades per bin), inclusive.
            chg_type: 'pct_px_chg' or 'px_chg'.
        """

        if chg_type not in ['pct_px_chg', 'px_chg']:
            raise ValueError(f"Argument 'chg_type' must be 'pct_px_chg' or 'px_chg', not '{chg_type}'.")

        # Merge home run, trades, and score data frames
        df = self._merge_frames(
            resample_type=resample_type,
            vol_window_params=vol_window_params,
            tolerance=merge_tolerance
        )

        # Initialize forward binned columns with NaN values
        nan_dict = {
            f'f_{chg_type}_w{w}_b{b}': [np.nan] * len(df)
            for w in range(window_params[0], window_params[1] + 1) 
            for b in range(bin_params[0], bin_params[1] + 1)
        }
        nan_df = pd.DataFrame(
            data = nan_dict,
            index = range(0, len(df))
        )
        df = df.join(nan_df, how='left')

        # Calculate forward binned method for home run events
        homerun_df = df[df['homerun_dummy'] == 1].copy()
        for idx, row in tqdm(homerun_df.iterrows(), total=len(homerun_df)):
            homerun_px = row['price']
            
            for w in range(window_params[0], window_params[1] + 1):
                for b in range(bin_params[0], bin_params[1] + 1):
                    start_idx = idx + w
                    end_idx = idx + w + b - 1

                    # Out of bounds check
                    if end_idx >= len(df):
                        continue

                    # Ensure there is not another home run event in the window
                    region = df.loc[idx:end_idx]
                    if region['homerun_dummy'].sum() > 1:
                        continue
                    
                    forward_bin = df.loc[start_idx:end_idx]

                    # Ensure we only get data from the same ticker
                    if set(forward_bin['ticker']) != {row['ticker']}:
                        continue

                    avg_px_forward_bin = forward_bin['price'].mean()

                    # Assign price change
                    if chg_type == 'pct_px_chg':
                        df.loc[idx, f'f_pct_px_chg_w{w}_b{b}'] = (avg_px_forward_bin - homerun_px) / homerun_px
                    else:
                        df.loc[idx, f'f_px_chg_w{w}_b{b}'] = avg_px_forward_bin - homerun_px

        self.forward_df = df

        return df

    def get_double_df(
        self,
        resample_type: str = 'median',
        vol_window_params: tuple = (2, 50),
        merge_tolerance: str = '3min',
        window_params: tuple = (1, 6),
        bin_params: tuple = (1, 6),
        chg_type: str = 'pct_px_chg'
    ) -> pd.DataFrame:
        """
        Engineers price reaction applying the double binned mean price change method for several window/bin combinations.

        Args:
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
            tolerance=merge_tolerance
        )

        # Pre-extract data
        n = len(df)
        price_arr = df['price'].to_numpy(dtype=np.float64)
        homerun_arr = df['homerun_dummy'].to_numpy(dtype=np.int8)
        ticker_arr = df['ticker'].to_numpy()   
        rolling_vol_cols = [c for c in df.columns if 'rolling_' in c]
        vol_arrs = {c: df[c].to_numpy(dtype=np.float64) for c in rolling_vol_cols}
        w_range = range(window_params[0], window_params[1] + 1)
        b_range = range(bin_params[0], bin_params[1] + 1)
        homerun_indices = df.index[df['homerun_dummy'] == 1].tolist()

        # Processes a single home run event
        def process_event(idx):
            """Returns a flat dict for one home run row."""
            ticker = ticker_arr[idx]
            row_results = {}

            for w in w_range:
                for b in b_range:
                    b1s = idx - w - b + 1   # before-bin start
                    b1e = idx - w           # before-bin end (inclusive)
                    b2s = idx + w           # after-bin start
                    b2e = idx + w + b - 1   # after-bin end (inclusive)

                    # Bounds check
                    if b1s < 0 or b2e >= n:
                        continue

                    # Ensure there is not a second home run in the entire region
                    if homerun_arr[b1s : b2e + 1].sum() > 1:
                        continue

                    # Ensure price data is from the same game (ticker)
                    if (ticker_arr[b1s : b1e + 1] != ticker).any() or \
                    (ticker_arr[b2s : b2e + 1] != ticker).any():
                        continue

                    # Compute before and after prices
                    avg_before = price_arr[b1s : b1e + 1].mean()
                    avg_after  = price_arr[b2s : b2e + 1].mean()

                    row_results[f'd_price_w{w}_b{b}'] = avg_before

                    if chg_type == 'pct_px_chg':
                        row_results[f'd_pct_px_chg_w{w}_b{b}'] = (avg_after - avg_before) / avg_before
                    else:
                        row_results[f'd_px_chg_w{w}_b{b}'] = avg_after - avg_before

                    for col, arr in vol_arrs.items():
                        row_results[f'd_{col}_w{w}_b{b}'] = arr[b1s] # First element of before bin

            return idx, row_results

        # Run processes in parallel
        raw_results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(process_event)(idx)
            for idx in tqdm(homerun_indices)
        )

        # Build pre-allocated arrays (initalised to NaN)
        nan_px_dict = {
            f'd_price_w{w}_b{b}': np.full(n, np.nan)
            for w in w_range for b in b_range
        }
        nan_chg_dict = {
            f'd_{chg_type}_w{w}_b{b}': np.full(n, np.nan)
            for w in w_range for b in b_range
        }
        nan_vol_dict = {
            f'd_{col}_w{w}_b{b}': np.full(n, np.nan)
            for w in w_range for b in b_range for col in rolling_vol_cols
        }
        all_new_cols = nan_px_dict | nan_chg_dict | nan_vol_dict

        # Fill computed values into the pre-allocated arrays
        for idx, row_results in raw_results:
            for col, val in row_results.items():
                all_new_cols[col][idx] = val

        # Attach new columns
        new_cols_df = pd.DataFrame(all_new_cols, index=df.index)
        df = pd.concat([df, new_cols_df], axis=1)

        # Drop old rolling volatility columns
        df = df.drop(columns=rolling_vol_cols)

        self.double_df = df

        return df
    
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
        generic_df: None,
        forward_df: None,
        double_df: None
        ):
        """
        Runs data diagnostics on the feature engineered data frames: generic_df, forward_df, and double_df.
        """

        print()
        print("="*10, "DATA FRAME DIAGNOSTICS FOR FEATURE ENGINEERED DATA", "="*10)

        if generic_df is None and self.generic_df is not None:
            generic_df = self.generic_df.copy()
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

        if forward_df is None and self.forward_df is not None:
            forward_df = self.forward_df.copy()
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

        if double_df is None and self.double_df is not None:
            double_df = self.double_df.copy()
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
    sport: str = 'mlb',
    teams: list = ['LAA'],
    min_close_date: str = "2025-03-27",
    max_close_date: str = "2025-09-28",
    resample_type: str = 'median',
    vol_window_params: tuple = (2, 50),
    merge_tolerance: str = '3min',
    trade_tolerance: str = '3min',
    window_params: tuple = (1, 6),
    bin_params: tuple = (1, 6),
    chg_type: str = 'pct_px_chg',
    homeruns_only: bool = True
):
    
    # Team name mapping for mlbstatsapi
    team_name_map = {
        'LAA': 'Angels',
        'LAD': 'Dodgers',
    }
    
    for team in teams:
        team_lower = team.lower()
        
        # ======   Download Kalshi & mlbstatsapi Data  =======
        raw_data_paths = {
            "raw_trade_data"   : f"../data/raw/{team_lower}_kalshi_trade_data.parquet",
            "raw_homerun_data" : f"../data/raw/{team_lower}_homerun_data.parquet",
            "raw_score_data"   : f"../data/raw/{team_lower}_score_data.parquet"
        }
        
        # Check if any raw data files are missing
        if not all(Path(file_path).exists() for file_path in raw_data_paths.values()):
            print()
            print(f"Downloading and processing Kalshi and mlbstatsapi data for {team}...")
            kd = KalshiData()
            
            # Get Kalshi game tickers
            tickers = kd.get_game_tickers(
                sport = sport, 
                team = team, 
                team_opt = None,
                min_close_date = min_close_date,
                max_close_date = max_close_date,
                limit = 250
            )
            
            # Get Kalshi trades for game tickers
            game_trades = kd.get_game_trades(
                game_tickers = tickers,
                sport = sport, 
                team = team, 
                team_opt = None,
                min_close_date = min_close_date,
                max_close_date = max_close_date,
                limit = 250
            )
            
            # Download Kalshi trades locally
            print()
            print("Downloading Kalshi trades locally...")
            kd.download_game_trades(
                game_trades = game_trades,
                file_name = f"{team_lower}_kalshi_trade_data.parquet",
                dir_path = "../data/raw/",
                compression = 'gzip'
            )

            # Get and validate date inputs for mlbstatsapi
            while True:
                print()
                print("Match dates from Kalshi data to mlbstatsapi data...")
                print(f"First available game ticker from Kalshi: {tickers[-1]}")
                start_dt_input = input("Enter the date for first available game (YYYY-MM-DD): ")
                print(f"Last available game ticker from Kalshi: {tickers[0]}")
                end_dt_input = input("Enter the date for last available game (YYYY-MM-DD): ")
                try:
                    start_dt = datetime.strptime(start_dt_input, "%Y-%m-%d")
                    end_dt = datetime.strptime(end_dt_input, "%Y-%m-%d")
                    
                    if start_dt > end_dt:
                        print("Start date must be before end date.")
                        print()
                        continue
                    
                    break
                except ValueError:
                    print("Incorrect date format, try again using (YYYY-MM-DD).")
                    print()
            
            mlb = MLBData(
                team = team_name_map[team],
                start_dt = start_dt_input,
                end_dt = end_dt_input
            )

            # Get mlbstatsapi data
            hr_data_dict, score_data_dict = mlb.get_data()

            # Download home run data locally
            print()
            print("Downloading mlbstatsapi home run data locally...")
            mlb.download_homerun_data(
                hr_data_dict = hr_data_dict,
                file_name = f"{team_lower}_homerun_data.parquet",
                dir_path = "../data/raw/",
                compression = 'gzip'
            )

            # Download score data locally
            print()
            print("Downloading mlbstatsapi score data locally...")
            mlb.download_score_data(
                score_data_dict = score_data_dict,
                file_name = f"{team_lower}_score_data.parquet",
                dir_path = "../data/raw/",
                compression = 'gzip'
            )
        else:
            print()
            print(f"Raw data files already exist, skipping download for {team}.")

        # ==========  Feature Engineering  ===========
        processed_data_paths = {
            "generic_data" : f"../data/processed/{team_lower}_generic_data.parquet",
            "forward_data" : f"../data/processed/{team_lower}_forward_data.parquet",
            "double_data"  : f"../data/processed/{team_lower}_double_data.parquet"
        }
        
        # Check if any processed data files are missing
        if not all(Path(file_path).exists() for file_path in processed_data_paths.values()):
            print()
            print(f"Running feature engineering for {team}...")
            fe = FeatureEngineering(
                trade_path = raw_data_paths["raw_trade_data"],
                score_path = raw_data_paths["raw_score_data"],
                homerun_path = raw_data_paths["raw_homerun_data"],
                team = team
            )

            # Processed data diagnostics
            fe.run_data_diagnostics()

            # Feature engineering using generic percent price change method
            print()
            print("Feature engineering generic percent price change method...")
            generic_df = fe.get_generic_df(
                resample_type=resample_type,
                vol_window_params=vol_window_params,
                merge_tolerance=merge_tolerance,
                trade_tolerance=trade_tolerance,
                chg_type=chg_type
            )

            # Feature engineering using forward binned mean percent price change method
            print()
            print("Feature engineering forward binned mean percent price change method...")
            forward_df = fe.get_forward_df(
                resample_type=resample_type,
                vol_window_params=vol_window_params,
                merge_tolerance=merge_tolerance,
                window_params=window_params,
                bin_params=bin_params,
                chg_type=chg_type
            )

            # Feature engineering using double binned mean percent price change method
            print()
            print("Feature engineering double binned mean percent price change method...")
            double_df = fe.get_double_df(
                resample_type=resample_type,
                vol_window_params=vol_window_params,
                merge_tolerance=merge_tolerance,
                window_params=window_params,
                bin_params=bin_params,
                chg_type=chg_type
            )

            # Feature engineered data diagnostics
            if homeruns_only: # If we want only home run data
                generic_df = generic_df[generic_df['homerun_dummy'] == 1].reset_index(drop=True)
                forward_df = forward_df[forward_df['homerun_dummy'] == 1].reset_index(drop=True)
                double_df = double_df[double_df['homerun_dummy'] == 1].reset_index(drop=True)
                fe.run_fe_diagnostics(generic_df=generic_df, forward_df=forward_df, double_df=double_df)
            else: # Otherwise, we want all the data
                fe.run_fe_diagnostics()

            # Download data locally
            print()
            print("Downloading generic data locally...")
            fe.download_data(
                data = generic_df,
                file_name = f"{team_lower}_generic_data.parquet",
                dir_path = "../data/processed/",
                compression = 'gzip'
            )

            print()
            print("Downloading forward binned mean data locally...")
            fe.download_data(
                data = forward_df,
                file_name = f"{team_lower}_forward_data.parquet",
                dir_path = "../data/processed/",
                compression = 'gzip'
            )

            print()
            print("Downloading double binned mean data locally...")
            fe.download_data(
                data = double_df,
                file_name = f"{team_lower}_double_data.parquet",
                dir_path = "../data/processed/",
                compression = 'gzip'
            )
            
        else:
            print()
            print(f"Feature engineered data files already exist, skipping feature engineering for {team}.")