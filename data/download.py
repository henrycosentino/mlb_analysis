from utils.data import data_pipeline

team_name_map = {
    'LAA': 'Angels',
    'LAD': 'Dodgers',
    'SF' : 'Giants',
    'CWS': 'White Sox',
    'CHC': 'Cubs',   
}

def main():
    data_pipeline(
        team_map = team_name_map,
        min_ts = 1742281200,          # March 18, 2025
        resample_type = 'median',
        vol_window_params = (2, 20),
        merge_tolerance = '3min',
        trade_tolerance = '3min',
        window_params = (1, 6),
        bin_params = (1, 6),
        chg_type = 'px_chg'
    )

if __name__ == '__main__':
    main()