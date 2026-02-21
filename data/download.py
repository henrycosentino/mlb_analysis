from utils.data_handlers import data_pipeline

def main():
    data_pipeline(
        sport = 'mlb',
        teams = ["LAA", "LAD"],
        min_close_date = "2025-03-27",
        max_close_date = "2025-09-28", 
        resample_type = 'median',
        vol_window_params = (2, 50),
        merge_tolerance = '3min',
        trade_tolerance = '3min',
        window_params = (1, 6),
        bin_params = (1, 6),
        chg_type = 'pct_px_chg',
        homeruns_only = True
    )

if __name__ == '__main__':
    main()