import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from xgboost import XGBRegressor
from ngboost import NGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

def ngb_gscv(
    y_col: str,
    x_cols: list,
    df: pd.DataFrame,
    param_grid: dict,
    cv: int = 5,
    test_size: float = 0.30,
    stratify_col: str = 'laa_homerun_dummy',
    random_state: int = 42,
    n_jobs: int = 1
) -> dict:
    """
    Performs Grid Search CV to determine optimal NGBoost model.
    """

    X = df[x_cols]
    y = np.array(df[y_col])

    X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=random_state,
    stratify=df[stratify_col]
    )
    
    ngb = NGBRegressor(
        random_state=random_state,
        verbose=False
    )
    
    # Perform grid search
    grid_search = GridSearchCV(
        ngb, 
        param_grid=param_grid, 
        cv=cv,
        n_jobs=n_jobs,
        verbose=False,
        return_train_score=True,
    )
    
    grid_search.fit(X_train, y_train)

    # Predictions using best estimator
    best_ngb = grid_search.best_estimator_
    y_pred = best_ngb.predict(X_test)
    
    # Evaluate best estimator
    train_mse = mean_squared_error(y_train, best_ngb.predict(X_train))
    train_r2 = r2_score(y_train, best_ngb.predict(X_train))
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    return {
        'grid_search'      : grid_search,
        'best_estimator'   : best_ngb,
        'best_score'       : grid_search.best_score_,
        'best_params'      : grid_search.best_params_,
        'best_train_mse'   : train_mse,
        'best_train_r2'    : train_r2,
        'best_test_mse'    : test_mse,
        'best_test_r2'     : test_r2,
        'best_overfitting' : train_r2 - test_r2,
        'cv_results'       : pd.DataFrame(grid_search.cv_results_),
        'X_train'          : X_train,
        'X_test'           : X_test,
        'y_train'          : y_train,
        'y_test'           : y_test,
    }

def ngb_pipeline(
    y_cols: list,      
    x_cols: list,
    df: pd.DataFrame,
    n_threshold: int,
    param_grid: dict,
    cv: int = 5,
    test_size: float = 0.30,
    stratify_col: str = 'laa_homerun_dummy',
    random_state: int = 42,
    n_jobs: int = 1,
    dir_path: str = "../models/ngb"
):
    """
    Pipeline for NGBoost Grid Search CV over different Double Binned Mean Data targets.
    """
    
    print()
    print("Performing Grid Search Cross Validation for NGBoost...")
    for y_col in tqdm(y_cols, desc="Total Progress"):
        
        print()
        # Check if model already exists
        ngb_dir_path = Path(dir_path)
        ngb_dir_path.mkdir(parents=True, exist_ok=True)
        file_path = ngb_dir_path / f"model_{y_col}.pkl"
        if file_path.exists():
            print()
            print(f"Skipping '{y_col}': Optimal model and statistics already exist in system.")
            print()
            continue

        # Remove the price at home run features that do not correspond to the y_col
        w = y_col.split("_")[-2][1]
        b = y_col.split("_")[-1][1]
        base_features = [x for x in x_cols if ('d_yes_price' not in x) and ('d_rolling' not in x)]
        price_feature = f'd_yes_price_w{w}_b{b}'
        rolling_vol_features = [x for x in x_cols if 'd_rolling' in x and f'w{w}_b{b}' in x]
        x_cols_filtered = base_features + [price_feature] + rolling_vol_features
        df_temp = df[x_cols_filtered + [y_col]].dropna().reset_index(drop=True)
        
        # Check if there is enough data to train on
        if len(df_temp) < n_threshold:
            print()
            print(f"Skipping {y_col}: only {len(df_temp)} samples, minimum of {n_threshold} samples required.")
            print()
            continue

        print()
        print(f"Working on target: {y_col}...")
        results = ngb_gscv(
            y_col=y_col,
            x_cols=x_cols_filtered,
            df=df_temp,
            param_grid=param_grid,
            cv=cv,
            test_size=test_size,
            stratify_col=stratify_col,
            random_state=random_state,
            n_jobs=n_jobs
        )

        print(f"Downloading optimal model and statistics for {y_col}...")
        with file_path.open("wb") as f:
                pickle.dump(results, f)

    print("Finished Grid Search CV for NGBoost.")

def xgb_rscv(
    y_col: str,
    x_cols: list,
    df: pd.DataFrame,
    param_grid: dict,
    scoring: str = 'neg_mean_absolute_error',
    n_iter: int = 50,
    cv: int = 5,
    test_size: float = 0.30,
    stratify_col: str = 'laa_homerun_dummy',
    random_state: int = 42,
    n_jobs: int = 1
) -> dict:
    """
    Performs Randomized Search CV to determine optimal XGBoost model.
    """

    X = df[x_cols]
    y = np.array(df[y_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=df[stratify_col]
    )
    
    xgb = XGBRegressor(
        random_state=random_state, 
        n_jobs=n_jobs
    )
    
    # Perform grid search
    grid_search = RandomizedSearchCV(
        xgb, 
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=0,
        return_train_score=True,
    )
    
    grid_search.fit(X_train, y_train)

    # Predictions using best estimator
    best_xgb = grid_search.best_estimator_
    y_pred = best_xgb.predict(X_test)
    
    # Evaluate best estimator
    train_mse = mean_squared_error(y_train, best_xgb.predict(X_train))
    train_r2 = r2_score(y_train, best_xgb.predict(X_train))
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    return {
        'grid_search'      : grid_search,
        'best_estimator'   : best_xgb,
        'best_score'       : grid_search.best_score_,
        'best_params'      : grid_search.best_params_,
        'best_train_mse'   : train_mse,
        'best_train_r2'    : train_r2,
        'best_test_mse'    : test_mse,
        'best_test_r2'     : test_r2,
        'best_overfitting' : train_r2 - test_r2,
        'cv_results'       : pd.DataFrame(grid_search.cv_results_),
        'X_train'          : X_train,
        'X_test'           : X_test,
        'y_train'          : y_train,
        'y_test'           : y_test,
    }

def xgb_pipeline(
    y_cols: list,      
    x_cols: list,
    df: pd.DataFrame,
    n_threshold: int,
    param_grid: dict,
    scoring: str = 'neg_mean_absolute_error',
    n_iter: int = 50,
    cv: int = 5,
    test_size: float = 0.30,
    stratify_col: str = 'laa_homerun_dummy',
    random_state: int = 42,
    n_jobs: int = 1,
    dir_path: str = "../models/xgb"
):
    """
    Pipeline for XGBoost Randomized Search CV over different Double Binned Mean Data targets.
    """
    
    print()
    print("Performing Grid Search Cross Validation for XGBoost...")
    for y_col in tqdm(y_cols, desc="Total Progress"):
        
        print()
        # Check if model already exists
        xgb_dir_path = Path(dir_path)
        xgb_dir_path.mkdir(parents=True, exist_ok=True)
        file_path = xgb_dir_path / f"model_{y_col}.pkl"
        if file_path.exists():
            print()
            print(f"Skipping '{y_col}': Optimal model and statistics already exist in system.")
            print()
            continue
        
        # Remove the price at home run features that do not correspond to the y_col
        w = y_col.split("_")[-2][1]
        b = y_col.split("_")[-1][1]
        base_features = [x for x in x_cols if ('d_yes_price' not in x) and ('d_rolling' not in x)]
        price_feature = f'd_yes_price_w{w}_b{b}'
        rolling_vol_features = [x for x in x_cols if 'd_rolling' in x and f'w{w}_b{b}' in x]
        x_cols_filtered = base_features + [price_feature] + rolling_vol_features
        df_temp = df[x_cols_filtered + [y_col]].dropna().reset_index(drop=True)
        
        # Check if there is enough data to train on
        if len(df_temp) < n_threshold:
            print()
            print(f"Skipping {y_col}: only {len(df_temp)} samples, minimum of {n_threshold} samples required.")
            print()
            continue

        print()
        print(f"Working on target: {y_col}...")
        results = xgb_rscv(
            y_col=y_col,
            x_cols=x_cols_filtered,
            df=df_temp,
            param_grid=param_grid,
            scoring=scoring,
            n_iter=n_iter,
            cv=cv,
            test_size=test_size,
            stratify_col=stratify_col,
            random_state=random_state,
            n_jobs=n_jobs
        )

        print(f"Downloading optimal model and statistics for {y_col}...")
        with file_path.open("wb") as f:
                pickle.dump(results, f)

    print("Finished Grid Search CV for XGBoost.")