import pandas as pd
from ngboost.distns import Normal
from sklearn.tree import DecisionTreeRegressor
from utils.modeling_helpers import ngb_pipeline, xgb_pipeline


# --- Load in LAA Double Binned Data ---
laa_double_df = pd.read_parquet("../data/processed/laa_double_data.parquet")
laa_double_df = laa_double_df.drop(
    columns = [
        'ticker', 'time_pst', 'price', 'trade_count', 'pct_px_chg',
        'homerun_dummy','opp_homerun_dummy','score_delta_abs'
    ]
)
y_cols = [col for col in laa_double_df.columns if col.startswith("d_pct_px_chg")]
x_cols = [col for col in laa_double_df.columns if not col.startswith("d_pct_px_chg")]


# --- NGBoost Scorer, Base Models, & Parameter Grid ---
b1 = DecisionTreeRegressor(
    criterion='friedman_mse',
    max_depth=2, 
    min_samples_split=20,
    min_samples_leaf=10
)

b2 = DecisionTreeRegressor(
    criterion='friedman_mse',
    max_depth=3,
    min_samples_split=15, 
    min_samples_leaf=8
)

ngb_param_grid = {
    'Dist'           : [Normal],
    'Base'           : [b1, b2],
    'n_estimators'   : [50, 100, 150],
    'minibatch_frac' : [0.5, 0.8, 1.0],
    'learning_rate'  : [0.01, 0.02, 0.05]
}


# --- XGBoost Parameter Grid ---
xgb_param_grid = {
    'n_estimators'     : [50, 100, 200],
    # Number of trees in the random forest 
    'max_depth'        : [2, 3],                  
    # Maximum depth of a tree. 
    # Increasing this value will make the model more complex and more likely to overfit.
    'learning_rate'    : [0.01, 0.05, 0.1, 0.3],
    'min_child_weight' : [5, 10, 15],   
    # Corresponds to minimum number of instances needed to be in each node.
    # The larger min_child_weight is, the more conservative the algorithm will be.
    'colsample_bytree' : [0.5, 0.7],   
    # The subsample ratio of columns when constructing each tree.
    'reg_lambda'       : [10, 50, 100],
    # L2 regularization term on weights. Increasing this value will make model more conservative.
    'reg_alpha'        : [1, 5, 10],
    # L1 regularization term on weights. Increasing this value will make model more conservative.
    'gamma'            : [0.5, 1.0, 2.0]  
    # Minimum loss reduction required to make a further partition on a leaf node of the tree. 
    # The larger gamma is, the more conservative the algorithm will be.
}


def main():
    ngb_pipeline(
        y_cols=y_cols,
        x_cols=x_cols,   
        df=laa_double_df,
        n_threshold=50,
        param_grid=ngb_param_grid,
        cv=5,
        test_size=0.30,
        stratify_col='laa_homerun_dummy',
        random_state=42,
        n_jobs=-1,
        dir_path = "../models/ngb"
    )

    xgb_pipeline(
        y_cols=y_cols,
        x_cols=x_cols,   
        df=laa_double_df,
        n_threshold=50,
        param_grid=xgb_param_grid,
        scoring='neg_mean_absolute_error',
        n_iter=50,
        cv=5,
        test_size=0.30,
        stratify_col='laa_homerun_dummy',
        random_state=42,
        n_jobs=1,
        dir_path = "../models/xgb"
    )


if __name__ == "__main__":
    main()