import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go

def get_model_stats(
    df: pd.DataFrame,
    df_type: str,
    y_cols: list,
    X_cols: list
) -> list:
    """
    Fits regression models for multiple dependent variables and returns coefficients, p-values, and R-squared.
    """
    
    model_stats = []

    for col in y_cols:

        if df_type == 'double':
            w, b = col.split('_')[-2], col.split('_')[-1]
            px_features_to_remove = [
                c for c in df.columns 
                if 'd_price' in c
                if c != f'd_price_{w}_{b}'
            ]
            filtered_X_cols = [x for x in X_cols if x not in px_features_to_remove]
            subset_df = df.dropna(subset=[col] + filtered_X_cols)
            current_X_cols = filtered_X_cols 
        else:
            subset_df = df.dropna(subset=[col] + X_cols)
            current_X_cols = X_cols

        if subset_df.empty: 
            continue

        if subset_df[col].std() == 0 or len(subset_df[col].unique()) == 1:
            print(f"Skipping {col}: No variance in dependent variable (all values are identical)")
            continue
        
        else:   
            X1 = subset_df[current_X_cols]
            X1 = sm.add_constant(X1)
            y1 = subset_df[col]

            model = sm.OLS(y1, X1)
            result = model.fit() 

            beta_dict = {f"beta_{k}": v for k, v in dict(result.params).items()}
            p_value_dict = {f"p_value_{k}": v for k, v in dict(result.pvalues).items()}

            if df_type in ['forward', 'double']:
                w, b = int(col.split('_')[-2][1]), int(col.split('_')[-1][1])

                model_stats.append({
                    'window': w,
                    'bin': b,
                    'r_squared': result.rsquared,
                    **beta_dict,  
                    **p_value_dict    
                })
            else:
                model_stats.append({
                    'r_squared': result.rsquared,
                    **beta_dict,  
                    **p_value_dict 
                })   

    return model_stats

def graph_model_stats(
    df: pd.DataFrame,
    stat_type: str = 'beta',
    title: str = 'Model Coefficient',
    z: str = 'laa_homerun_dummy',
    gradient: str = 'p_value',
    team: str = 'LAA'
):
    """
    Creates a 3D scatter plot visualizing regression statistics across different window and quantile combinations.
    """
    
    LAA_RED = '#BA0021'
    LAD_BLUE = '#005A9C'
    SILVER = '#C4CED4'
    
    accent_color = LAA_RED if team.upper() == 'LAA' else LAD_BLUE
    custom_scale = [[0, SILVER], [1, accent_color]]

    if stat_type == 'beta':
        z_col = f'beta_{z}'
    else:
        z_col = 'r_squared'

    if gradient == 'p_value':
        gradient_col = f'p_value_{z}'
        gradient_title = 'P-value'
        custom_scale = [[0, accent_color], [1, SILVER]]
    else:
        gradient_col = 'r_squared'
        gradient_title = 'R²'

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=df['window'],
        y=df['bin'],
        z=df[z_col],
        mode='markers',
        marker=dict(
            size=4,
            color=df[gradient_col],
            colorscale=custom_scale,
            opacity=0.9,
            colorbar=dict(
                title=gradient_title,
                thickness=15,
                tickfont=dict(color=SILVER)
            ),
            showscale=True
        )
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#111111',
        title=dict(
            text=f'<b>{title} for Different Targets ({team})</b>',
            font={'size': 24, 'color': SILVER},
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(title='Window Size (α)', gridcolor='rgb(60, 60, 60)', showbackground=False),
            yaxis=dict(title='Bin Size (β)', gridcolor='rgb(60, 60, 60)', showbackground=False),
            zaxis=dict(title=z, gridcolor='rgb(60, 60, 60)', showbackground=False),
            bgcolor='#111111',
            aspectmode='cube'
        ),
        width=1000,
        height=700,
        margin=dict(l=0, r=0, b=0, t=60),
        showlegend=False
    )

    return fig