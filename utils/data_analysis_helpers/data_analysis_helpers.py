import pandas as pd
import plotly.express as px

def plot_corr_matrix(
    df: pd.DataFrame,
    df_name: str,
    target_cols: list,
    feature_cols: list,
    team: str,
    team_colors: list
):

    corr_m = df[target_cols + feature_cols].corr()

    laa_diverging_colors = [
        [0.0, team_colors[0]],
        [0.5, team_colors[1]],
        [1.0, team_colors[2]]
    ]

    fig = px.imshow(
        corr_m,
        color_continuous_scale=laa_diverging_colors,
        zmin=-1, zmax=1,
        text_auto='.2f',
        aspect="auto"
    )

    fig.update_layout(
        template='plotly_dark',
        title={
            'text': f"<b>{df_name} Price Change Correlation Matrix ({team})</b>",
            'font': {'size': 22, 'color': team_colors[1]},
            'x': 0.5,
            'xanchor': 'center'
        },
        paper_bgcolor='#111111',
        plot_bgcolor='#111111',
        coloraxis_colorbar=dict(
            thicknessmode="pixels", thickness=15,
            lenmode="fraction", len=0.8,
        )
    )

    return fig