## Project Overview

This project aims to determine the effect a home run has on price in sports trading markets and develop a trading system to exploit any observable effects. The reader should begin with the [data analysis notebook](https://github.com/henrycosentino/mlb_analysis/blob/main/notebooks/data_analysis.ipynb), which explores the nuances in the data. From there, they should move to the [hypothesis testing notebook](https://github.com/henrycosentino/mlb_analysis/blob/main/notebooks/hypothesis_testing.ipynb), which contains in-depth regression analysis testing the effect of a home run on price in sports trading markets. Next, the reader should make their way to the [predictive modeling notebook](https://github.com/henrycosentino/mlb_analysis/blob/main/notebooks/predictive_modeling.ipynb), which walks through the machine learning models developed to predict the price of a sports trading market after a home run has been hit. Lastly, the reader is encouraged to explore the [flight path notebook](https://github.com/henrycosentino/mlb_analysis/blob/main/notebooks/flight_path.ipynb), which uses the [home run algorithm](https://github.com/henrycosentino/mlb_analysis/blob/main/utils/homerun/homerun.py) used to determine whether a batted baseball is a home run.

### Hypothesis

- When the home team hits a home run, the price (probability of the home team winning) increases.

- When the opponent hits a home run, the price (probability of the home team winning) decreases.

- Two home teams were considered: the Los Angeles Angels (LAA) and the Los Angeles Dodgers (LAD).

### Trading Idea

Determine whether a home run has been hit before the ball lands. A colocated camera system at an MLB stadium could supply the necessary model parameters for the [home run algorithm](https://github.com/henrycosentino/mlb_analysis/blob/main/utils/homerun/homerun.py) to instantaneously determine if a home run was hit as the ball leaves the bat. Finally, a [predictive model](https://github.com/henrycosentino/mlb_analysis/blob/main/notebooks/predictive_modeling.ipynb) could estimate the expected price change in the sports trading market to inform trading decisions.

**Exploit three inefficiencies:**

1. **Location**: Being local to the ballpark allows for faster trading than watching remotely.

2. **Time**: Being able to place a trade faster via a more efficient trading system.

3. **Outcome**: Being able to model whether a home run is hit when the ball is coming off the bat.

### Project Findings

The analysis [finds](https://github.com/henrycosentino/mlb_analysis/blob/main/notebooks/hypothesis_testing.ipynb) there is a positive relationship between LAA hitting a home run and price change.

### Data Sources

- **[MLB Stats API](https://statsapi.mlb.com/)**: Baseball game data

- **[Kalshi](https://kalshi.com/)**: Sports trading market prices and data


## Setup

To fully view the analysis, you need to install the project and its dependencies.

### 1. Clone the Repository
```bash
git clone https://github.com/henrycosentino/mlb_analysis.git
cd mlb_analysis

```

### 2. Install the Project
```bash
pip install -e .

```

### 3. Explore!