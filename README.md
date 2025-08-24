# NFL Survivor Pool Picking Strategy Tester and Optimizer
An interactive data science tool to simulate and identify optimal picking strategies in NFL Survivor Pools using historical and simulated betting data.


## Goal

This project aims to explore and identify the best picking strategies for NFL Survivor Pools. By simulating thousands of seasons using real historical data and bookmaker odds, the project evaluates different picking algorithms to determine which offers the highest average weeks survived and expected value in competitive pool settings.


## Data Sources

- **Historical NFL Odds and Results**: Collected from [Australia Sports Betting](https://www.aussportsbetting.com/data/historical-nfl-results-and-odds-data/)
- **NFL Schedules (Historical and Upcoming)**: Collected from [Pro Football Reference](https://www.pro-football-reference.com/years/2024/games.htm)
- **Upcoming Season Odds**: Retrieved via API from [the-odds-api](https://the-odds-api.com)

Datasets are not in this repo due to size. Vist the above sites to ingest data


## Project Structure
```plaintext
nfl_survivor_pool_picker/
│
├── README.md                       # This file
├── data/    #not included in repo
│   ├── historical_scores_txt_to_csv.ipynb        # Converts historical schedule csv's in string format from Pro Football Reference to CSVs
│   ├── upcoming_season_weeks_txt_to_csv.ipynb    # Does the same for the upcoming season
│   ├── cleaned_odds.csv                          # Real cleaned historical odds from Australia Sports Betting
│   ├── df_for_simulation.csv                     # Historical odds data processed for simulating
│   ├── historical_scores.csv                     # Historical schedules from pro football reference
│   ├── nfl.xlsx                                  # Raw odds data from Australia Sports Betting
│   ├── schedule_2025.csv                         # Upcoming season schedule
│   ├── simulated_nfl_histories                   # Simulations of real historical NFL seasons
│   └── simulated_upcoming_season                 # Simulations of the upcomng (2025) NFL seasons
│
├── notebooks/
│   ├── 01_Data_Wrangling.ipynb                           # EDA and preprocessing of historical odds and results
│   ├── 02_Synthetic_Historical_Data_Creation.ipynb       # Runs Monte Carlo simulations of real historical NFL seasons
│   ├── 03_Simulating_Upcoming_Season.ipynb               # Runs Monte Carlo simulations on the upcoming NFL season 
│   ├── 04_Average_Weeks_Survived.ipynb                   # Evaluates pickers on average weeks survived across datasets
│   └── 05_Comparing_Pickers_Head_to_Head.ipynb           # Refines head to head logic for Streamlit App
│
├── scripts/
│   └── PickerDefinitions.py          # Contains all strategy classes (pickers)
│
├── streamlit_app/
│   └── StreamlitApp.py              # Main Streamlit app file (interactive dashboard)
│   
└── outputs/
    ├── best_hyperparameters.txt         # Saved dictionary of best hyperparameters across strategies and datasets
    └── picker_performance_comparisons   # Final average weeks survived numbers across strategies after hyperparameter tuning
```


## Strategy Formulation Process

1. **Two Goals**: Maximize number of weeks survived and maximize winnings when playing against various strategies in a Survivor Pool.
2. **Picker Classes**:
    - `Picker`: Picks a random team each week
    - `BestOddsPicker`: Always picks the team with the best odds
    - `TopKOddsPicker`: Picks among the top K teams with best odds
    - `MaxOddsPicker`: Picks each week based on the combination of teams that maximizes survival probability over the entire season. Uses the Hungarian Algorithm to do this efficiently
    - `MaxOddsWithDecayPicker`: Same logic as above, but introduces a decay factor to put more weight on picking early weeks correctly
    - `SlidingWindowPicker`: Same logic as above, except that it maximizes the surivival probability over a predefined window of games, as opposed to the whole season, and picks according to that
3. **Evaluation Logic**:
    - Each picker is evaluated across different season datasets.
    - Pickers are measured on how many weeks they survive on average across thousands of simulations 
    - Pickers are also evaluated against one another on simulations based on their expected proportion of the pot they expect to win

## Final Results: Average Weeks Survived

| Picker                  | Average Number of Weeks Survived |
|-------------------------|----------------------------------|
| SlidingWindowPicker     | 4.477                            |
| BestOddsPicker          | 4.373                            |
| MaxOddsPicker           | 4.358                            |
| MaxOddsWithDecayPicker  | 4.355                            |
| TopKOddsPicker          | 3.481                            |
| Picker                  | 1.005                            |



## Running the Streamlit App

### Setup Instructions

```bash
# 1. Clone the repo
git clone https://github.com/cstec15/nfl_survivor_pool_picker.git
cd nfl_survivor_pool_picker

# 2. Ensure that datasources are ingested from above resources

# 3. Run the app
streamlit run streamlit_app/StreamlitApp.py
```

## Future...

1. Speed everything up
2. Integrate cloud storage for large datasets
3. Iteratively improve upon UI for Streamlit App
4. Add more picking strategies


## Author

**Christian Stec**  
Data Scientist  
[LinkedIn](https://linkedin.com/in/christian-stec) | [GitHub](https://github.com/cstec15)
