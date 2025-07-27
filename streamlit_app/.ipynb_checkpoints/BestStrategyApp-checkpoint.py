import streamlit as st
import pandas as pd
from collections import defaultdict
import sys
import os
sys.path.append(os.path.abspath("scripts"))

def play_survival_pool_season(players, season_df):
    """
    Simulates a survival pool season for a list of picker strategies.

    Parameters:
        players (list): List of picker classes (not instances) to evaluate.
        season_df (pd.DataFrame): DataFrame containing season data, including weekly games and results.

    Returns:
        tuple:
            - best_players (list): Picker class names that survived the most weeks (can be a tie).
            - best_weeks (int): Number of weeks survived by the top performer(s).
            - performances (dict): Mapping of picker class names to weeks survived.
    """
    best_weeks = 0
    best_players = []
    performances = {}
    num_each_player = {}
    for pkr in players:
        
        if pkr.__name__ not in num_each_player:
            pkr_name = pkr.__name__ + str(1)
            num_each_player[pkr.__name__] = 1
        else:
            pkr_name = pkr.__name__ + str(num_each_player[pkr.__name__] + 1)
            num_each_player[pkr.__name__] += 1
            
        this_pkr = pkr(season_df)
        this_pkr.make_season_picks()
        performance = this_pkr.evaluate_performance()
        if performance == best_weeks:
            best_players.append(pkr_name)
        elif performance > best_weeks:
            best_players = [pkr_name]
            best_weeks = performance
        performances[pkr_name] = performance

    return best_players, best_weeks, performances


def count_picker_wins(players, multi_season_df):
    """
    Evaluates picker strategies over multiple seasons and summarizes win statistics.

    Parameters:
        players (list): List of picker classes to simulate.
        multi_season_df (pd.DataFrame): Combined season data with a 'Season' column for grouping.

    Returns:
        dict:
            - 'Wins or Ties': Count of seasons each picker either won or tied for first.
            - 'Outright Wins': Count of seasons each picker won outright.
            - 'Expected Value Per Season ($1 Pool)': Picker’s expected earnings from a $1 winner(s) take all pool
    """
    win_or_tie_counts = defaultdict(int)
    outright_win_counts = defaultdict(int)
    expected_value = defaultdict(float) # this will include each strategies expected proportion of the winnings each season. 
    # If this is greater than 1/num_players, it would be considered a winning strategy

    # Group data by season for efficiency
    season_groups = multi_season_df.groupby('Season')
    num_seasons = len(season_groups)
    for _, season_df in season_groups:
        best_players, _, _ = play_survival_pool_season(players, season_df)

        # Increment win count for each winner (handles ties)
        num_winners = len(best_players)
        for p in best_players:
            win_or_tie_counts[p] += 1
            expected_value[p] += 1 / num_winners / num_seasons
        if num_winners == 1:
            outright_win_counts[best_players[0]] += 1

    return {'Wins or Ties': dict(win_or_tie_counts), 
            'Outright Wins': dict(outright_win_counts), 
            'Expected Value Per Season ($1 Pool)': dict(expected_value)}

def calculate_edge(expected_value_dict):
    """
    Calculates the edge the best picker has over the average expected value.
    
    Parameters:
    -----------
    expected_value_dict : dict
        Dictionary mapping picker names to their expected value (e.g., {'PickerA': 0.25, 'PickerB': 0.18, ...})
    
    Returns:
    --------
    dict with:
        - 'best_picker': name of the picker with the highest expected value
        - 'best_value': the expected value of the best picker
        - 'average_value': average expected value if all pickers were equal (1 / number of pickers)
        - 'edge': the difference between the best picker's EV and the average
    """
    num_pickers = len(expected_value_dict)
    average_value = 1 / num_pickers
    best_picker = max(expected_value_dict, key=expected_value_dict.get)
    best_value = expected_value_dict[best_picker]
    edge = best_value - average_value

    return {
        'best_picker': best_picker,
        'best_value': best_value,
        'average_value': average_value,
        'edge': edge
    }



# Import picker classes
from PickerDefinitions import (
    BestOddsPicker,
    TopKOddsPicker,
    MaxOddsPicker,
    MaxOddsWithDecayPicker,
    SlidingWindowPicker,
    Picker
)


simulated_upcoming_season = pd.read_csv("data/simulated_upcoming_season")

# Title
st.title("NFL Survivor Pool Optimal Strategy Picker")
st.write("Select the strategies you expect your opponents to be using")
st.write("Based on simulations of the 2025 season, the best picking strategy will be returned")

# Picker options
all_pickers = [
    Picker,
    BestOddsPicker,
    TopKOddsPicker,
    MaxOddsPicker,
    MaxOddsWithDecayPicker,
    SlidingWindowPicker
]
picker_options = {picker.__name__: picker for picker in all_pickers}

opponents = []
for name, cls in picker_options.items():
    count = st.number_input(f"{name}", min_value=0, max_value=10, value=0, step=1)
    opponents.extend([cls] * count)
# Run evaluation
if opponents and st.button("Run Simulation"):
    with st.spinner("Evaluating all strategies..."):
        st.markdown("---")
    
        best_strategy = None
        best_edge = -float("inf")
    
        for candidate in all_pickers:
            players = [candidate] + opponents 
            results = count_picker_wins(players, simulated_upcoming_season)
            
            ev = results["Expected Value Per Season ($1 Pool)"]
            edge_result = calculate_edge(ev)
    
            if edge_result['best_picker'] == candidate.__name__ + str(1) and edge_result['edge'] > best_edge:
                best_edge = edge_result['edge']
                best_strategy = candidate.__name__

    # Show best strategy
    st.success(f"🏆 Best strategy to use against selected opponents: **{best_strategy}** (Edge: {best_edge:.4f})")

else:
    st.warning("👈 Please select at least one opponent picker to begin.")
