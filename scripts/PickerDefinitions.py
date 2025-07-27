#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import pandas as pd
import random
import itertools
from scipy.optimize import linear_sum_assignment

# Define various picker classes that utilize different strategies to pick each week

class Picker:
    """
    Base class for all picker strategies in a survivor pool.

    This class implements the core logic for managing weekly picks,
    tracking used teams, and evaluating performance. The default strategy
    randomly picks an available team each week.
    """

    def __init__(self, season_df):
        """
        Initializes the Picker with the season's data.

        Parameters:
            season_df (pd.DataFrame): season data
        """
        self.season_df = season_df
        self.weeks = sorted(np.unique(season_df['Week']))
        self.picked_teams = {}
        

    def make_pick_for_week(self, week_df):
        """
        Picks a team for the given week from available (not yet picked) teams.

        Parameters:
            week_df (pd.DataFrame): Data for a specific week.

        Returns:
            str: Chosen team's name.
        """
        available_teams = self.get_available_teams_for_week(week_df)
        return random.choice(available_teams)
        
    def make_season_picks(self):
        """
        Makes picks for each week of the season using the picker strategy.

        Returns:
            dict: Mapping from week number to picked team.
        """
        for w in self.weeks:
            week_df = self.season_df[self.season_df['Week'] == w]
            picked_team = self.make_pick_for_week(week_df)
            self.picked_teams[w] = picked_team
        return self.picked_teams
        
    def get_available_teams_for_week(self, week_df):
        """
        Returns teams eligible to be picked (i.e., not picked in prior weeks).

        Parameters:
            week_df (pd.DataFrame): Data for a specific week.

        Returns:
            list: List of eligible team names.
        """
        this_weeks_teams = week_df['Team']
        all_available_teams = [team for team in this_weeks_teams if team not in set(self.picked_teams.values())]
        return all_available_teams
        
    def evaluate_performance(self):
        """
        Evaluates how many weeks the picker survived without making a wrong pick.

        Returns:
            int: Number of weeks survived.
        """
        if not self.picked_teams:
            print('Picks havent been made yet!')
            return 0
    
        season_lookup = (
            self.season_df.set_index(['Week', 'Team'])['Won?']
            .to_dict()
        )
    
        for i, (week, picked_team) in enumerate(self.picked_teams.items(), 1):
            if season_lookup.get((week, picked_team), False):
                continue
            return i - 1  # eliminated this week
        return len(self.picked_teams)  # survived all weeks


class BestOddsPicker(Picker):
    """
    Picker that selects the team with the best (lowest) odds to win each week.
    """
        
    def make_pick_for_week(self, week_df):
        """
        Picks the team with the lowest odds among the eligible teams.

        Parameters:
            week_df (pd.DataFrame): Data for a specific week.

        Returns:
            str: Chosen team's name.
        """
        available_teams = self.get_available_teams_for_week(week_df)
        available_week_df = week_df[week_df['Team'].isin(available_teams)].sort_values(by='Odds', ascending=True)
        return available_week_df.iloc[0]['Team']
        
class TopKOddsPicker(Picker):
    """
    Picker that randomly chooses from the top-k best-odds teams each week.

    Parameters:
        k (int): Number of top teams to randomly select from.
    """
    def __init__(self, season_df, k=3):
        """
        Initializes the picker with a specified 'k' value.

        Parameters:
            season_df (pd.DataFrame): Season data.
            k (int): Number of top teams to consider each week.
        """
        super().__init__(season_df)
        self.k = k
        
    def make_pick_for_week(self, week_df):
        """
        Randomly picks a team out of the team with the k lowest odds among the eligible teams.

        Parameters:
            week_df (pd.DataFrame): Data for a specific week.

        Returns:
            str: Chosen team's name.
        """
        available_teams = self.get_available_teams_for_week(week_df)
        available_week_df = week_df[week_df['Team'].isin(available_teams)].sort_values(by='Odds', ascending=True)
        return available_week_df.iloc[random.randint(0, self.k-1)]['Team']

class MaxOddsPicker(Picker):
    """
    Picker that optimizes the entire season using the Hungarian algorithm to
    maximize total implied win probability across all weeks.
    """
    
    def make_pick_for_week(self, week_df):
        print('This method is not applicable for this child class')
        
    def make_season_picks(self):
        """
        Uses the Hungarian algorithm to assign one team to each week to maximize
        total implied probability.

        Returns:
            dict: Mapping from week number to picked team.
        """
        
        def hungarian_algorithm(season_df):
            # Pivot Teams x Weeks matrix of probabilities
            prob_matrix = season_df.pivot(index="Team", columns="Week", values="Implied Prob")
            weeks = sorted(season_df['Week'].unique())
            teams = prob_matrix.index.tolist()
        
            # Replace NaNs with very small prob to penalize unused teams
            small_value = 1e-9
            prob_matrix_filled = prob_matrix.fillna(small_value)
        
            # Cost matrix: -log(probability)
            cost_matrix = -np.log(prob_matrix_filled.values)
        
            # Solve assignment problem
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
            # Build result
            week_to_team = {
                weeks[c]: teams[r] for r, c in zip(row_ind, col_ind)
                }
            week_to_team = dict(sorted(week_to_team.items()))
            return week_to_team
        self.picked_teams = hungarian_algorithm(self.season_df)
        return self.picked_teams
        
class MaxOddsWithDecayPicker(Picker):
    """
    Like MaxOddsPicker, but applies a decay factor to prioritize earlier wins.

    Parameters:
        decay_factor (float): Multiplier <1 to reduce weight of later weeks.
    """
    def __init__(self, season_df, decay_factor=0.25):
        """
        Initializes the picker with a decay factor.

        Parameters:
            season_df (pd.DataFrame): Season data.
            decay_factor (float): Week weighting decay factor.
        """
        super().__init__(season_df)
        self.decay_factor = decay_factor
        
    def make_pick_for_week(self, week_df):
        print('This method is not applicable for this child class')
        
    def make_season_picks(self):
        """
        Uses the Hungarian algorithm to assign one team to each week to maximize
        total implied probability (with decay factor applied).

        Returns:
            dict: Mapping from week number to picked team.
        """
        # hungarian algorithm! https://en.wikipedia.org/wiki/Hungarian_algorithm
        
        def hungarian_algorithm(season_df, decay_factor):
            # Pivot Teams x Weeks matrix of probabilities
            prob_matrix = season_df.pivot(index="Team", columns="Week", values="Implied Prob")
            for i, week in enumerate(prob_matrix.columns, start=1):
                decay_multiplier = decay_factor ** (i - 1)
                prob_matrix[week] *= decay_multiplier
            weeks = sorted(season_df['Week'].unique())
            teams = prob_matrix.index.tolist()
        
            # Replace NaNs with very small prob to penalize unused teams
            small_value = 1e-9
            prob_matrix_filled = prob_matrix.fillna(small_value)
        
            # Cost matrix: -log(probability)
            cost_matrix = -np.log(prob_matrix_filled.values)
        
            # Solve assignment problem
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
            # Build result
            week_to_team = {
                weeks[c]: teams[r] for r, c in zip(row_ind, col_ind)
                }
            week_to_team = dict(sorted(week_to_team.items()))
            return week_to_team
        self.picked_teams = hungarian_algorithm(self.season_df, self.decay_factor)
        return self.picked_teams
        


class SlidingWindowPicker(Picker):
    """
    Applies a sliding window strategy using the Hungarian algorithm.
    At each week, it considers a window of future weeks and assigns teams 
    to maximize win probability with decay over the window.

    Parameters:
        decay_factor (float): Weighting factor for later weeks in each window.
        window_size (int): Number of weeks to include in the sliding window.
    """
    def __init__(self, season_df, decay_factor=0.75, window_size=5):
        """
        Initializes the picker with sliding window parameters.

        Parameters:
            season_df (pd.DataFrame): Season data.
            decay_factor (float): Multiplier for decaying future week importance.
            window_size (int): Number of weeks to consider ahead.
        """
        super().__init__(season_df)
        self.window_size = window_size
        self.decay_factor = decay_factor

    def make_pick_for_week(self, week_df):
        print('This method is not applicable for this child class')

    def make_season_picks(self):
        weeks = sorted(self.season_df['Week'].unique())
        used_teams = set()
        week_to_team = {}

        for current_week in weeks:
            # Build sliding window of weeks
            window_weeks = [w for w in weeks if current_week <= w < current_week + self.window_size]
            window_df = self.season_df[self.season_df['Week'].isin(window_weeks)]

            # Exclude already used teams
            window_df = window_df[~window_df['Team'].isin(used_teams)]

            # Pivot to probability matrix
            prob_matrix = window_df.pivot(index="Team", columns="Week", values="Implied Prob").copy()

            # Replace NaNs with small value to penalize unavailable slots
            prob_matrix_filled = prob_matrix.fillna(1e-9)

            # Cost matrix: -log(probability)
            cost_matrix = -np.log(prob_matrix_filled.values) / (self.decay_factor ** np.arange(len(prob_matrix.columns)))

            # Solve assignment problem
            teams = prob_matrix.index.tolist()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Map weeks in window to teams
            window_week_to_team = {prob_matrix.columns[c]: teams[r] for r, c in zip(row_ind, col_ind)}

            # Choose team assigned to current week
            chosen_team = window_week_to_team[current_week]
            week_to_team[current_week] = chosen_team
            used_teams.add(chosen_team)

        self.picked_teams = week_to_team
        return week_to_team

# In[ ]:




