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
    All Pickers will inherit from this class. Defines basic functionality of a picker
        and chooses a winner randomly each week
    """

    def __init__(self, season_df):
        self.season_df = season_df
        self.weeks = sorted(np.unique(season_df['Week']))
        self.picked_teams = {}
        

    def make_pick_for_week(self, week_df):
        available_teams = self.get_available_teams_for_week(week_df)
        return random.choice(available_teams)
        
    def make_season_picks(self):
        for w in self.weeks:
            week_df = self.season_df[self.season_df['Week'] == w]
            picked_team = self.make_pick_for_week(week_df)
            self.picked_teams[w] = picked_team
        return self.picked_teams
        
    def get_available_teams_for_week(self, week_df):
        this_weeks_teams = week_df['Team']
        all_available_teams = [team for team in this_weeks_teams if team not in set(self.picked_teams.values())]
        return all_available_teams
        
    def evaluate_performance(self):
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

    # def evaluate_performance(self):
    #     num_weeks_survived = 0
    #     if not self.picked_teams:
    #         print('Picks havent been made yet!')
    #         return
    #     for week, picked_team in self.picked_teams.items():
    #         team_week_df = self.season_df[(self.season_df['Week'] == week) & (self.season_df['Team'] == picked_team)]
    #         if len(team_week_df[team_week_df['Won?'] == True]) > 0:
    #             num_weeks_survived += 1
    #             #print(f'Correctly picked the {picked_team} to win in Week {week}')
    #         else:
    #             #print(f'INCORRECTLY picked the {picked_team} to win in Week {week}')
    #             #print(f'You survived {num_weeks_survived} weeks!')
    #             break
    #     return num_weeks_survived


class BestOddsPicker(Picker):
    """
    Picks the team with the best odds each week out of the remaining available weeks
    """
        
    def make_pick_for_week(self, week_df):
        available_teams = self.get_available_teams_for_week(week_df)
        available_week_df = week_df[week_df['Team'].isin(available_teams)].sort_values(by='Odds', ascending=True)
        return available_week_df.iloc[0]['Team']
        
class TopKOddsPicker(Picker):
    """
    Randomly picks a team each week out of the remaining eligible teams with the top k best odds
    """
    def __init__(self, season_df, k=3):
        super().__init__(season_df)
        self.k = k
        
    def make_pick_for_week(self, week_df):
        available_teams = self.get_available_teams_for_week(week_df)
        available_week_df = week_df[week_df['Team'].isin(available_teams)].sort_values(by='Odds', ascending=True)
        return available_week_df.iloc[random.randint(0, self.k-1)]['Team']

class MaxOddsPicker(Picker):
    """
    This picker will choose its picks based on what will give it the maximum possible odds
    across all weeks
    """
    
    def make_pick_for_week(self, week_df):
        print('This method is not applicable for this child class')
        
    def make_season_picks(self):
        
        # hungarian algorithm! https://en.wikipedia.org/wiki/Hungarian_algorithm
        
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
    This picker will choose its picks based on what will give it the maximum possible odds
    across all weeks, but favoring earlier weeks by decaying odds the further in the future they are
    """
    def __init__(self, season_df, decay_factor=0.25):
        super().__init__(season_df)
        self.decay_factor = decay_factor
        
    def make_pick_for_week(self, week_df):
        print('This method is not applicable for this child class')
        
    def make_season_picks(self):
        
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
    Picks teams using a sliding window Hungarian algorithm.
    At each week, considers a window of future weeks (default 3)
    and selects the team assigned to the current week in that window.
    """
    def __init__(self, season_df, decay_factor=0.75, window_size=5):
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


# FUTURE: Reinforcement learning to find best strategy?

# In[ ]:




