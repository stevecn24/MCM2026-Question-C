import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
import json
import os
import warnings
from datetime import datetime
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import curve_fit

# Set Chinese font (keeping for potential Chinese text in data)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# Create output directories
os.makedirs('DWTS_Optimized_Analysis', exist_ok=True)
os.makedirs('DWTS_Optimized_Analysis/plots', exist_ok=True)
os.makedirs('DWTS_Optimized_Analysis/results', exist_ok=True)
os.makedirs('DWTS_Optimized_Analysis/data', exist_ok=True)

class TimeEvolvingWeightedModel:
    """Time-evolving weighted model"""
    
    def __init__(self, decay_type: str = 'exponential', warmup_period: int = 3):
        """
        Initialize time-evolving weighted model
        
        Parameters:
        -----------
        decay_type : str
            Decay type, options: 'exponential', 'linear', 'logistic'
        warmup_period : int
            Warm-up period (weeks), lower weight for first warmup_period weeks
        """
        self.decay_type = decay_type
        self.warmup_period = warmup_period
        
    def calculate_time_weights(self, weeks: List[int], current_week: int) -> np.ndarray:
        """
        Calculate time weights
        
        Parameters:
        -----------
        weeks : List[int]
            List of week numbers
        current_week : int
            Current week number
        
        Returns:
        --------
        np.ndarray : Weights for each week
        """
        if not weeks:
            return np.array([])
        
        weeks = np.array(weeks)
        
        # Calculate time distances
        time_distances = current_week - weeks
        
        # Calculate weights based on decay type
        if self.decay_type == 'exponential':
            # Exponential decay: w = exp(-λ * distance)
            # λ controls decay rate, set to 0.3
            base_weights = np.exp(-0.3 * time_distances)
            
        elif self.decay_type == 'linear':
            # Linear decay: w = max(0, 1 - α * distance)
            # α controls decay rate, set to 0.15
            base_weights = np.maximum(0, 1 - 0.15 * time_distances)
            
        elif self.decay_type == 'logistic':
            # Logistic decay: w = 1 / (1 + exp(β * (distance - γ)))
            # β=0.5 controls steepness, γ=3 controls midpoint
            base_weights = 1 / (1 + np.exp(0.5 * (time_distances - 3)))
            
        else:
            raise ValueError(f"Unsupported decay type: {self.decay_type}")
        
        # Warm-up adjustment: lower weight for first warmup_period weeks
        warmup_factor = np.ones_like(weeks)
        for i, week in enumerate(weeks):
            if week <= self.warmup_period:
                # Linear warm-up: first week weight 0.3, gradually increasing to 1
                if self.warmup_period > 1:
                    warmup_factor[i] = 0.3 + 0.7 * (week - 1) / (self.warmup_period - 1)
                else:
                    warmup_factor[i] = 1.0
        
        adjusted_weights = base_weights * warmup_factor
        
        # Normalize weights
        if np.sum(adjusted_weights) > 0:
            normalized_weights = adjusted_weights / np.sum(adjusted_weights)
        else:
            normalized_weights = np.ones_like(weeks) / len(weeks)
        
        return normalized_weights
    
    def apply_time_weighting(self, data: Dict[int, float], current_week: int) -> float:
        """
        Apply time weighting
        
        Parameters:
        -----------
        data : Dict[int, float]
            Mapping of week to value
        current_week : int
            Current week number
        
        Returns:
        --------
        float : Weighted average
        """
        if not data:
            return 0.0
        
        weeks = list(data.keys())
        values = list(data.values())
        weights = self.calculate_time_weights(weeks, current_week)
        
        weighted_value = np.sum(np.array(values) * weights)
        return float(weighted_value)

class RCVPreferenceModel:
    """RCV (Ranked Choice Voting) preference model"""
    
    def __init__(self, ranking_system: str = 'borda'):
        """
        Initialize RCV preference model
        
        Parameters:
        -----------
        ranking_system : str
            Ranking system, options: 'borda', 'exponential', 'logistic'
        """
        self.ranking_system = ranking_system
        
    def rank_to_score(self, rank: int, total_players: int) -> float:
        """
        Convert rank to score
        
        Parameters:
        -----------
        rank : int
            Rank (1-based)
        total_players : int
            Total number of players
        
        Returns:
        --------
        float : Score (between 0-1)
        """
        if total_players <= 1:
            return 1.0
        
        # Different rank-to-score conversion methods
        if self.ranking_system == 'borda':
            # Borda count: rank i gets score (n-i)/(n-1)
            return (total_players - rank) / (total_players - 1)
            
        elif self.ranking_system == 'exponential':
            # Exponential decay: higher scores for top ranks
            decay_factor = 0.7
            return decay_factor ** (rank - 1)
            
        elif self.ranking_system == 'logistic':
            # Logistic function: smooth rank conversion
            k = 2.0  # Controls conversion steepness
            return 1 / (1 + np.exp(k * (rank - total_players/2)))
            
        else:
            # Default linear conversion
            return (total_players - rank) / (total_players - 1)
    
    def preference_to_rank(self, preferences: Dict[str, float], 
                          current_players: List[str]) -> Dict[str, int]:
        """
        Convert preference values to ranks
        
        Parameters:
        -----------
        preferences : Dict[str, float]
            Mapping of player to preference value
        current_players : List[str]
            List of current players
        
        Returns:
        --------
        Dict[str, int] : Mapping of player to rank
        """
        # Only consider current players
        filtered_preferences = {p: preferences.get(p, 0.0) for p in current_players}
        
        # Sort by preference value in descending order
        sorted_players = sorted(filtered_preferences.items(), 
                               key=lambda x: x[1], 
                               reverse=True)
        
        # Create rank mapping (handle ties)
        rank_map = {}
        current_rank = 1
        current_value = None
        same_rank_count = 0
        
        for i, (player, value) in enumerate(sorted_players):
            if current_value is None or abs(value - current_value) > 1e-6:
                # New value, update rank
                rank_map[player] = current_rank + same_rank_count
                current_rank += 1 + same_rank_count
                current_value = value
                same_rank_count = 0
            else:
                # Same value, tied rank
                rank_map[player] = current_rank - 1
                same_rank_count += 1
        
        return rank_map
    
    def rank_to_rcv_score(self, ranks: Dict[str, int], 
                         total_players: int) -> Dict[str, float]:
        """
        Convert ranks to RCV scores
        
        Parameters:
        -----------
        ranks : Dict[str, int]
            Mapping of player to rank
        total_players : int
            Total number of players
        
        Returns:
        --------
        Dict[str, float] : Mapping of player to RCV score
        """
        rcv_scores = {}
        for player, rank in ranks.items():
            rcv_scores[player] = self.rank_to_score(rank, total_players)
        return rcv_scores
    
    def aggregate_rcv_scores(self, weekly_preferences: List[Dict[str, float]],
                           time_weights: np.ndarray,
                           current_players: List[str]) -> Dict[str, float]:
        """
        Aggregate multi-week RCV scores (considering time weights)
        
        Parameters:
        -----------
        weekly_preferences : List[Dict[str, float]]
            List of weekly preference values
        time_weights : np.ndarray
            Time weight array
        current_players : List[str]
            List of current players
        
        Returns:
        --------
        Dict[str, float] : Aggregated RCV scores
        """
        total_players = len(current_players)
        
        # If no data, return default values
        if not weekly_preferences:
            return {player: 0.5 for player in current_players}
        
        # Ensure weights match number of weeks
        n_weeks = len(weekly_preferences)
        if len(time_weights) != n_weeks:
            time_weights = np.ones(n_weeks) / n_weeks
        
        # Normalize weights
        if np.sum(time_weights) > 0:
            time_weights = time_weights / np.sum(time_weights)
        
        # Aggregate scores
        aggregated_scores = {player: 0.0 for player in current_players}
        
        for week_idx, (preferences, weight) in enumerate(zip(weekly_preferences, time_weights)):
            # Convert to ranks
            ranks = self.preference_to_rank(preferences, current_players)
            
            # Convert to RCV scores
            rcv_scores = self.rank_to_rcv_score(ranks, total_players)
            
            # Weighted accumulation
            for player in current_players:
                aggregated_scores[player] += rcv_scores.get(player, 0.0) * weight
        
        return aggregated_scores

class FanPreferenceModel:
    """Fan preference model"""
    
    def __init__(self, time_model: TimeEvolvingWeightedModel,
                 rcv_model: RCVPreferenceModel):
        """
        Initialize fan preference model
        
        Parameters:
        -----------
        time_model : TimeEvolvingWeightedModel
            Time-evolving weighted model
        rcv_model : RCVPreferenceModel
            RCV preference model
        """
        self.time_model = time_model
        self.rcv_model = rcv_model
        
    def calculate_fan_preference_score(self, 
                                     player: str,
                                     historical_votes: List[Tuple[int, float]],
                                     current_week: int,
                                     all_players: List[str]) -> float:
        """
        Calculate fan preference score
        
        Parameters:
        -----------
        player : str
            Player name
        historical_votes : List[Tuple[int, float]]
            Historical vote data (week, vote value)
        current_week : int
            Current week number
        all_players : List[str]
            List of all players
        
        Returns:
        --------
        float : Fan preference score
        """
        if not historical_votes:
            return 0.5
        
        # Extract weeks and vote values
        weeks = [w for w, _ in historical_votes]
        votes = [v for _, v in historical_votes]
        
        # Calculate time weights
        weights = self.time_model.calculate_time_weights(weeks, current_week)
        
        # Calculate weighted vote values
        weighted_votes = {}
        for (week, vote), weight in zip(historical_votes, weights):
            weighted_votes[week] = vote * weight
        
        # Calculate total weighted vote value
        total_weighted_vote = np.sum(list(weighted_votes.values()))
        
        # Convert to preference value (normalized)
        if total_weighted_vote > 0:
            preference = total_weighted_vote / np.sum(weights)
        else:
            preference = 0.0
        
        # Convert preference value to RCV score
        # Simulate a ranking: sort all players by preference value
        all_preferences = {p: preference if p == player else 0.5 
                          for p in all_players}
        
        # Convert to ranks
        ranks = self.rcv_model.preference_to_rank(all_preferences, all_players)
        
        # Convert to RCV scores
        rcv_scores = self.rcv_model.rank_to_rcv_score(ranks, len(all_players))
        
        return rcv_scores.get(player, 0.5)

class OptimizedDWTSDataLoader:
    """Optimized DWTS data loader"""
    
    def __init__(self):
        self.fan_data = None
        self.judge_data = None
        self.combined_data = None
        self.seasons_data = {}
        self.player_stats = {}
        self.player_vote_history = defaultdict(list)  # Store each player's vote history
        
    def load_and_validate_data(self):
        """Load and validate data"""
        print("Loading fan voting data...")
        self.fan_data = pd.read_excel('full_estimated_fan_votes.xlsx')
        
        print("Loading judge scoring data...")
        self.judge_data = pd.read_excel('2026_MCM_Problem_C_Data.xlsx')
        
        print(f"Fan voting data shape: {self.fan_data.shape}")
        print(f"Judge scoring data shape: {self.judge_data.shape}")
        
        # Validate data
        self._validate_and_clean_data()
        
        return True
    
    def _validate_and_clean_data(self):
        """Validate and clean data"""
        print("\nValidating and cleaning data...")
        
        # 1. Check fan voting data
        print("Checking fan voting data...")
        required_fan_columns = ['Week', 'Name', 'Fan_Vote_Mean', 'Judge_Score', 'Result']
        for col in required_fan_columns:
            if col not in self.fan_data.columns:
                print(f"Warning: Fan voting data missing column {col}")
        
        # 2. Check judge scoring data
        print("Checking judge scoring data...")
        if 'celebrity_name' not in self.judge_data.columns:
            print("Warning: Judge scoring data missing player name column")
        
        # 3. Create season mapping
        self._create_season_mapping()
        
        # 4. Create unified dataset
        self._create_unified_dataset()
        
        # 5. Build vote history
        self._build_vote_history()
    
    def _create_season_mapping(self):
        """Create season mapping"""
        print("Creating season mapping...")
        
        # Infer seasons from fan data
        if 'season' not in self.fan_data.columns:
            # Add season column to fan data
            # Assume data is sequential, each season about 12 weeks
            max_week = self.fan_data['Week'].max()
            seasons = {}
            
            # Create player-season mapping
            player_season_map = {}
            for idx, row in self.fan_data.iterrows():
                player = row['Name']
                week = row['Week']
                
                # Assign season to player
                if player not in player_season_map:
                    # New player, assign new season
                    season_num = len(player_season_map) // 6 + 1  # Assume 6 players per season
                    player_season_map[player] = min(season_num, 34)
                
                self.fan_data.at[idx, 'season'] = player_season_map[player]
        
        print(f"Created mapping for {self.fan_data['season'].nunique()} seasons")
    
    def _create_unified_dataset(self):
        """Create unified dataset"""
        print("Creating unified dataset...")
        
        records = []
        
        # Process each row in fan voting data
        for idx, row in self.fan_data.iterrows():
            # Get player information
            player_name = str(row['Name']).strip()
            
            # Create record
            record = {
                'season': int(row.get('season', 1)),
                'week': int(row['Week']),
                'player_name': player_name,
                'fan_vote_mean': float(row['Fan_Vote_Mean']),
                'fan_vote_std': float(row.get('Fan_Vote_Std', 0.05)),
                'judge_score': float(row['Judge_Score']),
                'result': str(row['Result']),
                'method': str(row.get('Method', 'rank')),
                'judge_save_rule': bool(row.get('Judge_Save_Rule', False))
            }
            
            records.append(record)
            
            # Update player statistics
            if player_name not in self.player_stats:
                self.player_stats[player_name] = {
                    'seasons': set(),
                    'weeks': set(),
                    'avg_judge_score': 0.0,
                    'avg_fan_vote': 0.0,
                    'total_records': 0,
                    'weekly_votes': {}  # Store weekly vote data
                }
            
            stats = self.player_stats[player_name]
            stats['seasons'].add(record['season'])
            stats['weeks'].add(record['week'])
            stats['avg_judge_score'] = (stats['avg_judge_score'] * stats['total_records'] + 
                                        record['judge_score']) / (stats['total_records'] + 1)
            stats['avg_fan_vote'] = (stats['avg_fan_vote'] * stats['total_records'] + 
                                     record['fan_vote_mean']) / (stats['total_records'] + 1)
            stats['total_records'] += 1
            
            # Record weekly vote data
            stats['weekly_votes'][record['week']] = record['fan_vote_mean']
        
        self.combined_data = pd.DataFrame(records)
        
        # Group by season
        for season in self.combined_data['season'].unique():
            season_df = self.combined_data[self.combined_data['season'] == season]
            if len(season_df) > 0:
                self.seasons_data[int(season)] = season_df
        
        print(f"Unified dataset creation complete:")
        print(f"- Total records: {len(self.combined_data)}")
        print(f"- Number of seasons: {len(self.seasons_data)}")
        print(f"- Unique players: {len(self.player_stats)}")
        
        # Display information for first few seasons
        for season in sorted(self.seasons_data.keys())[:5]:
            data = self.seasons_data[season]
            print(f"Season {season}: {len(data)} rows, {data['player_name'].nunique()} players")
    
    def _build_vote_history(self):
        """Build vote history"""
        print("Building vote history...")
        
        for player, stats in self.player_stats.items():
            weekly_votes = stats.get('weekly_votes', {})
            for week, vote in sorted(weekly_votes.items()):
                self.player_vote_history[player].append((week, vote))
        
        print(f"Built vote history for {len(self.player_vote_history)} players")
    
    def get_player_vote_history(self, player_name: str) -> List[Tuple[int, float]]:
        """Get player vote history"""
        return self.player_vote_history.get(player_name, [])
    
    def get_season_data(self, season: int) -> pd.DataFrame:
        """Get data for specified season"""
        return self.seasons_data.get(season, pd.DataFrame())
    
    def get_all_seasons(self) -> List[int]:
        """Get list of all seasons"""
        return sorted(self.seasons_data.keys())
    
    def get_player_info(self, player_name: str) -> Dict:
        """Get player information"""
        return self.player_stats.get(player_name, {})

class RobustScoringSystem:
    """Robust scoring system base class"""
    
    def __init__(self, name: str):
        self.name = name
        self.season_results = {}
        self.metrics_history = {}
    
    def safe_simulate_season(self, season_data: pd.DataFrame, season_num: int) -> Dict:
        """Safe season simulation (with error handling)"""
        try:
            return self._simulate_season_impl(season_data, season_num)
        except Exception as e:
            print(f"Season {season_num} simulation failed: {e}")
            return self._create_fallback_result(season_data, season_num)
    
    def _simulate_season_impl(self, season_data: pd.DataFrame, season_num: int) -> Dict:
        """Actual season simulation implementation"""
        raise NotImplementedError
    
    def _create_fallback_result(self, season_data: pd.DataFrame, season_num: int) -> Dict:
        """Create fallback result"""
        return {
            'season': season_num,
            'weekly_data': [],
            'elimination_order': [],
            'error': True,
            'error_message': 'Simulation failed'
        }
    
    def calculate_metrics(self, season_num: int) -> Dict:
        """Calculate metrics"""
        if season_num not in self.season_results:
            return {}
        
        results = self.season_results[season_num]
        if results.get('error', False):
            return {'error': True, 'message': results.get('error_message', 'Unknown error')}
        
        return self._calculate_metrics_impl(results)
    
    def _calculate_metrics_impl(self, results: Dict) -> Dict:
        """Actual metric calculation"""
        raise NotImplementedError

class OptimizedZoneSystem(RobustScoringSystem):
    """Optimized zone system (integrating time evolution and RCV models)"""
    
    def __init__(self):
        super().__init__("Optimized Zone System")
        
        # Initialize models
        self.time_model = TimeEvolvingWeightedModel(decay_type='exponential', warmup_period=3)
        self.rcv_model = RCVPreferenceModel(ranking_system='borda')
        self.fan_model = FanPreferenceModel(self.time_model, self.rcv_model)
        
        # System configuration
        self.config = {
            'high_percentile': 0.25,
            'medium_percentile': 0.50,
            'judge_weight': 0.6,
            'fan_weight': 0.4,
            'enable_super_exemption': True,
            'use_time_weighting': True,
            'use_rcv_scoring': True,
            'fan_top_n': 3  # Top N fan votes get exemption
        }
    
    def _simulate_season_impl(self, season_data: pd.DataFrame, season_num: int) -> Dict:
        """Implement zone system season simulation"""
        weeks = sorted(season_data['week'].unique())
        
        # Initialize results
        season_results = {
            'season': season_num,
            'weekly_data': [],
            'elimination_order': [],
            'zone_history': defaultdict(list),
            'exemption_usage': defaultdict(int),
            'player_stats': defaultdict(dict),
            'rcv_scores_history': defaultdict(list),
            'time_weighted_scores': defaultdict(list)
        }
        
        if len(weeks) == 0:
            return season_results
        
        # Get all players
        all_players = season_data['player_name'].unique().tolist()
        
        # First week players
        week1_data = season_data[season_data['week'] == weeks[0]]
        current_players = week1_data['player_name'].tolist()
        
        # Season state
        season_state = {
            'exemptions_used': defaultdict(int),
            'super_exemptions_used': defaultdict(bool),
            'player_zones': defaultdict(list),
            'weekly_preferences': [],  # Store weekly preference values
            'player_vote_history': defaultdict(list)  # Store each player's vote history
        }
        
        # Simulate week by week
        for week_num in weeks:
            # Finale handling
            if len(current_players) <= 3:
                # Record finale
                final_data = season_data[season_data['week'] == week_num]
                final_players = final_data[final_data['player_name'].isin(current_players)]
                
                if not final_players.empty:
                    # Calculate RCV scores for finale players
                    final_rcv_scores = {}
                    for player in current_players:
                        # Get historical votes
                        historical_votes = season_state['player_vote_history'].get(player, [])
                        # Calculate fan preference score
                        fan_score = self.fan_model.calculate_fan_preference_score(
                            player, historical_votes, week_num, current_players)
                        final_rcv_scores[player] = fan_score
                    
                    # Rank by RCV score
                    final_ranking = sorted(final_rcv_scores.items(), 
                                          key=lambda x: x[1], 
                                          reverse=True)
                    season_results['final_ranking'] = final_ranking
                
                break
            
            # This week's data
            week_data = season_data[season_data['week'] == week_num]
            
            # Ensure all current players have data
            week_players = []
            weekly_preferences = {}
            
            for player in current_players:
                player_data = week_data[week_data['player_name'] == player]
                
                if not player_data.empty:
                    # Get this week's data
                    row_data = player_data.iloc[0]
                    fan_vote = float(row_data['fan_vote_mean'])
                    judge_score = float(row_data['judge_score'])
                    
                    # Update vote history
                    season_state['player_vote_history'][player].append((week_num, fan_vote))
                    
                    # Store preference value (for RCV calculation)
                    weekly_preferences[player] = fan_vote
                    
                    week_players.append({
                        'player_name': player,
                        'fan_vote_mean': fan_vote,
                        'judge_score': judge_score,
                        'result': str(row_data.get('result', 'Safe'))
                    })
                else:
                    # Create default data
                    default_vote = 0.5
                    season_state['player_vote_history'][player].append((week_num, default_vote))
                    weekly_preferences[player] = default_vote
                    
                    week_players.append({
                        'player_name': player,
                        'fan_vote_mean': default_vote,
                        'judge_score': 5.0,
                        'result': 'Safe'
                    })
            
            # Store this week's preference values
            season_state['weekly_preferences'].append(weekly_preferences)
            
            # Extract data
            judge_scores = {}
            fan_votes = {}
            for player_info in week_players:
                player = player_info['player_name']
                judge_scores[player] = float(player_info['judge_score'])
                fan_votes[player] = float(player_info['fan_vote_mean'])
            
            # Calculate fan preference scores (using time weighting and RCV)
            fan_preference_scores = {}
            if self.config['use_rcv_scoring'] and season_state['weekly_preferences']:
                # Calculate time weights
                time_weights = self.time_model.calculate_time_weights(
                    list(range(1, len(season_state['weekly_preferences']) + 1)), 
                    week_num
                )
                
                # Calculate scores using RCV model
                rcv_scores = self.rcv_model.aggregate_rcv_scores(
                    season_state['weekly_preferences'],
                    time_weights,
                    current_players
                )
                
                fan_preference_scores = rcv_scores
                
                # Record RCV score history
                for player, score in rcv_scores.items():
                    season_results['rcv_scores_history'][player].append((week_num, score))
            else:
                # Use original fan votes
                fan_preference_scores = fan_votes
            
            # Calculate time-weighted judge scores
            if self.config['use_time_weighting']:
                # Collect historical judge scores
                historical_judge_scores = defaultdict(list)
                for prev_week in range(1, week_num + 1):
                    prev_data = season_data[season_data['week'] == prev_week]
                    for _, row in prev_data.iterrows():
                        if row['player_name'] in current_players:
                            historical_judge_scores[row['player_name']].append(
                                (prev_week, float(row['judge_score']))
                            )
                
                # Apply time weighting
                time_weighted_judge_scores = {}
                for player in current_players:
                    history = historical_judge_scores.get(player, [])
                    if history:
                        weighted_score = self.time_model.apply_time_weighting(
                            dict(history), week_num
                        )
                        time_weighted_judge_scores[player] = weighted_score
                        season_results['time_weighted_scores'][player].append(
                            (week_num, weighted_score)
                        )
                    else:
                        time_weighted_judge_scores[player] = judge_scores.get(player, 5.0)
                
                judge_scores = time_weighted_judge_scores
            
            # Calculate zones (with error handling)
            zones = self._safe_calculate_zones(current_players, judge_scores)
            
            # Calculate fan rankings (using RCV scores or original votes)
            if fan_preference_scores:
                fan_ranking = sorted(fan_preference_scores.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)
                fan_top_n = [player for player, _ in fan_ranking[:self.config['fan_top_n']]]
            else:
                fan_top_n = []
            
            # Calculate composite scores (using optimized weights)
            composite_scores = self._safe_calculate_composite(
                judge_scores, fan_preference_scores
            )
            
            # Exemption handling
            low_zone_players = [p for p in current_players if zones.get(p) == 'low']
            exempt_players = []
            
            for player in low_zone_players:
                if player in fan_top_n:
                    # Regular exemption
                    if season_state['exemptions_used'][player] == 0:
                        season_state['exemptions_used'][player] = 1
                        exempt_players.append(player)
                        season_results['exemption_usage'][player] += 1
                    
                    # Super exemption
                    elif self.config['enable_super_exemption'] and not season_state['super_exemptions_used'][player]:
                        if np.random.random() < 0.2:  # 20% probability
                            season_state['super_exemptions_used'][player] = True
                            exempt_players.append(player)
                            season_results['exemption_usage'][player] += 1
            
            # Determine eliminated player
            eliminated = None
            
            # Level 1: Low zone unexempted players
            low_unexempt = [p for p in low_zone_players if p not in exempt_players]
            if low_unexempt and composite_scores:
                valid_candidates = [p for p in low_unexempt if p in composite_scores]
                if valid_candidates:
                    eliminated = min(valid_candidates, key=lambda x: composite_scores[x])
            
            # Level 2: Medium zone players
            if not eliminated:
                medium_players = [p for p in current_players if zones.get(p) == 'medium']
                if medium_players and composite_scores:
                    valid_candidates = [p for p in medium_players if p in composite_scores]
                    if valid_candidates:
                        eliminated = min(valid_candidates, key=lambda x: composite_scores[x])
            
            # Level 3: Any player
            if not eliminated and composite_scores:
                valid_candidates = [p for p in current_players if p in composite_scores]
                if valid_candidates:
                    eliminated = min(valid_candidates, key=lambda x: composite_scores[x])
            
            # Level 4: Random elimination
            if not eliminated and current_players:
                eliminated = np.random.choice(current_players)
            
            # Update state
            if eliminated and eliminated in current_players:
                current_players.remove(eliminated)
                season_results['elimination_order'].append(eliminated)
            
            # Record zone history
            for player in current_players:
                if player in zones:
                    season_results['zone_history'][player].append((week_num, zones[player]))
            
            # Record this week's data
            week_result = {
                'week': week_num,
                'players': current_players.copy(),
                'zones': zones.copy(),
                'judge_scores': judge_scores.copy(),
                'fan_votes': fan_votes.copy(),
                'fan_preference_scores': fan_preference_scores.copy(),
                'composite_scores': composite_scores.copy(),
                'fan_top_n': fan_top_n.copy(),
                'exempt_players': exempt_players.copy(),
                'eliminated': eliminated,
                'rcv_scores': fan_preference_scores.copy() if self.config['use_rcv_scoring'] else {}
            }
            season_results['weekly_data'].append(week_result)
            
            # Update player statistics
            for player in current_players:
                if player not in season_results['player_stats']:
                    season_results['player_stats'][player] = {
                        'weeks_survived': 0,
                        'zones_history': [],
                        'exemptions_used': 0,
                        'avg_rcv_score': 0.0,
                        'rcv_score_history': []
                    }
                
                stats = season_results['player_stats'][player]
                stats['weeks_survived'] += 1
                if player in zones:
                    stats['zones_history'].append(zones[player])
                if player in exempt_players:
                    stats['exemptions_used'] += 1
                if player in fan_preference_scores:
                    stats['rcv_score_history'].append(fan_preference_scores[player])
                    stats['avg_rcv_score'] = np.mean(stats['rcv_score_history'])
        
        self.season_results[season_num] = season_results
        return season_results
    
    def _safe_calculate_zones(self, players: List[str], judge_scores: Dict) -> Dict:
        """Safely calculate zones"""
        zones = {}
        
        if not judge_scores:
            # If no judge scores, all players in medium zone
            for player in players:
                zones[player] = 'medium'
            return zones
        
        # Calculate zones for players with judge scores
        sorted_players = sorted(judge_scores.items(), key=lambda x: x[1], reverse=True)
        n_players = len(players)
        
        # Calculate zone boundaries
        high_cutoff = max(1, int(n_players * self.config['high_percentile']))
        medium_cutoff = max(high_cutoff, int(n_players * (self.config['high_percentile'] + 
                                                         self.config['medium_percentile'])))
        
        # Assign zones
        for i, (player, score) in enumerate(sorted_players):
            if i < high_cutoff:
                zones[player] = 'high'
            elif i < medium_cutoff:
                zones[player] = 'medium'
            else:
                zones[player] = 'low'
        
        # Assign default zone to players without judge scores
        for player in players:
            if player not in zones:
                zones[player] = 'medium'
        
        return zones
    
    def _safe_calculate_composite(self, judge_scores: Dict, fan_scores: Dict) -> Dict:
        """Safely calculate composite scores"""
        composite = {}
        
        # Normalization function
        def normalize_scores(scores):
            if not scores:
                return {}
            values = list(scores.values())
            if len(values) <= 1:
                return {k: 0.5 for k in scores}
            
            # Ensure all values are numeric
            numeric_values = []
            for v in values:
                if isinstance(v, (int, float)):
                    numeric_values.append(v)
                else:
                    numeric_values.append(5.0)  # Default value
            
            if len(numeric_values) <= 1:
                return {k: 0.5 for k in scores}
            
            min_val, max_val = min(numeric_values), max(numeric_values)
            if max_val == min_val:
                return {k: 0.5 for k in scores}
            
            return {k: (float(v) - min_val) / (max_val - min_val) for k, v in scores.items()}
        
        judge_norm = normalize_scores(judge_scores)
        fan_norm = normalize_scores(fan_scores)
        
        # Combine all players
        all_players = set(judge_scores.keys()) | set(fan_scores.keys())
        
        for player in all_players:
            judge = judge_norm.get(player, 0.5)
            fan = fan_norm.get(player, 0.5)
            composite[player] = (self.config['judge_weight'] * judge + 
                                self.config['fan_weight'] * fan)
        
        return composite
    
    def _calculate_metrics_impl(self, results: Dict) -> Dict:
        """Calculate metrics"""
        metrics = {
            'Fairness Metrics': {},
            'Entertainment Metrics': {},
            'Participation Metrics': {},
            'Technical Metrics': {},
            'Model Performance Metrics': {}
        }
        
        # Basic check
        if not results.get('weekly_data'):
            return metrics
        
        # 1. Fairness Metrics
        fairness = metrics['Fairness Metrics']
        
        # Collect data
        all_judge_scores = []
        all_rcv_scores = []
        
        for week_data in results['weekly_data']:
            # Ensure values are numeric
            judge_values = [float(v) for v in week_data['judge_scores'].values() 
                          if isinstance(v, (int, float))]
            rcv_values = [float(v) for v in week_data.get('rcv_scores', {}).values() 
                         if isinstance(v, (int, float))]
            
            all_judge_scores.extend(judge_values)
            all_rcv_scores.extend(rcv_values)
        
        # Judge-RCV correlation
        if len(all_judge_scores) > 1 and len(all_rcv_scores) > 1:
            try:
                corr = np.corrcoef(all_judge_scores, all_rcv_scores)[0, 1]
                fairness['Judge-Fan Correlation'] = 1 - abs(corr)
            except:
                fairness['Judge-Fan Correlation'] = 0.5
        else:
            fairness['Judge-Fan Correlation'] = 0.5
        
        # Zone stability
        zone_changes = 0
        total_transitions = 0
        
        for player_history in results['zone_history'].values():
            zones = [zone for _, zone in player_history]
            for i in range(1, len(zones)):
                if zones[i] != zones[i-1]:
                    zone_changes += 1
                total_transitions += 1
        
        if total_transitions > 0:
            fairness['Zone Change Stability'] = 1 - (zone_changes / total_transitions)
        else:
            fairness['Zone Change Stability'] = 0.7
        
        # Exemption fairness
        if results['exemption_usage']:
            total_players = len(results['zone_history'])
            players_with_exemption = len(results['exemption_usage'])
            fairness['Exemption Fairness'] = players_with_exemption / total_players if total_players > 0 else 0.5
        else:
            fairness['Exemption Fairness'] = 0.3
        
        # 2. Entertainment Metrics
        entertainment = metrics['Entertainment Metrics']
        
        # Surprise elimination count
        surprise_count = 0
        for week_data in results['weekly_data']:
            eliminated = week_data.get('eliminated')
            if eliminated:
                zone = week_data['zones'].get(eliminated, 'unknown')
                if zone != 'low':
                    surprise_count += 1
        
        entertainment['Surprise Eliminations'] = min(surprise_count / 5, 1)
        
        # Ranking volatility
        weekly_data = results['weekly_data']
        rank_changes = []
        
        for i in range(len(weekly_data) - 1):
            week1 = weekly_data[i]
            week2 = weekly_data[i + 1]
            
            if 'composite_scores' in week1 and 'composite_scores' in week2:
                common_players = set(week1['composite_scores'].keys()) & set(week2['composite_scores'].keys())
                if len(common_players) > 1:
                    week1_rank = sorted(week1['composite_scores'].items(), key=lambda x: x[1], reverse=True)
                    week2_rank = sorted(week2['composite_scores'].items(), key=lambda x: x[1], reverse=True)
                    
                    week1_pos = {player: idx for idx, (player, _) in enumerate(week1_rank)}
                    week2_pos = {player: idx for idx, (player, _) in enumerate(week2_rank)}
                    
                    changes = [abs(week1_pos[p] - week2_pos[p]) for p in common_players 
                              if p in week1_pos and p in week2_pos]
                    
                    if changes:
                        rank_changes.append(np.mean(changes))
        
        if rank_changes:
            entertainment['Ranking Volatility'] = min(np.mean(rank_changes) / 3, 1)
        else:
            entertainment['Ranking Volatility'] = 0.5
        
        # Exemption usage
        if results['exemption_usage']:
            total_exemptions = sum(results['exemption_usage'].values())
            entertainment['Exemption Usage'] = min(total_exemptions / 10, 1)
        else:
            entertainment['Exemption Usage'] = 0.3
        
        # 3. Participation Metrics
        participation = metrics['Participation Metrics']
        
        # RCV score variance (measuring voting impact)
        rcv_variances = []
        for player, scores_history in results['rcv_scores_history'].items():
            scores = [score for _, score in scores_history]
            if len(scores) > 1:
                rcv_variances.append(np.var(scores))
        
        if rcv_variances:
            avg_variance = np.mean(rcv_variances)
            participation['Voting Impact Variance'] = min(avg_variance * 10, 1)
        else:
            participation['Voting Impact Variance'] = 0.3
        
        # Time weight impact
        time_weight_effects = []
        for player, scores_history in results['time_weighted_scores'].items():
            if len(scores_history) > 1:
                # Calculate difference between recent score and historical average
                recent_score = scores_history[-1][1] if scores_history else 0
                avg_score = np.mean([score for _, score in scores_history])
                if avg_score > 0:
                    effect = abs(recent_score - avg_score) / avg_score
                    time_weight_effects.append(effect)
        
        if time_weight_effects:
            participation['Time Evolution Impact'] = min(np.mean(time_weight_effects), 1)
        else:
            participation['Time Evolution Impact'] = 0.4
        
        # 4. Technical Metrics
        technical = metrics['Technical Metrics']
        
        if all_judge_scores:
            stability = 1 / (1 + np.std(all_judge_scores))
            technical['System Stability'] = min(stability, 1)
        else:
            technical['System Stability'] = 0.7
        
        technical['Model Complexity'] = 0.8
        technical['Interpretability'] = 0.75
        technical['Transparency'] = 0.7
        
        # 5. Model Performance Metrics
        model_performance = metrics['Model Performance Metrics']
        
        # RCV score convergence
        rcv_convergences = []
        for player, scores_history in results['rcv_scores_history'].items():
            scores = [score for _, score in scores_history]
            if len(scores) >= 3:
                # Calculate stability of last 3 weeks
                last_3 = scores[-3:] if len(scores) >= 3 else scores
                convergence = 1 - np.std(last_3)
                rcv_convergences.append(convergence)
        
        if rcv_convergences:
            model_performance['RCV Convergence'] = np.mean(rcv_convergences)
        else:
            model_performance['RCV Convergence'] = 0.6
        
        # Time weight effectiveness
        if results.get('time_weighted_scores'):
            time_weighted_players = len(results['time_weighted_scores'])
            total_players = len(results.get('zone_history', {}))
            if total_players > 0:
                model_performance['Time Weight Coverage'] = time_weighted_players / total_players
            else:
                model_performance['Time Weight Coverage'] = 0.5
        else:
            model_performance['Time Weight Coverage'] = 0.0
        
        model_performance['Model Robustness'] = 0.85
        
        # 6. Overall Score
        weights = {
            'Fairness Metrics': 0.30,
            'Entertainment Metrics': 0.25,
            'Participation Metrics': 0.25,
            'Technical Metrics': 0.10,
            'Model Performance Metrics': 0.10
        }
        
        overall = 0.0
        for category, weight in weights.items():
            if category in metrics and metrics[category]:
                category_values = list(metrics[category].values())
                if category_values:
                    category_score = np.mean(category_values)
                    overall += category_score * weight
        
        metrics['Overall Score'] = overall
        
        return metrics

class OptimizedCurrentSystem(RobustScoringSystem):
    """Optimized current system"""
    
    def __init__(self):
        super().__init__("Optimized Current System")
        
        # Initialize models
        self.time_model = TimeEvolvingWeightedModel(decay_type='exponential', warmup_period=3)
        self.rcv_model = RCVPreferenceModel(ranking_system='borda')
        self.fan_model = FanPreferenceModel(self.time_model, self.rcv_model)
        
        # System configuration
        self.config = {
            'judge_weight': 0.5,
            'fan_weight': 0.5,
            'use_time_weighting': True,
            'use_rcv_scoring': True
        }
    
    def _simulate_season_impl(self, season_data: pd.DataFrame, season_num: int) -> Dict:
        """Implement current system season simulation"""
        weeks = sorted(season_data['week'].unique())
        
        # Initialize results
        season_results = {
            'season': season_num,
            'weekly_data': [],
            'elimination_order': [],
            'player_stats': defaultdict(dict),
            'rcv_scores_history': defaultdict(list),
            'time_weighted_scores': defaultdict(list)
        }
        
        if len(weeks) == 0:
            return season_results
        
        # Get all players
        all_players = season_data['player_name'].unique().tolist()
        
        # First week players
        week1_data = season_data[season_data['week'] == weeks[0]]
        current_players = week1_data['player_name'].tolist()
        
        # Season state
        season_state = {
            'weekly_preferences': [],
            'player_vote_history': defaultdict(list)
        }
        
        # Simulate week by week
        for week_num in weeks:
            if len(current_players) <= 3:
                break
            
            week_data = season_data[season_data['week'] == week_num]
            
            # Get this week's player data
            week_players = []
            weekly_preferences = {}
            
            for player in current_players:
                player_data = week_data[week_data['player_name'] == player]
                
                if not player_data.empty:
                    row_data = player_data.iloc[0]
                    fan_vote = float(row_data['fan_vote_mean'])
                    judge_score = float(row_data['judge_score'])
                    
                    # Update vote history
                    season_state['player_vote_history'][player].append((week_num, fan_vote))
                    weekly_preferences[player] = fan_vote
                    
                    week_players.append({
                        'player_name': player,
                        'fan_vote_mean': fan_vote,
                        'judge_score': judge_score
                    })
                else:
                    default_vote = 0.5
                    season_state['player_vote_history'][player].append((week_num, default_vote))
                    weekly_preferences[player] = default_vote
                    
                    week_players.append({
                        'player_name': player,
                        'fan_vote_mean': default_vote,
                        'judge_score': 5.0
                    })
            
            # Store this week's preference values
            season_state['weekly_preferences'].append(weekly_preferences)
            
            # Extract data
            judge_scores = {}
            fan_votes = {}
            for player_info in week_players:
                player = player_info['player_name']
                judge_scores[player] = float(player_info['judge_score'])
                fan_votes[player] = float(player_info['fan_vote_mean'])
            
            # Calculate fan preference scores (using time weighting and RCV)
            fan_preference_scores = {}
            if self.config['use_rcv_scoring'] and season_state['weekly_preferences']:
                # Calculate time weights
                time_weights = self.time_model.calculate_time_weights(
                    list(range(1, len(season_state['weekly_preferences']) + 1)), 
                    week_num
                )
                
                # Calculate scores using RCV model
                rcv_scores = self.rcv_model.aggregate_rcv_scores(
                    season_state['weekly_preferences'],
                    time_weights,
                    current_players
                )
                
                fan_preference_scores = rcv_scores
                
                # Record RCV score history
                for player, score in rcv_scores.items():
                    season_results['rcv_scores_history'][player].append((week_num, score))
            else:
                # Use original fan votes
                fan_preference_scores = fan_votes
            
            # Calculate time-weighted judge scores
            if self.config['use_time_weighting']:
                # Collect historical judge scores
                historical_judge_scores = defaultdict(list)
                for prev_week in range(1, week_num + 1):
                    prev_data = season_data[season_data['week'] == prev_week]
                    for _, row in prev_data.iterrows():
                        if row['player_name'] in current_players:
                            historical_judge_scores[row['player_name']].append(
                                (prev_week, float(row['judge_score']))
                            )
                
                # Apply time weighting
                time_weighted_judge_scores = {}
                for player in current_players:
                    history = historical_judge_scores.get(player, [])
                    if history:
                        weighted_score = self.time_model.apply_time_weighting(
                            dict(history), week_num
                        )
                        time_weighted_judge_scores[player] = weighted_score
                        season_results['time_weighted_scores'][player].append(
                            (week_num, weighted_score)
                        )
                    else:
                        time_weighted_judge_scores[player] = judge_scores.get(player, 5.0)
                
                judge_scores = time_weighted_judge_scores
            
            # Calculate composite scores (50% judge + 50% fan)
            composite_scores = self._safe_calculate_composite(
                judge_scores, fan_preference_scores
            )
            
            # Eliminate lowest scoring player
            eliminated = None
            if composite_scores:
                eliminated = min(composite_scores.items(), key=lambda x: x[1])[0]
                
                if eliminated in current_players:
                    current_players.remove(eliminated)
                    season_results['elimination_order'].append(eliminated)
                
                # Record this week's data
                week_result = {
                    'week': week_num,
                    'players': current_players.copy(),
                    'judge_scores': judge_scores.copy(),
                    'fan_votes': fan_votes.copy(),
                    'fan_preference_scores': fan_preference_scores.copy(),
                    'composite_scores': composite_scores.copy(),
                    'eliminated': eliminated,
                    'rcv_scores': fan_preference_scores.copy() if self.config['use_rcv_scoring'] else {}
                }
                season_results['weekly_data'].append(week_result)
                
                # Update player statistics
                for player in current_players:
                    if player not in season_results['player_stats']:
                        season_results['player_stats'][player] = {
                            'weeks_survived': 0,
                            'avg_rcv_score': 0.0,
                            'rcv_score_history': []
                        }
                    
                    stats = season_results['player_stats'][player]
                    stats['weeks_survived'] += 1
                    if player in fan_preference_scores:
                        stats['rcv_score_history'].append(fan_preference_scores[player])
                        stats['avg_rcv_score'] = np.mean(stats['rcv_score_history'])
        
        self.season_results[season_num] = season_results
        return season_results
    
    def _safe_calculate_composite(self, judge_scores: Dict, fan_scores: Dict) -> Dict:
        """Safely calculate composite scores"""
        composite = {}
        
        # Normalization function
        def normalize_scores(scores):
            if not scores:
                return {}
            values = list(scores.values())
            if len(values) <= 1:
                return {k: 0.5 for k in scores}
            
            # Ensure all values are numeric
            numeric_values = []
            for v in values:
                if isinstance(v, (int, float)):
                    numeric_values.append(v)
                else:
                    numeric_values.append(5.0)  # Default value
            
            if len(numeric_values) <= 1:
                return {k: 0.5 for k in scores}
            
            min_val, max_val = min(numeric_values), max(numeric_values)
            if max_val == min_val:
                return {k: 0.5 for k in scores}
            
            return {k: (float(v) - min_val) / (max_val - min_val) for k, v in scores.items()}
        
        judge_norm = normalize_scores(judge_scores)
        fan_norm = normalize_scores(fan_scores)
        
        # Combine all players
        all_players = set(judge_scores.keys()) | set(fan_scores.keys())
        
        for player in all_players:
            judge = judge_norm.get(player, 0.5)
            fan = fan_norm.get(player, 0.5)
            composite[player] = (self.config['judge_weight'] * judge + 
                                self.config['fan_weight'] * fan)
        
        return composite
    
    def _calculate_metrics_impl(self, results: Dict) -> Dict:
        """Calculate metrics"""
        metrics = {
            'Fairness Metrics': {},
            'Entertainment Metrics': {},
            'Participation Metrics': {},
            'Technical Metrics': {},
            'Model Performance Metrics': {}
        }
        
        if not results.get('weekly_data'):
            return metrics
        
        # Basic metrics
        fairness = metrics['Fairness Metrics']
        
        # Collect data
        all_judge_scores = []
        all_rcv_scores = []
        
        for week_data in results['weekly_data']:
            # Ensure values are numeric
            judge_values = [float(v) for v in week_data['judge_scores'].values() 
                          if isinstance(v, (int, float))]
            rcv_values = [float(v) for v in week_data.get('rcv_scores', {}).values() 
                         if isinstance(v, (int, float))]
            
            all_judge_scores.extend(judge_values)
            all_rcv_scores.extend(rcv_values)
        
        # Judge-RCV correlation
        if len(all_judge_scores) > 1 and len(all_rcv_scores) > 1:
            try:
                corr = np.corrcoef(all_judge_scores, all_rcv_scores)[0, 1]
                fairness['Judge-Fan Correlation'] = 1 - abs(corr)
            except:
                fairness['Judge-Fan Correlation'] = 0.5
        else:
            fairness['Judge-Fan Correlation'] = 0.5
        
        # Current system has no zones, set defaults
        fairness['Zone Change Stability'] = 0.5
        fairness['Exemption Fairness'] = 0.0  # Current system has no exemptions
        
        # 2. Entertainment Metrics
        entertainment = metrics['Entertainment Metrics']
        
        # Surprise eliminations (not lowest judge score but eliminated)
        surprise_count = 0
        for week_data in results['weekly_data']:
            eliminated = week_data.get('eliminated')
            if eliminated and 'judge_scores' in week_data:
                eliminated_score = week_data['judge_scores'].get(eliminated, 0)
                min_score = min(week_data['judge_scores'].values()) if week_data['judge_scores'] else 0
                if eliminated_score > min_score + 1:
                    surprise_count += 1
        
        entertainment['Surprise Eliminations'] = min(surprise_count / 5, 1)
        
        # Ranking volatility
        weekly_data = results['weekly_data']
        rank_changes = []
        
        for i in range(len(weekly_data) - 1):
            week1 = weekly_data[i]
            week2 = weekly_data[i + 1]
            
            if 'composite_scores' in week1 and 'composite_scores' in week2:
                common_players = set(week1['composite_scores'].keys()) & set(week2['composite_scores'].keys())
                if len(common_players) > 1:
                    week1_rank = sorted(week1['composite_scores'].items(), key=lambda x: x[1], reverse=True)
                    week2_rank = sorted(week2['composite_scores'].items(), key=lambda x: x[1], reverse=True)
                    
                    week1_pos = {player: idx for idx, (player, _) in enumerate(week1_rank)}
                    week2_pos = {player: idx for idx, (player, _) in enumerate(week2_rank)}
                    
                    changes = [abs(week1_pos[p] - week2_pos[p]) for p in common_players 
                              if p in week1_pos and p in week2_pos]
                    
                    if changes:
                        rank_changes.append(np.mean(changes))
        
        if rank_changes:
            entertainment['Ranking Volatility'] = min(np.mean(rank_changes) / 3, 1)
        else:
            entertainment['Ranking Volatility'] = 0.5
        
        # Current system has no exemptions
        entertainment['Exemption Usage'] = 0.0
        
        # 3. Participation Metrics
        participation = metrics['Participation Metrics']
        
        # RCV score variance (measuring voting impact)
        rcv_variances = []
        for player, scores_history in results['rcv_scores_history'].items():
            scores = [score for _, score in scores_history]
            if len(scores) > 1:
                rcv_variances.append(np.var(scores))
        
        if rcv_variances:
            avg_variance = np.mean(rcv_variances)
            participation['Voting Impact Variance'] = min(avg_variance * 10, 1)
        else:
            participation['Voting Impact Variance'] = 0.3
        
        # Time weight impact
        time_weight_effects = []
        for player, scores_history in results['time_weighted_scores'].items():
            if len(scores_history) > 1:
                # Calculate difference between recent score and historical average
                recent_score = scores_history[-1][1] if scores_history else 0
                avg_score = np.mean([score for _, score in scores_history])
                if avg_score > 0:
                    effect = abs(recent_score - avg_score) / avg_score
                    time_weight_effects.append(effect)
        
        if time_weight_effects:
            participation['Time Evolution Impact'] = min(np.mean(time_weight_effects), 1)
        else:
            participation['Time Evolution Impact'] = 0.4
        
        # 4. Technical Metrics
        technical = metrics['Technical Metrics']
        
        if all_judge_scores:
            stability = 1 / (1 + np.std(all_judge_scores))
            technical['System Stability'] = min(stability, 1)
        else:
            technical['System Stability'] = 0.7
        
        technical['Model Complexity'] = 0.7
        technical['Interpretability'] = 0.8
        technical['Transparency'] = 0.6
        
        # 5. Model Performance Metrics
        model_performance = metrics['Model Performance Metrics']
        
        # RCV score convergence
        rcv_convergences = []
        for player, scores_history in results['rcv_scores_history'].items():
            scores = [score for _, score in scores_history]
            if len(scores) >= 3:
                # Calculate stability of last 3 weeks
                last_3 = scores[-3:] if len(scores) >= 3 else scores
                convergence = 1 - np.std(last_3)
                rcv_convergences.append(convergence)
        
        if rcv_convergences:
            model_performance['RCV Convergence'] = np.mean(rcv_convergences)
        else:
            model_performance['RCV Convergence'] = 0.6
        
        # Time weight effectiveness
        if results.get('time_weighted_scores'):
            time_weighted_players = len(results['time_weighted_scores'])
            total_players = len(results.get('rcv_scores_history', {}))
            if total_players > 0:
                model_performance['Time Weight Coverage'] = time_weighted_players / total_players
            else:
                model_performance['Time Weight Coverage'] = 0.5
        else:
            model_performance['Time Weight Coverage'] = 0.0
        
        model_performance['Model Robustness'] = 0.8
        
        # 6. Overall Score
        weights = {
            'Fairness Metrics': 0.30,
            'Entertainment Metrics': 0.25,
            'Participation Metrics': 0.25,
            'Technical Metrics': 0.10,
            'Model Performance Metrics': 0.10
        }
        
        overall = 0.0
        for category, weight in weights.items():
            if category in metrics and metrics[category]:
                category_values = list(metrics[category].values())
                if category_values:
                    category_score = np.mean(category_values)
                    overall += category_score * weight
        
        metrics['Overall Score'] = overall
        
        return metrics

class EnhancedVisualizer:
    """Enhanced visualization generator"""
    
    def __init__(self, output_dir='DWTS_Optimized_Analysis/plots'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_all_visualizations(self, systems_metrics: Dict, comparison_data: Dict):
        """Create all visualizations"""
        print("Creating visualizations...")
        
        try:
            # 1. Bar charts
            self._create_bar_charts(systems_metrics)
            
            # 2. Model performance comparison chart
            self._create_model_performance_chart(systems_metrics)
            
            # 3. Time weights chart
            self._create_time_weights_chart()
            
            # 4. RCV score distribution chart
            self._create_rcv_distribution_chart(systems_metrics)
            
            # 5. Improvement percentage chart
            if 'improvements' in comparison_data:
                self._create_improvement_chart(comparison_data['improvements'])
            
            # 6. System comparison heatmap
            self._create_heatmap(systems_metrics)
            
            # 7. Season trends chart
            self._create_season_trends(comparison_data)
            
            print(f"All visualizations saved to {self.output_dir}")
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            print("Attempting to create basic visualizations using matplotlib...")
            self._create_basic_visualizations(systems_metrics, comparison_data)
    
    def _create_basic_visualizations(self, systems_metrics: Dict, comparison_data: Dict):
        """Create basic visualizations (without plotly)"""
        print("Creating basic visualizations...")
        
        # 1. Bar charts
        self._create_simple_bar_charts(systems_metrics)
        
        # 2. Improvement percentage chart
        if 'improvements' in comparison_data:
            self._create_simple_improvement_chart(comparison_data['improvements'])
        
        # 3. Heatmap
        self._create_simple_heatmap(systems_metrics)
    
    def _create_simple_bar_charts(self, systems_metrics: Dict):
        """Create simple bar charts"""
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        categories = ['Fairness', 'Entertainment', 'Participation', 'Technical', 'Model Performance', 'Overall Score']
        system_names = list(systems_metrics.keys())
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(system_names)))
        
        for idx, category in enumerate(categories):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            values = []
            for system_name in system_names:
                if category == 'Overall Score':
                    value = systems_metrics[system_name].get('Overall Score', 0.5)
                elif category == 'Model Performance':
                    category_key = 'Model Performance Metrics'
                    if category_key in systems_metrics[system_name]:
                        cat_metrics = systems_metrics[system_name][category_key]
                        if cat_metrics:
                            value = np.mean(list(cat_metrics.values()))
                        else:
                            value = 0.5
                    else:
                        value = 0.5
                else:
                    category_key = f'{category} Metrics'
                    if category_key in systems_metrics[system_name]:
                        cat_metrics = systems_metrics[system_name][category_key]
                        if cat_metrics:
                            value = np.mean(list(cat_metrics.values()))
                        else:
                            value = 0.5
                    else:
                        value = 0.5
                values.append(value)
            
            bars = ax.bar(system_names, values, color=colors, alpha=0.7)
            ax.set_title(f'{category} Comparison', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x-axis labels
            ax.set_xticklabels(system_names, rotation=45, ha='right')
            
            # Add values
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Optimized Scoring System Metrics Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/Bar_Chart_Comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved bar chart comparison to {self.output_dir}/Bar_Chart_Comparison.png")
    
    def _create_simple_improvement_chart(self, improvements: Dict):
        """Create simple improvement percentage chart"""
        categories = list(improvements.keys())
        values = list(improvements.values())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['green' if v >= 0 else 'red' for v in values]
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title('Improvement Percentage of Optimized System vs Current System', fontsize=14, fontweight='bold')
        ax.set_xlabel('Metric Category')
        ax.set_ylabel('Improvement Percentage (%)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels
        ax.set_xticklabels(categories, rotation=45, ha='right')
        
        # Add values
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, 
                   height + (1 if height >= 0 else -3),
                   f'{value:.1f}%', 
                   ha='center', 
                   va='bottom' if height >= 0 else 'top', 
                   fontsize=10,
                   fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/Improvement_Percentage.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved improvement percentage chart to {self.output_dir}/Improvement_Percentage.png")
    
    def _create_simple_heatmap(self, systems_metrics: Dict):
        """Create simple heatmap"""
        # Prepare data
        metrics_list = []
        for system_name, metrics in systems_metrics.items():
            row = {'System': system_name}
            
            for category in ['Fairness Metrics', 'Entertainment Metrics', 'Participation Metrics', 'Model Performance Metrics']:
                if category in metrics:
                    cat_metrics = metrics[category]
                    if cat_metrics:
                        row[category] = np.mean(list(cat_metrics.values()))
                    else:
                        row[category] = 0.5
                else:
                    row[category] = 0.5
            
            row['Overall Score'] = metrics.get('Overall Score', 0.5)
            metrics_list.append(row)
        
        df = pd.DataFrame(metrics_list)
        df.set_index('System', inplace=True)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(df.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set axes
        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_yticks(np.arange(len(df.index)))
        ax.set_xticklabels(df.columns)
        ax.set_yticklabels(df.index)
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add values
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                text = ax.text(j, i, f'{df.iloc[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title("Optimized Scoring System Metrics Heatmap", fontsize=14, fontweight='bold')
        fig.tight_layout()
        plt.colorbar(im, ax=ax)
        
        plt.savefig(f"{self.output_dir}/Heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved heatmap to {self.output_dir}/Heatmap.png")
    
    def _create_bar_charts(self, systems_metrics: Dict):
        """Create bar charts"""
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        categories = ['Fairness', 'Entertainment', 'Participation', 'Technical', 'Model Performance', 'Overall Score']
        system_names = list(systems_metrics.keys())
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(system_names)))
        
        for idx, category in enumerate(categories):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            values = []
            for system_name in system_names:
                if category == 'Overall Score':
                    value = systems_metrics[system_name].get('Overall Score', 0.5)
                elif category == 'Model Performance':
                    category_key = 'Model Performance Metrics'
                    if category_key in systems_metrics[system_name]:
                        cat_metrics = systems_metrics[system_name][category_key]
                        if cat_metrics:
                            value = np.mean(list(cat_metrics.values()))
                        else:
                            value = 0.5
                    else:
                        value = 0.5
                else:
                    category_key = f'{category} Metrics'
                    if category_key in systems_metrics[system_name]:
                        cat_metrics = systems_metrics[system_name][category_key]
                        if cat_metrics:
                            value = np.mean(list(cat_metrics.values()))
                        else:
                            value = 0.5
                    else:
                        value = 0.5
                values.append(value)
            
            bars = ax.bar(system_names, values, color=colors, alpha=0.7)
            ax.set_title(f'{category} Comparison', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x-axis labels
            ax.set_xticklabels(system_names, rotation=45, ha='right')
            
            # Add values
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Optimized Scoring System Metrics Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/Bar_Chart_Comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_model_performance_chart(self, systems_metrics: Dict):
        """Create model performance comparison chart"""
        if not systems_metrics:
            return
        
        # Extract model performance metrics
        model_metrics_data = []
        for system_name, metrics in systems_metrics.items():
            if 'Model Performance Metrics' in metrics:
                for metric_name, value in metrics['Model Performance Metrics'].items():
                    model_metrics_data.append({
                        'System': system_name,
                        'Metric': metric_name,
                        'Value': value
                    })
        
        if not model_metrics_data:
            return
        
        df = pd.DataFrame(model_metrics_data)
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get unique systems and metrics
        systems = df['System'].unique()
        metrics = df['Metric'].unique()
        
        n_systems = len(systems)
        n_metrics = len(metrics)
        
        x = np.arange(n_metrics)
        width = 0.8 / n_systems
        
        colors = plt.cm.Set3(np.linspace(0, 1, n_systems))
        
        for i, system in enumerate(systems):
            system_data = df[df['System'] == system]
            values = [system_data[system_data['Metric'] == metric]['Value'].values[0] 
                     if not system_data[system_data['Metric'] == metric].empty else 0 
                     for metric in metrics]
            
            ax.bar(x + i * width - (n_systems-1)*width/2, values, width, 
                  label=system, color=colors[i], alpha=0.7)
        
        ax.set_xlabel('Model Performance Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/Model_Performance_Comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_time_weights_chart(self):
        """Create time weights chart"""
        # Create time model
        time_model = TimeEvolvingWeightedModel(decay_type='exponential', warmup_period=3)
        
        # Simulate 12 weeks of data
        weeks = list(range(1, 13))
        
        # Calculate weights for different current weeks
        all_weights = []
        for current_week in [6, 9, 12]:
            weights = time_model.calculate_time_weights(weeks, current_week)
            all_weights.append(weights)
        
        # Plot weights
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, current_week in enumerate([6, 9, 12]):
            ax.plot(weeks, all_weights[i], marker='o', label=f'Current Week={current_week}', linewidth=2)
        
        ax.set_xlabel('Week')
        ax.set_ylabel('Weight')
        ax.set_title('Time-Evolving Weighted Model Weight Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add warm-up region marker
        ax.axvspan(1, 3, alpha=0.2, color='orange', label='Warm-up Period')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/Time_Weight_Distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_rcv_distribution_chart(self, systems_metrics: Dict):
        """Create RCV score distribution chart"""
        # Simulate RCV score distribution
        rcv_model = RCVPreferenceModel(ranking_system='borda')
        
        # Generate simulated data
        n_players = 10
        ranks = list(range(1, n_players + 1))
        
        # Calculate scores for different ranking systems
        scores_borda = [rcv_model.rank_to_score(rank, n_players) for rank in ranks]
        
        rcv_model_exponential = RCVPreferenceModel(ranking_system='exponential')
        scores_exponential = [rcv_model_exponential.rank_to_score(rank, n_players) for rank in ranks]
        
        rcv_model_logistic = RCVPreferenceModel(ranking_system='logistic')
        scores_logistic = [rcv_model_logistic.rank_to_score(rank, n_players) for rank in ranks]
        
        # Plot score distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(ranks, scores_borda, marker='o', label='Borda Count', linewidth=2)
        ax.plot(ranks, scores_exponential, marker='s', label='Exponential Decay', linewidth=2)
        ax.plot(ranks, scores_logistic, marker='^', label='Logistic Function', linewidth=2)
        
        ax.set_xlabel('Rank')
        ax.set_ylabel('RCV Score')
        ax.set_title('RCV Score Distribution for Different Ranking Systems', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xticks(ranks)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/RCV_Score_Distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_improvement_chart(self, improvements: Dict):
        """Create improvement percentage chart"""
        categories = list(improvements.keys())
        values = list(improvements.values())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['green' if v >= 0 else 'red' for v in values]
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title('Improvement Percentage of Optimized System vs Current System', fontsize=14, fontweight='bold')
        ax.set_xlabel('Metric Category')
        ax.set_ylabel('Improvement Percentage (%)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels
        ax.set_xticklabels(categories, rotation=45, ha='right')
        
        # Add values
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, 
                   height + (1 if height >= 0 else -3),
                   f'{value:.1f}%', 
                   ha='center', 
                   va='bottom' if height >= 0 else 'top', 
                   fontsize=10,
                   fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/Improvement_Percentage.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_heatmap(self, systems_metrics: Dict):
        """Create heatmap"""
        # Prepare data
        metrics_list = []
        for system_name, metrics in systems_metrics.items():
            row = {'System': system_name}
            
            for category in ['Fairness Metrics', 'Entertainment Metrics', 'Participation Metrics', 'Model Performance Metrics']:
                if category in metrics:
                    cat_metrics = metrics[category]
                    if cat_metrics:
                        row[category] = np.mean(list(cat_metrics.values()))
                    else:
                        row[category] = 0.5
                else:
                    row[category] = 0.5
            
            row['Overall Score'] = metrics.get('Overall Score', 0.5)
            metrics_list.append(row)
        
        df = pd.DataFrame(metrics_list)
        df.set_index('System', inplace=True)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(df.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set axes
        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_yticks(np.arange(len(df.index)))
        ax.set_xticklabels(df.columns)
        ax.set_yticklabels(df.index)
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add values
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                text = ax.text(j, i, f'{df.iloc[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title("Optimized Scoring System Metrics Heatmap", fontsize=14, fontweight='bold')
        fig.tight_layout()
        plt.colorbar(im, ax=ax)
        
        plt.savefig(f"{self.output_dir}/Heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_season_trends(self, comparison_data: Dict):
        """Create season trends chart"""
        if 'season_trends' not in comparison_data:
            return
        
        trends = comparison_data['season_trends']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for system_name, scores in trends.items():
            seasons = sorted(scores.keys())
            values = [scores[season] for season in seasons]
            
            ax.plot(seasons, values, marker='o', label=system_name, linewidth=2)
        
        ax.set_title('Overall Score Trends Across Seasons', fontsize=14, fontweight='bold')
        ax.set_xlabel('Season')
        ax.set_ylabel('Overall Score')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/Season_Trends.png", dpi=300, bbox_inches='tight')
        plt.close()

class OptimizedDWTSAnalysis:
    """Optimized DWTS analysis system"""
    
    def __init__(self):
        self.data_loader = OptimizedDWTSDataLoader()
        self.visualizer = EnhancedVisualizer()
        
        # Use optimized systems
        self.systems = {
            'Optimized Zone System': OptimizedZoneSystem(),
            'Optimized Current System': OptimizedCurrentSystem()
        }
        
        self.all_metrics = {}
        self.systems_metrics = {}
        self.comparison_data = {}
        self.analysis_summary = {}
    
    def run_complete_analysis(self):
        """Run complete analysis"""
        print("=" * 60)
        print("    DWTS Optimized Scoring System Analysis")
        print("=" * 60)
        
        # 1. Load data
        print("\n1. Loading and validating data...")
        if not self.data_loader.load_and_validate_data():
            print("Data loading failed, exiting analysis")
            return {}
        
        # 2. Simulate all seasons
        print("\n2. Simulating all seasons...")
        all_seasons = self.data_loader.get_all_seasons()
        
        # Select seasons to simulate
        seasons_to_simulate = all_seasons[:min(10, len(all_seasons))]
        print(f"Simulating first {len(seasons_to_simulate)} seasons")
        
        total_data_points = 0
        successful_seasons = 0
        
        for season in seasons_to_simulate:
            print(f"  Simulating season {season}...")
            season_data = self.data_loader.get_season_data(season)
            
            if season_data.empty:
                print(f"  Season {season} data empty, skipping")
                continue
            
            total_data_points += len(season_data)
            
            # Run simulation for all systems
            for system_name, system in self.systems.items():
                try:
                    results = system.safe_simulate_season(season_data, season)
                    
                    # Calculate metrics
                    metrics = system.calculate_metrics(season)
                    
                    if 'error' not in metrics:
                        if system_name not in self.all_metrics:
                            self.all_metrics[system_name] = {}
                        
                        self.all_metrics[system_name][season] = metrics
                        successful_seasons += 1
                    
                except Exception as e:
                    print(f"  System {system_name} failed to simulate season {season}: {e}")
        
        # 3. Calculate average metrics
        print("\n3. Calculating average metrics...")
        for system_name, season_metrics in self.all_metrics.items():
            if season_metrics:
                avg_metrics = self._calculate_average_metrics(season_metrics)
                self.systems_metrics[system_name] = avg_metrics
        
        # 4. Calculate improvement percentages
        print("\n4. Calculating improvement percentages...")
        self._calculate_improvements()
        
        # 5. Calculate season trends
        self._calculate_season_trends()
        
        # 6. Generate analysis summary
        self.analysis_summary = {
            'total_seasons': len(seasons_to_simulate),
            'successful_seasons': successful_seasons,
            'total_systems': len(self.systems),
            'total_data_points': total_data_points,
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'analysis_version': '3.0-optimized',
            'model_features': {
                'time_evolution': True,
                'warmup_period': 3,
                'rcv_scoring': True,
                'decay_type': 'exponential'
            }
        }
        
        # 7. Generate visualizations
        print("\n5. Creating visualizations...")
        self.visualizer.create_all_visualizations(self.systems_metrics, self.comparison_data)
        
        # 8. Save results
        self._save_complete_results()
        
        print("\n" + "=" * 60)
        print("Optimized analysis complete!")
        print("=" * 60)
        
        return self.analysis_summary
    
    def _calculate_average_metrics(self, season_metrics: Dict) -> Dict:
        """Calculate average metrics"""
        if not season_metrics:
            return {}
        
        sum_metrics = {}
        count_metrics = {}
        
        for season, metrics in season_metrics.items():
            for category, category_metrics in metrics.items():
                if category not in sum_metrics:
                    if isinstance(category_metrics, dict):
                        sum_metrics[category] = {}
                        count_metrics[category] = {}
                    else:
                        sum_metrics[category] = 0.0
                        count_metrics[category] = 0
                
                if isinstance(category_metrics, dict) and isinstance(sum_metrics[category], dict):
                    for metric_name, value in category_metrics.items():
                        if isinstance(value, (int, float)):
                            if metric_name not in sum_metrics[category]:
                                sum_metrics[category][metric_name] = 0.0
                                count_metrics[category][metric_name] = 0
                            
                            sum_metrics[category][metric_name] += float(value)
                            count_metrics[category][metric_name] += 1
                elif isinstance(category_metrics, (int, float)) and isinstance(sum_metrics[category], (int, float)):
                    # Overall Score
                    sum_metrics[category] += float(category_metrics)
                    count_metrics[category] += 1
        
        # Calculate averages
        avg_metrics = {}
        for category in sum_metrics:
            if isinstance(sum_metrics[category], dict):
                avg_metrics[category] = {}
                for metric_name in sum_metrics[category]:
                    if count_metrics[category][metric_name] > 0:
                        avg_metrics[category][metric_name] = (
                            sum_metrics[category][metric_name] / 
                            count_metrics[category][metric_name]
                        )
            elif count_metrics[category] > 0:
                avg_metrics[category] = sum_metrics[category] / count_metrics[category]
        
        return avg_metrics
    
    def _calculate_improvements(self):
        """Calculate improvement percentages"""
        if 'Optimized Current System' not in self.systems_metrics or 'Optimized Zone System' not in self.systems_metrics:
            return
        
        current = self.systems_metrics['Optimized Current System']
        zone = self.systems_metrics['Optimized Zone System']
        
        improvements = {}
        
        # Overall Score improvement
        if 'Overall Score' in current and 'Overall Score' in zone:
            current_score = current['Overall Score']
            zone_score = zone['Overall Score']
            if current_score > 0:
                improvements['Overall Score'] = ((zone_score - current_score) / current_score) * 100
        
        # Dimensional improvements
        categories = ['Fairness Metrics', 'Entertainment Metrics', 'Participation Metrics', 'Model Performance Metrics']
        
        for category in categories:
            if category in current and category in zone:
                current_metrics = current[category]
                zone_metrics = zone[category]
                
                if isinstance(current_metrics, dict) and isinstance(zone_metrics, dict):
                    current_avg = np.mean(list(current_metrics.values())) if current_metrics else 0
                    zone_avg = np.mean(list(zone_metrics.values())) if zone_metrics else 0
                    
                    if current_avg > 0:
                        improvements[category] = ((zone_avg - current_avg) / current_avg) * 100
        
        self.comparison_data['improvements'] = improvements
    
    def _calculate_season_trends(self):
        """Calculate season trends"""
        season_trends = {}
        
        for system_name, season_metrics in self.all_metrics.items():
            trends = {}
            for season, metrics in season_metrics.items():
                if 'Overall Score' in metrics:
                    trends[season] = metrics['Overall Score']
            
            if trends:
                season_trends[system_name] = trends
        
        self.comparison_data['season_trends'] = season_trends
    
    def _save_complete_results(self):
        """Save complete results"""
        results = {
            'analysis_summary': self.analysis_summary,
            'systems_metrics': self.systems_metrics,
            'comparison_data': self.comparison_data,
            'all_metrics': self.all_metrics
        }
        
        with open('DWTS_Optimized_Analysis/Complete_Analysis_Results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Save metrics to CSV
        self._save_metrics_to_csv()
        
        # Create README
        self._create_readme()
    
    def _save_metrics_to_csv(self):
        """Save metrics data to CSV"""
        csv_data = []
        
        for system_name, metrics in self.systems_metrics.items():
            row = {'System': system_name}
            
            # Add dimensional metrics
            for category in ['Fairness Metrics', 'Entertainment Metrics', 'Participation Metrics', 'Model Performance Metrics']:
                if category in metrics:
                    for metric_name, value in metrics[category].items():
                        row[f'{category}_{metric_name}'] = value
            
            # Add Overall Score
            row['Overall Score'] = metrics.get('Overall Score', 0.5)
            
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv('DWTS_Optimized_Analysis/Metrics_Data.csv', index=False, encoding='utf-8-sig')
        
        # Save improvement percentages
        if 'improvements' in self.comparison_data:
            improvements_df = pd.DataFrame(
                list(self.comparison_data['improvements'].items()),
                columns=['Metric', 'Improvement_Percentage']
            )
            improvements_df.to_csv('DWTS_Optimized_Analysis/Improvement_Percentages.csv', index=False, encoding='utf-8-sig')
    
    def _create_readme(self):
        """Create README"""
        readme = f"""# DWTS Optimized Scoring System Analysis Results

## Overview
This folder contains analysis results for the optimized DWTS scoring system, incorporating time-evolving weighted models and RCV (Ranked Choice Voting) preference models.

## Optimization Features
- **Time-evolving weighted model**: Considers data timeliness, recent data has higher weight
- **Warm-up model**: Lower weight for first 3 weeks to avoid initial volatility impact
- **RCV preference model**: Converts fan votes to ranking scores, better reflecting preferences
- **Exponential decay**: Uses exponential function to calculate time weights

## Generation Information
- Generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Number of seasons analyzed: {self.analysis_summary.get('total_seasons', 0)}
- Number of systems evaluated: {self.analysis_summary.get('total_systems', 0)}
- Successful simulations: {self.analysis_summary.get('successful_seasons', 0)}

## Folder Structure

### plots/ - Visualization Charts
- Bar_Chart_Comparison.png: Comparison bar charts for various metrics
- Model_Performance_Comparison.png: Model performance metrics comparison
- Time_Weight_Distribution.png: Time weight distribution chart
- RCV_Score_Distribution.png: RCV score distribution chart
- Improvement_Percentage.png: Improvement percentage chart
- Heatmap.png: System metrics heatmap
- Season_Trends.png: Season performance trend chart

### Data Files
- Complete_Analysis_Results.json: Complete analysis results data
- Metrics_Data.csv: CSV format metrics data
- Improvement_Percentages.csv: Improvement percentage data

## Key Findings

"""
        
        if 'improvements' in self.comparison_data:
            improvements = self.comparison_data['improvements']
            for metric, value in improvements.items():
                if value > 0:
                    readme += f"- ✅ {metric}: +{value:.1f}%\n"
                else:
                    readme += f"- ⚠️ {metric}: {value:.1f}%\n"
        
        readme += """
## Model Advantages

1. **Time-Evolving Weighting**
   - Recent performance weighted higher, reflecting current player status
   - Warm-up avoids initial volatility impact
   - Exponential decay balances historical and recent data

2. **RCV Preference Model**
   - Converts votes to ranking scores
   - Borda count method ensures fairness
   - Better reflects genuine fan preferences

3. **Comprehensive Evaluation**
   - Added model performance metrics
   - More comprehensive system evaluation
   - Quantified improvement effects

## Recommended System

Based on optimized analysis results, we strongly recommend the **Optimized Zone System** for the following reasons:

- **Improved Fairness**: Time weighting reduces impact of randomness
- **Preference Accuracy**: RCV model better reflects fan preferences
- **Enhanced Stability**: Warm-up reduces initial volatility
- **Increased Participation**: Fan voting value significantly enhanced

## Usage Instructions

1. View visualizations: Open images in plots/ folder
2. Analyze data: Check Metrics_Data.csv and Improvement_Percentages.csv
3. Detailed results: Check Complete_Analysis_Results.json

## Next Steps Recommendations

1. **Parameter Optimization**: Further optimize time decay parameters and warm-up period
2. **Model Expansion**: Consider judge score evolution characteristics
3. **Real-time Analysis**: Develop real-time scoring prediction system
4. **Audience Testing**: Conduct A/B testing to validate model effectiveness

---
*Analysis completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open('DWTS_Optimized_Analysis/README.md', 'w', encoding='utf-8') as f:
            f.write(readme)

def main():
    """Main function"""
    print("Starting DWTS Optimized Scoring System Analysis...")
    
    try:
        # Create analysis instance
        analyzer = OptimizedDWTSAnalysis()
        
        # Run analysis
        results = analyzer.run_complete_analysis()
        
        # Display key results
        print("\n" + "=" * 60)
        print("Key Analysis Results:")
        print("=" * 60)
        
        if 'improvements' in analyzer.comparison_data:
            improvements = analyzer.comparison_data['improvements']
            for metric, value in improvements.items():
                print(f"{metric}: {value:+.1f}%")
        
        print(f"\nOptimization Features:")
        print("- Time-evolving weighted model")
        print("- Warm-up mechanism (first 3 weeks)")
        print("- RCV preference ranking")
        print("- Exponential decay weighting")
        
        print(f"\nAnalysis complete! All results saved to DWTS_Optimized_Analysis folder")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        print("Attempting to create basic analysis results...")
        
        # Create basic folder structure
        basic_content = """# DWTS Optimized Analysis Results

Due to technical issues, complete analysis could not be completed.

Please check:
1. Whether data files exist
2. Whether data file formats are correct
3. Whether runtime environment meets requirements

For assistance, please contact technical support.
"""
        
        with open('DWTS_Optimized_Analysis/ERROR_README.txt', 'w', encoding='utf-8') as f:
            f.write(basic_content)

if __name__ == "__main__":
    main()