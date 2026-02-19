"""
Data Loader Utility - Centralized data loading and preprocessing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class DataLoader:
    """Centralized data loading utility"""
    
    # Default data directory (relative to project root)
    DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
    
    @staticmethod
    def load_rank_features() -> pd.DataFrame:
        """Load rank features dataset"""
        file_path = DataLoader.DATA_DIR / "rank_features.csv"
        return pd.read_csv(file_path)
    
    @staticmethod
    def load_progression_features() -> pd.DataFrame:
        """Load progression features dataset"""
        file_path = DataLoader.DATA_DIR / "progression_features.csv"
        return pd.read_csv(file_path)
    
    @staticmethod
    def load_smurf_features() -> pd.DataFrame:
        """Load smurf features dataset"""
        file_path = DataLoader.DATA_DIR / "smurf_features.csv"
        return pd.read_csv(file_path)
    
    @staticmethod
    def load_match_features() -> pd.DataFrame:
        """Load match features dataset"""
        file_path = DataLoader.DATA_DIR / "match_features.csv"
        return pd.read_csv(file_path)
    
    @staticmethod
    def remap_tiers_to_4class(y_codes: np.ndarray) -> np.ndarray:
        """
        Remap 10-class tier codes to 4-class grouping:
        - Low: Iron (5) + Bronze (0) + Silver (8) → 0
        - Mid: Gold (4) + Platinum (7) + Emerald (3) → 1
        - High: Diamond (2) + Master (6) → 2
        - Elite: Grandmaster (9) + Challenger (1) → 3
        
        Tier code mapping (alphabetical):
        0: Bronze, 1: Challenger, 2: Diamond, 3: Emerald, 4: Gold, 5: Iron, 
        6: Master, 7: Platinum, 8: Silver, 9: Grandmaster
        """
        remap = {
            0: 0,  # Bronze → Low
            1: 3,  # Challenger → Elite
            2: 2,  # Diamond → High
            3: 1,  # Emerald → Mid
            4: 1,  # Gold → Mid
            5: 0,  # Iron → Low
            6: 2,  # Master → High
            7: 1,  # Platinum → Mid
            8: 0,  # Silver → Low
            9: 3,  # Grandmaster → Elite
        }
        return np.array([remap[code] for code in y_codes])
    
    @staticmethod
    def prepare_rank_features(remap_tiers: bool = False, add_interactions: bool = False) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Prepare rank features for training
        
        Args:
            remap_tiers: If True, remap 10-class to 4-class (Low/Mid/High/Elite)
            add_interactions: If True, add polynomial interaction features
        
        Returns:
            X, y, feature_names
        """
        df = DataLoader.load_rank_features()
        
        # Separate features and target FIRST
        y = pd.Categorical(df['tier']).codes  # Convert tier to numeric (0-9)
        
        # Remap to 4-class if requested
        if remap_tiers:
            y = DataLoader.remap_tiers_to_4class(y)
        
        X = df.drop(['tier', 'puuid', 'matches_used'], axis=1, errors='ignore')
        
        # Add interaction features BEFORE encoding roles
        if add_interactions:
            # Key interactions that should predict rank
            if 'win_rate' in X.columns and 'avg_kda' in X.columns:
                X['winrate_x_kda'] = X['win_rate'] * X['avg_kda']
            if 'win_rate' in X.columns and 'avg_gold_per_min' in X.columns:
                X['winrate_x_gold'] = X['win_rate'] * X['avg_gold_per_min']
            if 'recent_form_30' in X.columns and 'recent_form_10' in X.columns:
                X['form_momentum'] = X['recent_form_10'] - X['recent_form_30']
            if 'avg_kda' in X.columns and 'avg_gold_per_min' in X.columns:
                X['kda_x_gold'] = X['avg_kda'] * X['avg_gold_per_min']
            if 'kda_consistency' in X.columns and 'avg_kda' in X.columns:
                X['consistency_ratio'] = X['avg_kda'] / (X['kda_consistency'] + 1)
            if 'champion_pool' in X.columns and 'role_focus_pct' in X.columns:
                X['versatility'] = X['champion_pool'] * (1 - X['role_focus_pct'])
        
        # Encode main_role BEFORE filtering numeric (one-hot encoding)
        if 'main_role' in X.columns:
            X = pd.get_dummies(X, columns=['main_role'], prefix='role', drop_first=False, dtype=int)
        
        # Keep only numeric columns (now includes encoded role features as int)
        X = X.select_dtypes(include=['number'])
        
        feature_names = X.columns.tolist()
        return X.values, y, feature_names
    
    @staticmethod
    def prepare_progression_features() -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Prepare progression features for training (regression)
        
        Returns:
            X, y, feature_names
        """
        df = DataLoader.load_progression_features()
        
        # Separate features and target
        X = df.drop(['delta_winrate', 'puuid'], axis=1, errors='ignore')
        # Keep only numeric columns
        X = X.select_dtypes(include=['number'])
        
        # Handle missing values - fill with median
        X = X.fillna(X.median())
        
        y = df['delta_winrate'].values
        
        # Handle missing target values - fill with 0 (no change)
        y = np.nan_to_num(y, nan=0.0)
        
        feature_names = X.columns.tolist()
        return X.values, y, feature_names
    
    @staticmethod
    def prepare_smurf_features() -> Tuple[np.ndarray, list]:
        """
        Prepare smurf features for anomaly detection (unsupervised)
        
        Returns:
            X, feature_names
        """
        df = DataLoader.load_smurf_features()
        
        # Remove non-feature columns
        X = df.drop(['puuid', 'tier'], axis=1, errors='ignore')
        # Keep only numeric columns
        X = X.select_dtypes(include=['number'])
        
        # Handle missing values - fill with median
        X = X.fillna(X.median())
        
        feature_names = X.columns.tolist()
        return X.values, feature_names
    
    @staticmethod
    def prepare_match_features() -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Prepare match features for training (binary classification)
        
        Returns:
            X, y, feature_names
        """
        df = DataLoader.load_match_features()
        
        # Separate features and target
        X = df.drop(['team_won', 'match_id', 'team_id'], axis=1, errors='ignore')
        # Keep only numeric columns
        X = X.select_dtypes(include=['number'])
        
        # Handle missing values - fill with median
        X = X.fillna(X.median())
        
        y = df['team_won'].values
        
        # Handle missing target values
        y = np.nan_to_num(y, nan=0)
        
        feature_names = X.columns.tolist()
        return X.values, y, feature_names
    
    @staticmethod
    def get_data_info() -> dict:
        """Get basic info about all datasets"""
        info = {}
        
        try:
            rank_df = DataLoader.load_rank_features()
            info['rank_features'] = {
                'rows': len(rank_df),
                'cols': len(rank_df.columns),
                'path': str(DataLoader.DATA_DIR / "rank_features.csv")
            }
        except Exception as e:
            info['rank_features'] = {'error': str(e)}
        
        try:
            prog_df = DataLoader.load_progression_features()
            info['progression_features'] = {
                'rows': len(prog_df),
                'cols': len(prog_df.columns),
                'path': str(DataLoader.DATA_DIR / "progression_features.csv")
            }
        except Exception as e:
            info['progression_features'] = {'error': str(e)}
        
        try:
            smurf_df = DataLoader.load_smurf_features()
            info['smurf_features'] = {
                'rows': len(smurf_df),
                'cols': len(smurf_df.columns),
                'path': str(DataLoader.DATA_DIR / "smurf_features.csv")
            }
        except Exception as e:
            info['smurf_features'] = {'error': str(e)}
        
        try:
            match_df = DataLoader.load_match_features()
            info['match_features'] = {
                'rows': len(match_df),
                'cols': len(match_df.columns),
                'path': str(DataLoader.DATA_DIR / "match_features.csv")
            }
        except Exception as e:
            info['match_features'] = {'error': str(e)}
        
        return info
