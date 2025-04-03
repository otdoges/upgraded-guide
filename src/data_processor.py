import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeatherDataProcessor:
    """
    A class for processing weather data for machine learning models
    """
    
    def __init__(self, output_dir='/home/jackson/weather_prediction_bot/data'):
        """
        Initialize the data processor
        
        Args:
            output_dir (str): Directory to save processed data
        """
        self.output_dir = output_dir
        self.scaler = None
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_features(self, dataset):
        """
        Extract features from the raw dataset
        
        Args:
            dataset (dict): Dictionary containing raw weather data
            
        Returns:
            pandas.DataFrame: DataFrame with extracted features
        """
        features_list = []
        
        for date_str, date_data in dataset.items():
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            
            date_obj = datetime(year, month, day)
            day_of_year = date_obj.timetuple().tm_yday
            season = (month % 12 + 3) // 3
            
            for time_of_day, tod_data in date_data.items():
                # Extract hour from time of day
                hour = 0
                if time_of_day == 't0z':
                    hour = 0
                elif time_of_day == 't12z':
                    hour = 12
                elif time_of_day == 't18z':
                    hour = 18
                
                for file_name, file_data in tod_data.items():
                    # Basic feature extraction
                    for feature_name, feature_data in file_data.items():
                        # Extract statistics from the weather data
                        values = feature_data['values']
                        
                        # Skip if values is empty
                        if values.size == 0:
                            continue
                        
                        # Calculate statistical features
                        mean_val = np.mean(values)
                        max_val = np.max(values)
                        min_val = np.min(values)
                        std_val = np.std(values)
                        percentile_25 = np.percentile(values, 25)
                        percentile_75 = np.percentile(values, 75)
                        percentile_90 = np.percentile(values, 90)
                        percentile_95 = np.percentile(values, 95)
                        percentile_99 = np.percentile(values, 99)
                        
                        # Extract metadata
                        meta = feature_data['metadata']
                        name = meta['name']
                        level = meta['level']
                        forecast_time = meta['forecast_time']
                        
                        # Create feature dict
                        feature_dict = {
                            'date': date_str,
                            'year': year,
                            'month': month,
                            'day': day,
                            'day_of_year': day_of_year,
                            'season': season,
                            'hour': hour,
                            'feature_name': name,
                            'level': level,
                            'forecast_time': forecast_time,
                            'mean': mean_val,
                            'max': max_val,
                            'min': min_val,
                            'std': std_val,
                            'percentile_25': percentile_25,
                            'percentile_75': percentile_75,
                            'percentile_90': percentile_90,
                            'percentile_95': percentile_95,
                            'percentile_99': percentile_99,
                            'file_name': file_name
                        }
                        
                        features_list.append(feature_dict)
        
        # Convert list of dicts to DataFrame
        features_df = pd.DataFrame(features_list)
        logger.info(f"Extracted {len(features_df)} features")
        
        return features_df
    
    def pivot_features(self, features_df):
        """
        Pivot the features DataFrame to create a wide format suitable for ML
        
        Args:
            features_df (pandas.DataFrame): DataFrame with extracted features
            
        Returns:
            pandas.DataFrame: Pivoted DataFrame
        """
        # Group by date and create pivot
        # This creates columns like: hail_mean, hail_max, etc.
        stat_cols = ['mean', 'max', 'min', 'std', 'percentile_90', 'percentile_95', 'percentile_99']
        
        pivot_dfs = []
        
        for stat in stat_cols:
            pivot = features_df.pivot_table(
                index=['date', 'year', 'month', 'day', 'day_of_year', 'season'],
                columns=['feature_name', 'level', 'forecast_time'],
                values=stat,
                aggfunc='mean'
            )
            
            # Flatten multi-index columns
            pivot.columns = [f"{col[0]}_{col[1]}_{col[2]}_{stat}" for col in pivot.columns]
            pivot_dfs.append(pivot)
        
        # Merge all pivoted DataFrames
        result = pivot_dfs[0]
        for df in pivot_dfs[1:]:
            result = result.join(df)
        
        # Reset index to make date a column again
        result = result.reset_index()
        
        logger.info(f"Created pivoted features with {result.shape[1]} columns")
        return result
    
    def add_target_variables(self, pivoted_df, dataset, forecast_horizon=24):
        """
        Add target variables (what happened in reality) to the features
        
        Args:
            pivoted_df (pandas.DataFrame): DataFrame with pivoted features
            dataset (dict): Original dataset with actual weather data
            forecast_horizon (int): Forecast horizon in hours
            
        Returns:
            pandas.DataFrame: DataFrame with features and targets
        """
        # For demonstration, we'll use a simple approach:
        # The target will be the maximum hail probability in the next day
        dates = pivoted_df['date'].unique()
        
        targets = []
        
        for date_str in dates:
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            
            date_obj = datetime(year, month, day)
            next_date = date_obj + timedelta(hours=forecast_horizon)
            next_date_str = next_date.strftime('%Y%m%d')
            
            # Check if next date exists in dataset
            target_value = 0
            
            if next_date_str in dataset:
                next_date_data = dataset[next_date_str]
                
                # Find max values in next day's data
                for tod, tod_data in next_date_data.items():
                    for file_name, file_data in tod_data.items():
                        if 'hail' in file_name.lower():
                            for feature_name, feature_data in file_data.items():
                                values = feature_data['values']
                                if values.size > 0:
                                    max_val = np.max(values)
                                    target_value = max(target_value, max_val)
            
            targets.append({
                'date': date_str,
                'target_max_hail': target_value,
                'target_hail_binary': 1 if target_value > 0.5 else 0  # Binary classification threshold
            })
        
        targets_df = pd.DataFrame(targets)
        
        # Merge with pivoted features
        result = pd.merge(pivoted_df, targets_df, on='date', how='left')
        
        # Drop rows with missing targets
        result = result.dropna(subset=['target_max_hail'])
        
        logger.info(f"Added target variables, final dataset shape: {result.shape}")
        return result
    
    def scale_features(self, df, save_scaler=True):
        """
        Scale numerical features
        
        Args:
            df (pandas.DataFrame): DataFrame with features
            save_scaler (bool): Whether to save the scaler for later use
            
        Returns:
            pandas.DataFrame: DataFrame with scaled features
        """
        # Separate target and date columns
        target_cols = [col for col in df.columns if col.startswith('target_')]
        date_cols = ['date', 'year', 'month', 'day', 'day_of_year', 'season']
        
        # Get numerical feature columns
        feature_cols = [col for col in df.columns 
                       if col not in target_cols and col not in date_cols]
        
        # Create a copy to avoid modifying the original
        scaled_df = df.copy()
        
        # Initialize and fit scaler
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(df[feature_cols])
        
        # Replace original values with scaled values
        scaled_df[feature_cols] = scaled_data
        
        if save_scaler:
            scaler_path = os.path.join(self.output_dir, 'feature_scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved feature scaler to {scaler_path}")
        
        return scaled_df
    
    def split_data(self, df, test_size=0.2, val_size=0.1, time_split=True):
        """
        Split data into train, validation, and test sets
        
        Args:
            df (pandas.DataFrame): DataFrame with features and targets
            test_size (float): Proportion of data for testing
            val_size (float): Proportion of data for validation
            time_split (bool): Whether to split chronologically or randomly
            
        Returns:
            dict: Dictionary containing the split datasets
        """
        if time_split:
            # Sort by date
            df = df.sort_values('date')
            
            # Calculate split indices
            n = len(df)
            test_idx = int(n * (1 - test_size))
            val_idx = int(test_idx * (1 - val_size))
            
            # Split data
            train_df = df.iloc[:val_idx]
            val_df = df.iloc[val_idx:test_idx]
            test_df = df.iloc[test_idx:]
        else:
            # Random split
            train_df, temp_df = train_test_split(df, test_size=test_size+val_size, random_state=42)
            val_df, test_df = train_test_split(temp_df, test_size=test_size/(test_size+val_size), random_state=42)
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    
    def save_processed_data(self, data_dict):
        """
        Save processed data to files
        
        Args:
            data_dict (dict): Dictionary containing train, val, and test DataFrames
            
        Returns:
            dict: Dictionary with paths to saved files
        """
        paths = {}
        
        for split_name, df in data_dict.items():
            file_path = os.path.join(self.output_dir, f"{split_name}_data.csv")
            df.to_csv(file_path, index=False)
            paths[split_name] = file_path
            logger.info(f"Saved {split_name} data to {file_path}")
        
        return paths
    
    def process_data(self, dataset, forecast_horizon=24, test_size=0.2, val_size=0.1):
        """
        Process the dataset end-to-end
        
        Args:
            dataset (dict): Dictionary containing raw weather data
            forecast_horizon (int): Forecast horizon in hours
            test_size (float): Proportion of data for testing
            val_size (float): Proportion of data for validation
            
        Returns:
            dict: Dictionary containing processed data and paths
        """
        # Extract features
        features_df = self.extract_features(dataset)
        
        # Pivot features to wide format
        pivoted_df = self.pivot_features(features_df)
        
        # Add target variables
        with_targets_df = self.add_target_variables(pivoted_df, dataset, forecast_horizon)
        
        # Scale features
        scaled_df = self.scale_features(with_targets_df)
        
        # Split data
        data_splits = self.split_data(scaled_df, test_size, val_size)
        
        # Save processed data
        file_paths = self.save_processed_data(data_splits)
        
        return {
            'data': data_splits,
            'paths': file_paths
        } 