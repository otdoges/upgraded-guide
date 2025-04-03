#!/usr/bin/env python3
"""
Advanced Meteorological Features
===============================
This module provides advanced feature engineering functions
for improved accuracy in weather prediction.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import pygrib
import requests
from scipy import stats
from scipy.ndimage import gaussian_filter
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedFeatureExtractor:
    """Extract advanced meteorological features from weather data."""
    
    def __init__(self, data_dir='data', extra_data_dir='data/extra'):
        """
        Initialize the feature extractor.
        
        Args:
            data_dir (str): Directory containing weather data
            extra_data_dir (str): Directory containing supplementary data
        """
        self.data_dir = os.path.abspath(data_dir)
        self.extra_data_dir = os.path.abspath(extra_data_dir)
        
        # Create extra data directory if it doesn't exist
        os.makedirs(self.extra_data_dir, exist_ok=True)
        
        # Load supplementary data if available
        self.population_data = self._load_population_data()
        self.elevation_data = self._load_elevation_data()
        self.climate_indices = self._load_climate_indices()
        
        # Initialize cache for computed features
        self.feature_cache = {}
        
        logger.info("Advanced feature extractor initialized")
    
    def _load_population_data(self):
        """Load population density data."""
        pop_file = os.path.join(self.extra_data_dir, "us_population.dat")
        
        if os.path.exists(pop_file):
            try:
                logger.info(f"Loading population data from {pop_file}")
                return pd.read_csv(pop_file)
            except Exception as e:
                logger.error(f"Error loading population data: {e}")
        else:
            logger.warning("Population data file not found")
        
        return None
    
    def _load_elevation_data(self):
        """Load terrain elevation data."""
        elev_file = os.path.join(self.extra_data_dir, "elevation.dat")
        
        if os.path.exists(elev_file):
            try:
                logger.info(f"Loading elevation data from {elev_file}")
                return np.load(elev_file)
            except Exception as e:
                logger.error(f"Error loading elevation data: {e}")
        else:
            logger.warning("Elevation data file not found")
        
        return None
    
    def _load_climate_indices(self):
        """Load climate indices (ENSO, PDO, NAO, etc.)."""
        indices_file = os.path.join(self.extra_data_dir, "noaa_indices.dat")
        
        if os.path.exists(indices_file):
            try:
                logger.info(f"Loading climate indices from {indices_file}")
                return pd.read_csv(indices_file)
            except Exception as e:
                logger.error(f"Error loading climate indices: {e}")
        else:
            logger.warning("Climate indices file not found")
        
        return None
    
    def extract_advanced_features(self, grib_data, file_name):
        """
        Extract advanced features from GRIB data.
        
        Args:
            grib_data (dict): Data from a GRIB file
            file_name (str): Name of the file
            
        Returns:
            dict: Advanced features extracted from the data
        """
        features = {}
        
        # Check if cached
        if file_name in self.feature_cache:
            return self.feature_cache[file_name]
        
        try:
            # Extract basic meteorological features
            basic_features = self._extract_basic_features(grib_data)
            features.update(basic_features)
            
            # Extract derived meteorological features
            derived_features = self._extract_derived_features(grib_data)
            features.update(derived_features)
            
            # Extract pattern-based features
            pattern_features = self._extract_pattern_features(grib_data)
            features.update(pattern_features)
            
            # Extract temporal trend features if data is available
            trend_features = self._extract_trend_features(file_name, grib_data)
            features.update(trend_features)
            
            # Add climate index correlations if available
            if self.climate_indices is not None:
                climate_features = self._add_climate_index_features(file_name)
                features.update(climate_features)
            
            # Add terrain interaction features if elevation data available
            if self.elevation_data is not None:
                terrain_features = self._add_terrain_interaction_features(grib_data)
                features.update(terrain_features)
            
            # Cache the features
            self.feature_cache[file_name] = features
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting advanced features from {file_name}: {e}")
            return {}
    
    def _extract_basic_features(self, grib_data):
        """
        Extract basic meteorological features.
        
        Args:
            grib_data (dict): Data from a GRIB file
            
        Returns:
            dict: Basic meteorological features
        """
        features = {}
        
        for key, data in grib_data.items():
            values = data['values']
            
            # Skip non-meteorological variables
            if 'metadata' not in data or 'name' not in data['metadata']:
                continue
                
            var_name = data['metadata']['name']
            
            # Statistical features
            features[f"{var_name}_mean"] = np.mean(values)
            features[f"{var_name}_std"] = np.std(values)
            features[f"{var_name}_min"] = np.min(values)
            features[f"{var_name}_max"] = np.max(values)
            features[f"{var_name}_median"] = np.median(values)
            features[f"{var_name}_p90"] = np.percentile(values, 90)
            features[f"{var_name}_p10"] = np.percentile(values, 10)
            features[f"{var_name}_iqr"] = np.percentile(values, 75) - np.percentile(values, 25)
            
            # Distribution features
            features[f"{var_name}_skew"] = stats.skew(values.flatten())
            features[f"{var_name}_kurtosis"] = stats.kurtosis(values.flatten())
            
            # Spatial coverage features
            if np.max(values) > 0:
                coverage = np.sum(values > np.max(values) * 0.5) / values.size
                features[f"{var_name}_coverage"] = coverage
            else:
                features[f"{var_name}_coverage"] = 0
        
        return features
    
    def _extract_derived_features(self, grib_data):
        """
        Extract derived meteorological features.
        
        Args:
            grib_data (dict): Data from a GRIB file
            
        Returns:
            dict: Derived meteorological features
        """
        features = {}
        
        # Find temperature and humidity fields if they exist
        temp_data = None
        humidity_data = None
        wind_u_data = None
        wind_v_data = None
        
        for key, data in grib_data.items():
            if 'metadata' not in data or 'name' not in data['metadata']:
                continue
                
            name = data['metadata']['name'].lower()
            
            if 'temperature' in name:
                temp_data = data
            elif 'humidity' in name or 'specific humidity' in name:
                humidity_data = data
            elif 'u-component' in name:
                wind_u_data = data
            elif 'v-component' in name:
                wind_v_data = data
        
        # Derive heat index if both temperature and humidity are available
        if temp_data is not None and humidity_data is not None:
            try:
                temp_values = temp_data['values']
                humidity_values = humidity_data['values']
                
                # Simple heat index calculation (simplified)
                heat_index = 0.5 * (temp_values + 61.0 + ((temp_values - 68.0) * 1.2) + (humidity_values * 0.094))
                
                features["heat_index_mean"] = np.mean(heat_index)
                features["heat_index_max"] = np.max(heat_index)
                features["heat_index_coverage"] = np.sum(heat_index > 80) / heat_index.size
            except Exception as e:
                logger.error(f"Error calculating heat index: {e}")
        
        # Derive wind speed and direction if both components are available
        if wind_u_data is not None and wind_v_data is not None:
            try:
                u_values = wind_u_data['values']
                v_values = wind_v_data['values']
                
                # Calculate wind speed
                wind_speed = np.sqrt(u_values**2 + v_values**2)
                
                # Calculate wind direction (in degrees)
                wind_dir = np.arctan2(v_values, u_values) * (180/np.pi)
                wind_dir = (wind_dir + 360) % 360  # Convert to 0-360 range
                
                features["wind_speed_mean"] = np.mean(wind_speed)
                features["wind_speed_max"] = np.max(wind_speed)
                features["wind_dir_mean"] = np.mean(wind_dir)
                features["wind_dir_std"] = np.std(wind_dir)
                
                # Add wind quadrant features
                q1 = np.sum((wind_dir >= 0) & (wind_dir < 90)) / wind_dir.size
                q2 = np.sum((wind_dir >= 90) & (wind_dir < 180)) / wind_dir.size
                q3 = np.sum((wind_dir >= 180) & (wind_dir < 270)) / wind_dir.size
                q4 = np.sum((wind_dir >= 270) & (wind_dir < 360)) / wind_dir.size
                
                features["wind_q1_ratio"] = q1
                features["wind_q2_ratio"] = q2
                features["wind_q3_ratio"] = q3
                features["wind_q4_ratio"] = q4
            except Exception as e:
                logger.error(f"Error calculating wind features: {e}")
        
        return features
    
    def _extract_pattern_features(self, grib_data):
        """
        Extract pattern-based features from meteorological data.
        
        Args:
            grib_data (dict): Data from a GRIB file
            
        Returns:
            dict: Pattern-based features
        """
        features = {}
        
        for key, data in grib_data.items():
            if 'metadata' not in data or 'name' not in data['metadata']:
                continue
                
            var_name = data['metadata']['name']
            values = data['values']
            
            try:
                # Skip small grids
                if values.shape[0] < 10 or values.shape[1] < 10:
                    continue
                
                # Apply Gaussian filter to smooth data
                values_smooth = gaussian_filter(values, sigma=1.0)
                
                # Calculate gradients
                grad_y, grad_x = np.gradient(values_smooth)
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                
                # Gradient features
                features[f"{var_name}_grad_mean"] = np.mean(grad_mag)
                features[f"{var_name}_grad_max"] = np.max(grad_mag)
                
                # Calculate frontal detection metric
                front_metric = np.abs(grad_mag) > np.percentile(grad_mag, 90)
                features[f"{var_name}_front_ratio"] = np.sum(front_metric) / front_metric.size
                
                # Detect linear features
                linear_filter = np.zeros_like(values_smooth)
                for angle in range(0, 180, 15):
                    # Create a simple linear filter at given angle
                    theta = angle * np.pi / 180
                    x, y = np.cos(theta), np.sin(theta)
                    template = np.zeros((5, 5))
                    center = np.array([2, 2])
                    for i in range(5):
                        for j in range(5):
                            pos = np.array([i, j]) - center
                            dist = np.abs(pos[0]*y - pos[1]*x)
                            if dist < 0.5:
                                template[i, j] = 1
                    
                    # Apply filter
                    from scipy.signal import convolve2d
                    result = convolve2d(values_smooth, template, mode='same')
                    linear_filter = np.maximum(linear_filter, result)
                
                # Linear feature metric
                linear_metric = linear_filter > np.percentile(linear_filter, 95)
                features[f"{var_name}_linear_ratio"] = np.sum(linear_metric) / linear_metric.size
                
                # Detect cellular patterns
                from skimage.feature import peak_local_max
                peaks = peak_local_max(values_smooth, min_distance=5)
                features[f"{var_name}_peak_count"] = len(peaks) / (values.shape[0] * values.shape[1])
                
            except Exception as e:
                logger.error(f"Error calculating pattern features for {var_name}: {e}")
        
        return features
    
    def _extract_trend_features(self, file_name, grib_data):
        """
        Extract temporal trend features by comparing with previous data.
        
        Args:
            file_name (str): Name of the file
            grib_data (dict): Data from a GRIB file
            
        Returns:
            dict: Temporal trend features
        """
        features = {}
        
        # Extract date and time information from filename
        # Assuming format like *_YYYYMMDD_tHHz_*
        try:
            import re
            match = re.search(r'_(\d{8})_t(\d{2})z_', file_name)
            if not match:
                return features
                
            date_str = match.group(1)
            hour_str = match.group(2)
            
            # Calculate previous time step
            current_dt = datetime.strptime(f"{date_str}{hour_str}", "%Y%m%d%H")
            prev_dt = current_dt - timedelta(hours=6)
            prev_date_str = prev_dt.strftime("%Y%m%d")
            prev_hour_str = prev_dt.strftime("%H")
            
            # Look for previous file
            prev_pattern = f"*_{prev_date_str}_t{prev_hour_str}z_*"
            import glob
            prev_files = glob.glob(os.path.join(self.data_dir, prev_pattern))
            
            if not prev_files:
                return features
                
            # Load previous data
            # This is a simplified approach; in practice, you'd need to ensure
            # you're comparing the same variable types
            prev_file = prev_files[0]
            
            # For demonstration, we'll just compute a simple change metric
            for key, data in grib_data.items():
                if 'metadata' not in data or 'name' not in data['metadata']:
                    continue
                    
                var_name = data['metadata']['name']
                current_mean = np.mean(data['values'])
                
                features[f"{var_name}_change_6h"] = 0.0  # Default value
                features[f"{var_name}_change_rate"] = 0.0  # Default value
                
                # In a real implementation, you would load the previous data
                # and calculate actual changes
                
        except Exception as e:
            logger.error(f"Error calculating trend features: {e}")
        
        return features
    
    def _add_climate_index_features(self, file_name):
        """
        Add climate index correlations.
        
        Args:
            file_name (str): Name of the file
            
        Returns:
            dict: Climate index features
        """
        features = {}
        
        if self.climate_indices is None:
            return features
            
        try:
            # Extract date from filename
            import re
            match = re.search(r'_(\d{8})_', file_name)
            if not match:
                return features
                
            date_str = match.group(1)
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            
            # Find closest date in climate indices
            if 'date' in self.climate_indices.columns:
                # Convert indices dates to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(self.climate_indices['date']):
                    self.climate_indices['date'] = pd.to_datetime(self.climate_indices['date'])
                
                # Find closest date
                closest_idx = np.argmin(np.abs(self.climate_indices['date'] - date_obj))
                
                # Add relevant indices as features
                for col in self.climate_indices.columns:
                    if col != 'date':
                        features[f"climate_{col}"] = float(self.climate_indices.iloc[closest_idx][col])
        
        except Exception as e:
            logger.error(f"Error adding climate index features: {e}")
        
        return features
    
    def _add_terrain_interaction_features(self, grib_data):
        """
        Add terrain interaction features.
        
        Args:
            grib_data (dict): Data from a GRIB file
            
        Returns:
            dict: Terrain interaction features
        """
        features = {}
        
        if self.elevation_data is None:
            return features
            
        try:
            # Find wind data if available
            wind_u_data = None
            wind_v_data = None
            
            for key, data in grib_data.items():
                if 'metadata' not in data or 'name' not in data['metadata']:
                    continue
                    
                name = data['metadata']['name'].lower()
                
                if 'u-component' in name:
                    wind_u_data = data
                elif 'v-component' in name:
                    wind_v_data = data
            
            # Calculate terrain-flow interactions if wind data available
            if wind_u_data is not None and wind_v_data is not None:
                # In practice, you'd need to reproject the elevation data
                # to match the grid of the weather data
                
                # Placeholder for terrain interaction features
                features["terrain_upslope_flow"] = 0.5  # Example value
                features["terrain_blockage"] = 0.3  # Example value
                features["terrain_channeling"] = 0.7  # Example value
        
        except Exception as e:
            logger.error(f"Error adding terrain interaction features: {e}")
        
        return features
    
    def create_ensemble_features(self, grib_files):
        """
        Create ensemble features from multiple GRIB files.
        
        Args:
            grib_files (list): List of GRIB files
            
        Returns:
            dict: Ensemble features
        """
        features = {}
        
        try:
            # Extract features from each file
            all_file_features = []
            
            for file_path in tqdm(grib_files, desc="Processing ensemble members"):
                file_name = os.path.basename(file_path)
                
                # Load GRIB data (simplified)
                grib_data = self._load_grib_data(file_path)
                
                if grib_data:
                    # Extract features
                    file_features = self.extract_advanced_features(grib_data, file_name)
                    all_file_features.append(file_features)
            
            # Calculate ensemble statistics
            if all_file_features:
                # Create DataFrame for easier processing
                ensemble_df = pd.DataFrame(all_file_features)
                
                # Calculate statistics across ensemble members
                for col in ensemble_df.columns:
                    features[f"{col}_mean"] = ensemble_df[col].mean()
                    features[f"{col}_std"] = ensemble_df[col].std()
                    features[f"{col}_min"] = ensemble_df[col].min()
                    features[f"{col}_max"] = ensemble_df[col].max()
                    features[f"{col}_range"] = ensemble_df[col].max() - ensemble_df[col].min()
                    
                    # Calculate probability-based features
                    if 'tornado' in col or 'wind' in col or 'hail' in col:
                        threshold = 0.5  # Example threshold
                        prob = (ensemble_df[col] > threshold).mean()
                        features[f"{col}_prob"] = prob
        
        except Exception as e:
            logger.error(f"Error creating ensemble features: {e}")
        
        return features
    
    def _load_grib_data(self, file_path):
        """
        Load data from a GRIB file.
        
        Args:
            file_path (str): Path to the GRIB file
            
        Returns:
            dict: GRIB data
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"GRIB file not found: {file_path}")
                return {}
                
            grbs = pygrib.open(file_path)
            data = {}
            
            for grb in grbs:
                values, lats, lons = grb.data()
                name = grb.name
                level = grb.level
                forecast_time = grb.forecastTime
                
                key = f"{name}_{level}_{forecast_time}"
                data[key] = {
                    'values': values,
                    'lats': lats,
                    'lons': lons,
                    'metadata': {
                        'name': name,
                        'level': level,
                        'forecast_time': forecast_time,
                        'units': grb.units
                    }
                }
            
            grbs.close()
            return data
            
        except Exception as e:
            logger.error(f"Error loading GRIB file {file_path}: {e}")
            return {}
    
    def save_feature_cache(self, cache_path='features_cache.joblib'):
        """Save feature cache to disk."""
        try:
            joblib.dump(self.feature_cache, cache_path)
            logger.info(f"Feature cache saved to {cache_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving feature cache: {e}")
            return False
    
    def load_feature_cache(self, cache_path='features_cache.joblib'):
        """Load feature cache from disk."""
        if os.path.exists(cache_path):
            try:
                self.feature_cache = joblib.load(cache_path)
                logger.info(f"Feature cache loaded from {cache_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading feature cache: {e}")
        return False

# Create TensorFlow feature columns for the advanced features
def create_feature_columns(feature_names):
    """
    Create TensorFlow feature columns for model input.
    
    Args:
        feature_names (list): List of feature names
        
    Returns:
        list: TensorFlow feature columns
    """
    feature_columns = []
    
    # Categorize features
    basic_features = [f for f in feature_names if not any(p in f for p in ['_prob', '_grad', '_front', '_linear', '_peak', '_change'])]
    pattern_features = [f for f in feature_names if any(p in f for p in ['_grad', '_front', '_linear', '_peak'])]
    trend_features = [f for f in feature_names if '_change' in f]
    prob_features = [f for f in feature_names if '_prob' in f]
    
    # Add numeric columns for all features
    for feature in feature_names:
        feature_columns.append(tf.feature_column.numeric_column(feature))
    
    # Create bucketized columns for selected features
    bucketized_features = []
    for feature in basic_features[:5]:  # Just use a few features for bucketization
        boundaries = list(np.linspace(0, 1, 10))  # Example boundaries
        bucketized_col = tf.feature_column.bucketized_column(
            tf.feature_column.numeric_column(feature), boundaries)
        bucketized_features.append(bucketized_col)
    
    # Create crossed columns for selected feature pairs
    # This helps the model learn interactions between features
    if len(basic_features) >= 2:
        for i in range(min(5, len(basic_features))):
            for j in range(i+1, min(5, len(basic_features))):
                crossed_col = tf.feature_column.crossed_column(
                    [basic_features[i], bucketized_features[j]], hash_bucket_size=1000)
                feature_columns.append(tf.feature_column.indicator_column(crossed_col))
    
    return feature_columns

# Function to demonstrate advanced feature extraction
def demo():
    """Demonstrate advanced feature extraction."""
    # Create feature extractor
    extractor = AdvancedFeatureExtractor()
    
    # Find a sample GRIB file
    import glob
    grib_files = glob.glob('data/*/*/*/*/*.grib2')
    
    if grib_files:
        sample_file = grib_files[0]
        logger.info(f"Using sample file: {sample_file}")
        
        # Load GRIB data
        grib_data = extractor._load_grib_data(sample_file)
        
        if grib_data:
            # Extract features
            features = extractor.extract_advanced_features(grib_data, os.path.basename(sample_file))
            
            # Print features
            logger.info(f"Extracted {len(features)} advanced features")
            for i, (key, value) in enumerate(features.items()):
                logger.info(f"  {key}: {value}")
                if i >= 10:
                    logger.info(f"  ... and {len(features) - 10} more features")
                    break
    else:
        logger.warning("No GRIB files found")

if __name__ == "__main__":
    demo() 