#!/usr/bin/env python3
"""
Weather Prediction System
========================
This module initializes the weather prediction system, sets up TensorFlow models,
and provides functionality to train models and make predictions using the weather data.
"""

import os
import numpy as np
import tensorflow as tf # type: ignore
from tensorflow.keras import layers, models, callbacks # type: ignore
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import logging
import cartopy.crs as ccrs
from PIL import Image
import argparse
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import project modules
try:
    from src.data_loader import WeatherDataLoader
    from src.data_processor import DataProcessor
    from src.model_trainer import ModelTrainer
    from src.visualizer import Visualizer
    from src.weather_bot import WeatherBot
except ImportError:
    logger.warning("Could not import some modules from src directory. This is normal if running for the first time.")
    # Create placeholder classes that will be properly initialized later
    class WeatherDataLoader:
        def __init__(self, data_dir):
            self.data_dir = data_dir
    
    class DataProcessor:
        pass
    
    class ModelTrainer:
        def __init__(self, models_dir):
            self.models_dir = models_dir
    
    class Visualizer:
        def __init__(self, visualizations_dir):
            self.visualizations_dir = visualizations_dir
    
    class WeatherBot:
        pass

# Set tensorflow logging level
tf.get_logger().setLevel('ERROR')

# Apply CPU optimizations
def apply_cpu_optimizations():
    """Apply CPU-specific optimizations."""
    # Disable GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Set OpenMP, MKL, and other CPU threading settings
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    os.environ['NUMEXPR_NUM_THREADS'] = '4'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
    
    # Configure TensorFlow for CPU
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    
    # Enable OneDNN optimizations for CPU
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
    
    logger.info("Applied CPU optimizations")

class WeatherPredictionSystem:
    """Main class for the weather prediction system."""
    
    def __init__(self, data_dir='data', models_dir='models', visualizations_dir='visualizations'):
        """
        Initialize the weather prediction system.
        
        Args:
            data_dir (str): Directory containing weather data
            models_dir (str): Directory to store trained models
            visualizations_dir (str): Directory to store visualizations
        """
        self.data_dir = os.path.abspath(data_dir)
        self.models_dir = os.path.abspath(models_dir)
        self.visualizations_dir = os.path.abspath(visualizations_dir)
        
        # Create directories if they don't exist
        for directory in [self.models_dir, self.visualizations_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Initialize components
        self.data_loader = WeatherDataLoader(data_dir=self.data_dir)
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer(models_dir=self.models_dir)
        self.visualizer = Visualizer(visualizations_dir=self.visualizations_dir)
        self.weather_bot = WeatherBot()
        
        # Initialize TensorFlow models
        self.models = {
            'tornado': None,
            'wind': None,
            'hail': None,
            'temperature': None,
            'precipitation': None
        }
        
        logger.info("Weather prediction system initialized")
    
    def load_data(self, start_date=None, end_date=None, times_of_day=['t0z', 't12z', 't18z']):
        """
        Load and process data for the given date range.
        
        Args:
            start_date (str): Start date in YYYYMMDD format
            end_date (str): End date in YYYYMMDD format
            times_of_day (list): List of times of day to include
            
        Returns:
            dict: Processed dataset
        """
        logger.info(f"Loading data from {start_date} to {end_date}")
        
        # If dates not provided, use all available data
        if not start_date or not end_date:
            available_dates = self.data_loader.get_available_dates()
            if available_dates:
                if not start_date:
                    start_date = available_dates[0]
                if not end_date:
                    end_date = available_dates[-1]
        
        # Load raw data
        raw_dataset = self.data_loader.create_dataset(start_date, end_date, times_of_day)
        
        # Process the data
        processed_dataset = self.data_processor.process_dataset(raw_dataset)
        
        return processed_dataset
    
    def create_tf_feature_dataset(self, dataset, feature_type='all', split_ratio=0.8):
        """
        Create TensorFlow dataset with engineered features.
        
        Args:
            dataset (dict): Processed dataset
            feature_type (str): Type of features to include ('all', 'tornado', 'wind', 'hail')
            split_ratio (float): Train/test split ratio
            
        Returns:
            tuple: (train_dataset, test_dataset, feature_names)
        """
        logger.info(f"Creating TensorFlow dataset for {feature_type} features")
        
        # Feature engineering
        X, y, feature_names = self._engineer_features(dataset, feature_type)
        
        # Split data
        split_idx = int(len(X) * split_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        
        # Batch and prefetch for performance
        train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, test_dataset, feature_names
    
    def _engineer_features(self, dataset, feature_type='all'):
        """
        Engineer features from the processed dataset.
        
        Args:
            dataset (dict): Processed dataset
            feature_type (str): Type of features to include
            
        Returns:
            tuple: (X, y, feature_names)
        """
        all_features = []
        all_labels = []
        feature_names = []
        
        # Extract data from dataset
        for date, date_data in dataset.items():
            for tod, tod_data in date_data.items():
                for file_name, file_data in tod_data.items():
                    # Extract features based on type
                    features, labels, names = self._extract_features_from_file(file_data, file_name, feature_type)
                    
                    if features is not None and labels is not None:
                        all_features.append(features)
                        all_labels.append(labels)
                        
                        if not feature_names and names:
                            feature_names = names
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        # Add temporal features
        X, feature_names = self._add_temporal_features(X, dataset, feature_names)
        
        # Add spatial features
        X, feature_names = self._add_spatial_features(X, dataset, feature_names)
        
        # Normalize features
        X = self._normalize_features(X)
        
        return X, y, feature_names
    
    def _extract_features_from_file(self, file_data, file_name, feature_type):
        """
        Extract features from a file.
        
        Args:
            file_data (dict): Data from a single file
            file_name (str): Name of the file
            feature_type (str): Type of features to include
            
        Returns:
            tuple: (features, labels, feature_names)
        """
        features = []
        feature_names = []
        labels = None
        
        # Skip if not relevant to the feature type
        if feature_type != 'all' and feature_type not in file_name:
            return None, None, None
        
        # Extract features from each variable in the file
        for key, data in file_data.items():
            values = data['values'].flatten()
            
            # Basic statistics
            features.extend([
                np.mean(values),
                np.std(values),
                np.min(values),
                np.max(values),
                np.median(values),
                np.percentile(values, 90),
                np.percentile(values, 10)
            ])
            
            # Add feature names
            base_name = f"{data['metadata']['name']}_{data['metadata']['level']}"
            feature_names.extend([
                f"{base_name}_mean",
                f"{base_name}_std",
                f"{base_name}_min",
                f"{base_name}_max",
                f"{base_name}_median",
                f"{base_name}_p90",
                f"{base_name}_p10"
            ])
            
            # Determine labels based on file name
            if 'tornado' in file_name:
                labels = 1 if np.max(values) > 0.5 else 0  # Example threshold
            elif 'wind' in file_name:
                labels = 1 if np.max(values) > 40 else 0   # Example threshold
            elif 'hail' in file_name:
                labels = 1 if np.max(values) > 0.3 else 0  # Example threshold
            else:
                # Default label if type not recognized
                labels = 1 if np.max(values) > 0.5 else 0
        
        return np.array(features), labels, feature_names
    
    def _add_temporal_features(self, X, dataset, feature_names):
        """
        Add temporal features to the feature set.
        
        Args:
            X (numpy.ndarray): Feature array
            dataset (dict): Processed dataset
            feature_names (list): List of feature names
            
        Returns:
            tuple: (X_with_temporal, feature_names_with_temporal)
        """
        # Get dates from dataset
        dates = list(dataset.keys())
        
        # Convert to datetime objects
        dt_dates = [datetime.strptime(date, '%Y%m%d') for date in dates]
        
        # Extract day of year, month, season features
        doy = np.array([d.timetuple().tm_yday for d in dt_dates])
        month = np.array([d.month for d in dt_dates])
        
        # Calculate season (0:Winter, 1:Spring, 2:Summer, 3:Fall)
        season = np.array([(d.month % 12 + 3) // 3 - 1 for d in dt_dates])
        
        # Create new feature array with temporal features
        X_temporal = np.column_stack((X, doy, month, season))
        
        # Add feature names
        feature_names_temporal = feature_names + ['day_of_year', 'month', 'season']
        
        return X_temporal, feature_names_temporal
    
    def _add_spatial_features(self, X, dataset, feature_names):
        """
        Add spatial features to the feature set.
        
        Args:
            X (numpy.ndarray): Feature array
            dataset (dict): Processed dataset
            feature_names (list): List of feature names
            
        Returns:
            tuple: (X_with_spatial, feature_names_with_spatial)
        """
        # Example implementation - in a real scenario, this would use actual spatial data
        # from grib files to calculate regional means, gradients, etc.
        
        # For now, we'll just add placeholder features
        region_features = np.zeros((X.shape[0], 4))
        
        # Add feature names
        feature_names_spatial = feature_names + [
            'region_ne', 'region_se', 'region_sw', 'region_nw'
        ]
        
        # Create new feature array with spatial features
        X_spatial = np.column_stack((X, region_features))
        
        return X_spatial, feature_names_spatial
    
    def _normalize_features(self, X):
        """
        Normalize features to have zero mean and unit variance.
        
        Args:
            X (numpy.ndarray): Feature array
            
        Returns:
            numpy.ndarray: Normalized features
        """
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        
        # Avoid division by zero
        stds[stds == 0] = 1.0
        
        return (X - means) / stds
    
    def build_tensorflow_models(self):
        """
        Build TensorFlow models for different prediction tasks.
        """
        logger.info("Building TensorFlow models")
        
        # Common model parameters
        model_params = {
            'activation': 'relu',
            'dropout_rate': 0.3,
            'l2_reg': 1e-4
        }
        
        # Build models for each prediction task
        self.models['tornado'] = self._build_binary_classification_model(name='tornado', **model_params)
        self.models['wind'] = self._build_binary_classification_model(name='wind', **model_params)
        self.models['hail'] = self._build_binary_classification_model(name='hail', **model_params)
        self.models['temperature'] = self._build_regression_model(name='temperature', **model_params)
        self.models['precipitation'] = self._build_regression_model(name='precipitation', **model_params)
        
        logger.info("TensorFlow models built successfully")
    
    def _build_binary_classification_model(self, name, activation='relu', dropout_rate=0.3, l2_reg=1e-4):
        """
        Build a binary classification model.
        
        Args:
            name (str): Model name
            activation (str): Activation function
            dropout_rate (float): Dropout rate
            l2_reg (float): L2 regularization
            
        Returns:
            tensorflow.keras.Model: Binary classification model
        """
        regularizer = tf.keras.regularizers.l2(l2_reg)
        
        model = models.Sequential([
            layers.Dense(128, activation=activation, kernel_regularizer=regularizer),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(64, activation=activation, kernel_regularizer=regularizer),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(32, activation=activation, kernel_regularizer=regularizer),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(1, activation='sigmoid')
        ], name=f"{name}_model")
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def _build_regression_model(self, name, activation='relu', dropout_rate=0.3, l2_reg=1e-4):
        """
        Build a regression model.
        
        Args:
            name (str): Model name
            activation (str): Activation function
            dropout_rate (float): Dropout rate
            l2_reg (float): L2 regularization
            
        Returns:
            tensorflow.keras.Model: Regression model
        """
        regularizer = tf.keras.regularizers.l2(l2_reg)
        
        model = models.Sequential([
            layers.Dense(128, activation=activation, kernel_regularizer=regularizer),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(64, activation=activation, kernel_regularizer=regularizer),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(32, activation=activation, kernel_regularizer=regularizer),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(1)  # No activation for regression
        ], name=f"{name}_model")
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mean_absolute_error', 'mean_squared_error']
        )
        
        return model
    
    def build_convolutional_model(self, input_shape):
        """
        Build a convolutional model for spatial weather data.
        
        Args:
            input_shape (tuple): Input shape for the model (height, width, channels)
            
        Returns:
            tensorflow.keras.Model: Convolutional model
        """
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ], name='spatial_model')
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    def train_model(self, model_name, train_dataset, test_dataset, epochs=20):
        """
        Train a specific model.
        
        Args:
            model_name (str): Name of the model to train
            train_dataset (tf.data.Dataset): Training dataset
            test_dataset (tf.data.Dataset): Testing dataset
            epochs (int): Number of training epochs
            
        Returns:
            tensorflow.keras.callbacks.History: Training history
        """
        if model_name not in self.models or self.models[model_name] is None:
            logger.error(f"Model {model_name} not found")
            return None
        
        logger.info(f"Training {model_name} model")
        
        # Setup callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            callbacks.ModelCheckpoint(
                filepath=os.path.join(self.models_dir, f"{model_name}_best.h5"),
                monitor='val_loss',
                save_best_only=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        # Train the model
        history = self.models[model_name].fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=epochs,
            callbacks=callbacks_list
        )
        
        # Save the final model
        self.models[model_name].save(os.path.join(self.models_dir, f"{model_name}_final.h5"))
        
        logger.info(f"{model_name} model training completed")
        
        return history
    
    def evaluate_model(self, model_name, test_dataset):
        """
        Evaluate a trained model.
        
        Args:
            model_name (str): Name of the model to evaluate
            test_dataset (tf.data.Dataset): Testing dataset
            
        Returns:
            dict: Evaluation metrics
        """
        if model_name not in self.models or self.models[model_name] is None:
            logger.error(f"Model {model_name} not found")
            return None
        
        logger.info(f"Evaluating {model_name} model")
        
        # Evaluate the model
        metrics = self.models[model_name].evaluate(test_dataset)
        
        # Create metrics dictionary
        metrics_dict = dict(zip(self.models[model_name].metrics_names, metrics))
        
        logger.info(f"{model_name} model evaluation: {metrics_dict}")
        
        return metrics_dict
    
    def load_trained_model(self, model_name):
        """
        Load a trained model from disk.
        
        Args:
            model_name (str): Name of the model to load
            
        Returns:
            bool: True if successful, False otherwise
        """
        model_path = os.path.join(self.models_dir, f"{model_name}_best.h5")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        try:
            self.models[model_name] = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded model {model_name} from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    def predict(self, model_name, features):
        """
        Make predictions with a trained model.
        
        Args:
            model_name (str): Name of the model to use
            features (numpy.ndarray): Features to use for prediction
            
        Returns:
            numpy.ndarray: Predictions
        """
        if model_name not in self.models or self.models[model_name] is None:
            logger.error(f"Model {model_name} not found")
            return None
        
        # Make predictions
        predictions = self.models[model_name].predict(features)
        
        return predictions
    
    def create_ensemble_model(self, model_names, input_shape):
        """
        Create an ensemble model from multiple trained models.
        
        Args:
            model_names (list): List of model names to include in the ensemble
            input_shape (tuple): Input shape for the ensemble model
            
        Returns:
            tensorflow.keras.Model: Ensemble model
        """
        # Check if all models exist
        for name in model_names:
            if name not in self.models or self.models[name] is None:
                logger.error(f"Model {name} not found")
                return None
        
        # Create input layer
        inputs = tf.keras.Input(shape=input_shape)
        
        # Get outputs from each model
        outputs = []
        for name in model_names:
            # Use functional API to get the output of each model
            model_output = self.models[name](inputs)
            outputs.append(model_output)
        
        # Concatenate outputs
        concat = layers.Concatenate()(outputs)
        
        # Add final layers
        x = layers.Dense(64, activation='relu')(concat)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        predictions = layers.Dense(1, activation='sigmoid')(x)
        
        # Create ensemble model
        ensemble_model = tf.keras.Model(inputs=inputs, outputs=predictions)
        
        # Compile model
        ensemble_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        # Save ensemble model to models directory
        ensemble_model.save(os.path.join(self.models_dir, "ensemble_model.h5"))
        
        logger.info("Ensemble model created")
        
        return ensemble_model
    
    def run_full_pipeline(self, start_date=None, end_date=None, feature_type='all', epochs=20):
        """
        Run the full pipeline: load data, create datasets, train models, and evaluate.
        
        Args:
            start_date (str): Start date in YYYYMMDD format
            end_date (str): End date in YYYYMMDD format
            feature_type (str): Type of features to include
            epochs (int): Number of training epochs
            
        Returns:
            dict: Results including evaluation metrics for all models
        """
        # Load and process data
        dataset = self.load_data(start_date, end_date)
        
        # Create TensorFlow datasets
        train_dataset, test_dataset, feature_names = self.create_tf_feature_dataset(dataset, feature_type)
        
        # Build models
        self.build_tensorflow_models()
        
        # Train models
        results = {}
        for model_name in self.models:
            # Train model
            history = self.train_model(model_name, train_dataset, test_dataset, epochs)
            
            # Evaluate model
            metrics = self.evaluate_model(model_name, test_dataset)
            
            # Store results
            results[model_name] = {
                'history': history.history if history else None,
                'metrics': metrics
            }
        
        # Create visualizations for results
        self.visualizer.plot_training_history(results)
        self.visualizer.plot_model_comparison(results)
        
        logger.info("Full pipeline completed successfully")
        
        return results

    def visualize_data(self, date_str=None, time_of_day='t0z'):
        """
        Create visualizations for weather data.
        
        Args:
            date_str (str): Date in YYYYMMDD format, or None to use most recent
            time_of_day (str): Time of day (t0z, t12z, etc.)
        """
        logger.info(f"Creating visualizations for date: {date_str}, time: {time_of_day}")
        
        # If no date specified, use most recent
        if not date_str:
            available_dates = self.data_loader.get_available_dates()
            if not available_dates:
                logger.error("No data available")
                return
            date_str = available_dates[-1]
        
        # Create visualization directory
        viz_dir = os.path.join(self.visualizations_dir, date_str)
        os.makedirs(viz_dir, exist_ok=True)
        
        # Get data for the specified date and time
        year_month = date_str[:6]
        date_path = os.path.join(self.data_dir, year_month, date_str, time_of_day)
        
        if not os.path.exists(date_path):
            logger.error(f"No data found for date: {date_str}, time: {time_of_day}")
            return
        
        # Find images and GRIB files
        images = glob.glob(os.path.join(date_path, "*.png"))
        gribs = glob.glob(os.path.join(date_path, "*.grib2"))
        
        # Copy images to visualization directory
        for img_path in images:
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(viz_dir, img_name)
            try:
                # Load and save image
                img = Image.open(img_path)
                img.save(dest_path)
                logger.info(f"Saved visualization: {dest_path}")
            except Exception as e:
                logger.error(f"Error copying image {img_path}: {e}")
        
        # Create visualizations from GRIB files
        for grib_path in gribs:
            try:
                grib_name = os.path.basename(grib_path)
                base_name = os.path.splitext(grib_name)[0]
                
                # Load GRIB data
                grib_data = self.data_loader.load_grib_data(grib_path)
                
                if not grib_data:
                    continue
                
                # Create visualization for each variable
                for key, data in grib_data.items():
                    if 'values' not in data or 'lats' not in data or 'lons' not in data:
                        continue
                    
                    # Create a more descriptive name
                    var_name = data['metadata']['name'] if 'metadata' in data and 'name' in data['metadata'] else key
                    viz_name = f"{base_name}_{var_name}.png"
                    viz_path = os.path.join(viz_dir, viz_name)
                    
                    # Create plot
                    plt.figure(figsize=(10, 8))
                    
                    # Use Cartopy for map projection
                    ax = plt.axes(projection=ccrs.PlateCarree())
                    ax.coastlines()
                    ax.gridlines(draw_labels=True)
                    
                    # Plot data with color mapping
                    values = data['values']
                    lats = data['lats']
                    lons = data['lons']
                    
                    # Normalize values for better visualization
                    vmin = np.min(values)
                    vmax = np.max(values)
                    
                    # Plot data
                    plt.contourf(lons, lats, values, 
                                 levels=np.linspace(vmin, vmax, 20),
                                 transform=ccrs.PlateCarree(),
                                 cmap='viridis')
                    
                    plt.colorbar(label=data['metadata'].get('units', ''))
                    plt.title(f"{var_name} - {date_str} {time_of_day}")
                    
                    # Save plot
                    plt.savefig(viz_path)
                    plt.close()
                    
                    logger.info(f"Created visualization: {viz_path}")
            
            except Exception as e:
                logger.error(f"Error creating visualization for {grib_path}: {e}")
        
        logger.info(f"Visualizations created in {viz_dir}")

# Function to run the weather prediction system
def run(args=None):
    """
    Run the weather prediction system.
    
    Args:
        args: Command-line arguments
    """
    # Parse command-line arguments if not provided
    if args is None:
        parser = argparse.ArgumentParser(description="Run Weather Prediction System")
        parser.add_argument('--data-dir', type=str, default='data',
                            help='Directory containing weather data')
        parser.add_argument('--models-dir', type=str, default='models',
                            help='Directory to store trained models')
        parser.add_argument('--visualizations-dir', type=str, default='visualizations',
                            help='Directory to store visualizations')
        parser.add_argument('--date', type=str, default=None,
                            help='Date to visualize (YYYYMMDD format)')
        parser.add_argument('--time', type=str, default='t0z',
                            help='Time of day to visualize (t0z, t12z, etc.)')
        parser.add_argument('--cpu-only', action='store_true',
                            help='Force CPU-only mode even if GPU is available')
        parser.add_argument('--train', action='store_true',
                            help='Train models')
        
        args = parser.parse_args()
    
    # Apply CPU optimizations if requested
    if args.cpu_only:
        apply_cpu_optimizations()
    
    # Create the weather prediction system
    system = WeatherPredictionSystem(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        visualizations_dir=args.visualizations_dir
    )
    
    # Create visualizations
    system.visualize_data(args.date, args.time)
    
    # Train models if requested
    if args.train:
        # Load data
        available_dates = system.data_loader.get_available_dates()
        if not available_dates:
            logger.error("No data available for training")
            return
        
        # Use the last 90 days or all available data if less
        if len(available_dates) > 90:
            start_date = available_dates[-90]
        else:
            start_date = available_dates[0]
        
        end_date = available_dates[-1]
        
        logger.info(f"Training models using data from {start_date} to {end_date}")
        
        # Run the full pipeline
        results = system.run_full_pipeline(start_date, end_date)
        
        # Create ensemble model
        try:
            # Determine input shape
            dataset = system.load_data(start_date, end_date)
            train_dataset, test_dataset, feature_names = system.create_tf_feature_dataset(
                dataset, feature_type='all'
            )
            
            for x, y in train_dataset.take(1):
                input_shape = x.shape[1:]
                break
            
            # Create and train ensemble model
            model_names = list(system.models.keys())
            ensemble_model = system.create_ensemble_model(model_names, input_shape)
            
            if ensemble_model:
                logger.info("Training ensemble model")
                history = system.train_model('ensemble', train_dataset, test_dataset, epochs=20)
                metrics = system.evaluate_model('ensemble', test_dataset)
                
                results['ensemble'] = {
                    'history': history.history if history else None,
                    'metrics': metrics
                }
                
                # Update visualizations
                system.visualizer.plot_training_history(results)
                system.visualizer.plot_model_comparison(results)
        except Exception as e:
            logger.error(f"Error creating ensemble model: {e}")
    
    logger.info("Weather prediction system run completed")
    return system

# If run as main script
if __name__ == "__main__":
    # Try to install dependencies using setup.py if available
    try:
        import setup
        setup.main()
    except ImportError:
        logger.warning("setup.py not found or could not be imported. Skipping dependency installation.")
    
    # Add current directory to Python path to ensure modules can be found
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Try to import advanced features if available
    try:
        import advanced_features
        logger.info("Advanced features module imported successfully")
    except ImportError:
        logger.warning("advanced_features.py not found or could not be imported. Using basic features only.")
    
    # Run the system
    run() 