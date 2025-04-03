#!/usr/bin/env python3

import os
import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Import custom modules
from data_loader import WeatherDataLoader
from data_processor import WeatherDataProcessor
from model_trainer import WeatherModelTrainer
from visualizer import WeatherVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeatherPredictionBot:
    """
    Main class for the Weather Prediction Bot
    """
    
    def __init__(self, data_dir='/home/jackson/Downloads/forecasts',
                output_dir='/home/jackson/weather_prediction_bot'):
        """
        Initialize the Weather Prediction Bot
        
        Args:
            data_dir (str): Directory containing forecast data
            output_dir (str): Base directory for outputs
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Initialize components
        self.data_loader = WeatherDataLoader(data_dir)
        self.data_processor = WeatherDataProcessor(os.path.join(output_dir, 'data'))
        self.model_trainer = WeatherModelTrainer(os.path.join(output_dir, 'models'))
        self.visualizer = WeatherVisualizer(os.path.join(output_dir, 'visualizations'))
        
        # Storage for data and models
        self.dataset = None
        self.processed_data = None
        self.trained_models = None
        self.best_model_name = None
        
        logger.info("Weather Prediction Bot initialized")
    
    def load_data(self, start_date=None, end_date=None, limit=None):
        """
        Load and prepare the dataset
        
        Args:
            start_date (str): Start date in YYYYMMDD format
            end_date (str): End date in YYYYMMDD format
            limit (int): Limit number of dates to load (for testing)
            
        Returns:
            dict: Loaded dataset
        """
        logger.info(f"Loading data from {start_date} to {end_date}")
        
        if start_date is None:
            # Default to earliest available data
            available_dates = self.data_loader.get_available_dates()
            if available_dates:
                start_date = available_dates[0]
        
        if end_date is None:
            # Default to latest available data
            available_dates = self.data_loader.get_available_dates()
            if available_dates:
                end_date = available_dates[-1]
        
        # Create dataset
        self.dataset = self.data_loader.create_dataset(start_date, end_date)
        
        # Apply limit if specified
        if limit and isinstance(limit, int):
            dates = list(self.dataset.keys())
            if len(dates) > limit:
                selected_dates = dates[:limit]
                self.dataset = {date: self.dataset[date] for date in selected_dates}
                logger.info(f"Limited dataset to {limit} dates")
        
        logger.info(f"Loaded {len(self.dataset)} dates of forecast data")
        return self.dataset
    
    def process_data(self, forecast_horizon=24):
        """
        Process the loaded dataset
        
        Args:
            forecast_horizon (int): Forecast horizon in hours
            
        Returns:
            dict: Processed data
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        logger.info("Processing data...")
        self.processed_data = self.data_processor.process_data(
            self.dataset, 
            forecast_horizon=forecast_horizon
        )
        
        return self.processed_data
    
    def train_models(self, task='regression', target_col='target_max_hail'):
        """
        Train prediction models
        
        Args:
            task (str): 'regression' or 'classification'
            target_col (str): Target column to predict
            
        Returns:
            dict: Training results
        """
        if self.processed_data is None:
            raise ValueError("Data not processed. Call process_data() first.")
        
        logger.info(f"Training models for {task} task on target {target_col}")
        
        training_results = self.model_trainer.train_models(
            self.processed_data['data'],
            target_col=target_col,
            task=task
        )
        
        self.trained_models = training_results['models']
        self.best_model_name = training_results['best_model']
        
        # Save all models
        self.model_trainer.save_models()
        
        return training_results
    
    def create_visualizations(self, dates=None, limit=10):
        """
        Create visualizations of predictions
        
        Args:
            dates (list): List of dates to visualize (defaults to test set dates)
            limit (int): Limit number of visualizations to create
            
        Returns:
            list: Paths to created visualizations
        """
        if self.trained_models is None:
            raise ValueError("Models not trained. Call train_models() first.")
        
        logger.info("Creating visualizations...")
        
        # Get test data
        X_test = self.processed_data['data']['test']
        
        # If dates not specified, use test set dates
        if dates is None:
            dates = self.processed_data['data']['test']['date'].values
        
        # Limit the number of dates to visualize
        if limit and len(dates) > limit:
            dates = dates[:limit]
        
        visualizations = []
        
        # 1. Metrics comparison
        evaluations = {}
        for model_name, model in self.trained_models.items():
            if model_name != 'neural_network':  # Skip neural network for standard evaluation
                if 'test' in self.model_trainer._evaluate_model(
                    model, 
                    self.processed_data['data']['test'][X_test.columns], 
                    self.processed_data['data']['test']['target_max_hail']
                ):
                    evaluations[model_name] = self.model_trainer._evaluate_model(
                        model, 
                        self.processed_data['data']['test'][X_test.columns], 
                        self.processed_data['data']['test']['target_max_hail']
                    )
        
        metrics_path = self.visualizer.plot_evaluation_metrics(evaluations)
        visualizations.append(metrics_path)
        
        # 2. Create time series predictions
        # Get test data dates
        test_dates = self.processed_data['data']['test']['date'].values
        
        # Make predictions with each model
        predictions_dict = {}
        for model_name, model in self.trained_models.items():
            if model_name != 'neural_network':  # Skip for now
                X_test_model = self.processed_data['data']['test'][X_test.columns]
                try:
                    predictions = model.predict(X_test_model)
                    predictions_dict[model_name] = predictions
                except Exception as e:
                    logger.error(f"Error making predictions with {model_name}: {e}")
        
        # Get actual values
        actual_values = self.processed_data['data']['test']['target_max_hail'].values
        
        # Create time series comparison
        ts_path = self.visualizer.plot_model_comparison(
            test_dates, 
            predictions_dict, 
            actual=actual_values,
            target_name='Maximum Hail Probability'
        )
        visualizations.append(ts_path)
        
        # 3. Create individual forecast images for selected dates
        for date in dates[:5]:  # Limit to 5 dates for forecast images
            # Get the row for this date
            if isinstance(date, str):
                date_str = date
            else:
                date_str = str(date)
            
            date_row = self.processed_data['data']['test'][
                self.processed_data['data']['test']['date'] == date_str
            ]
            
            if len(date_row) == 0:
                continue
            
            # Make predictions for each model
            date_predictions = {}
            for model_name, model in self.trained_models.items():
                if model_name == 'neural_network':
                    continue  # Skip for now
                    
                try:
                    X_date = date_row[X_test.columns]
                    pred = model.predict(X_date)[0]
                    date_predictions[model_name] = {'hail_probability': float(pred)}
                except Exception as e:
                    logger.error(f"Error making prediction with {model_name} for date {date}: {e}")
            
            # Create forecast image
            forecast_path = self.visualizer.create_forecast_comparison(
                date_str, 
                date_predictions,
                actual={'hail_probability': float(date_row['target_max_hail'].values[0])}
            )
            visualizations.append(forecast_path)
        
        logger.info(f"Created {len(visualizations)} visualizations")
        return visualizations
    
    def make_prediction(self, date, features=None):
        """
        Make a prediction for a specific date
        
        Args:
            date (str): Date in YYYYMMDD format
            features (dict): Feature values (optional)
            
        Returns:
            dict: Prediction results
        """
        # Load the best model
        model_path = os.path.join(self.model_trainer.models_dir, f'best_model_regression.joblib')
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model not found at {model_path}. Train models first.")
        
        best_model = joblib.load(model_path)
        
        # If features not provided, try to load from dataset
        if features is None:
            if self.dataset and date in self.dataset:
                # Extract features from the dataset
                date_data = self.dataset[date]
                
                # This would require a more complex implementation to extract features in the same
                # format expected by the model. For now, we'll just return an error.
                raise NotImplementedError(
                    "Automatic feature extraction from raw data is not implemented. "
                    "Please provide preprocessed features."
                )
            else:
                raise ValueError(f"Date {date} not found in dataset and no features provided.")
        
        # Make prediction
        prediction = best_model.predict([features])[0]
        
        # Create visualization
        viz_path = self.visualizer.create_weather_forecast_image(
            date,
            {'hail_probability': float(prediction)}
        )
        
        return {
            'date': date,
            'prediction': float(prediction),
            'model': self.best_model_name,
            'visualization': viz_path
        }
    
    def run_pipeline(self, start_date=None, end_date=None, data_limit=None, 
                   forecast_horizon=24, task='regression', target_col='target_max_hail'):
        """
        Run the full pipeline: load data, process, train, visualize
        
        Args:
            start_date (str): Start date in YYYYMMDD format
            end_date (str): End date in YYYYMMDD format
            data_limit (int): Limit number of dates to load
            forecast_horizon (int): Forecast horizon in hours
            task (str): 'regression' or 'classification'
            target_col (str): Target column to predict
            
        Returns:
            dict: Results of the pipeline
        """
        try:
            # Load data
            self.load_data(start_date, end_date, limit=data_limit)
            
            # Process data
            self.process_data(forecast_horizon=forecast_horizon)
            
            # Train models
            training_results = self.train_models(task=task, target_col=target_col)
            
            # Create visualizations
            visualizations = self.create_visualizations()
            
            return {
                'status': 'success',
                'dataset_size': len(self.dataset),
                'best_model': self.best_model_name,
                'visualizations': visualizations,
                'training_results': training_results
            }
            
        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }


def main():
    """Main function to run the Weather Prediction Bot from the command line"""
    parser = argparse.ArgumentParser(description='Weather Prediction Bot')
    
    parser.add_argument('--data-dir', type=str, default='/home/jackson/Downloads/forecasts',
                       help='Directory containing forecast data')
    parser.add_argument('--output-dir', type=str, default='/home/jackson/weather_prediction_bot',
                       help='Base directory for outputs')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date in YYYYMMDD format')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date in YYYYMMDD format')
    parser.add_argument('--data-limit', type=int, default=None,
                       help='Limit number of dates to load (for testing)')
    parser.add_argument('--forecast-horizon', type=int, default=24,
                       help='Forecast horizon in hours')
    parser.add_argument('--task', type=str, choices=['regression', 'classification'],
                       default='regression', help='Prediction task type')
    parser.add_argument('--target-col', type=str, default='target_max_hail',
                       help='Target column to predict')
    
    args = parser.parse_args()
    
    # Initialize and run the bot
    bot = WeatherPredictionBot(args.data_dir, args.output_dir)
    
    results = bot.run_pipeline(
        start_date=args.start_date,
        end_date=args.end_date,
        data_limit=args.data_limit,
        forecast_horizon=args.forecast_horizon,
        task=args.task,
        target_col=args.target_col
    )
    
    if results['status'] == 'success':
        logger.info("Weather Prediction Bot completed successfully!")
        logger.info(f"Best model: {results['best_model']}")
        logger.info(f"Created {len(results['visualizations'])} visualizations")
    else:
        logger.error(f"Weather Prediction Bot failed: {results['message']}")


if __name__ == "__main__":
    main() 