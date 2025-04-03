#!/usr/bin/env python3
"""
Weather Prediction System Runner
===============================
This script runs the complete weather prediction system with advanced features
and CPU optimizations.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import local modules
try:
    from __init__ import WeatherPredictionSystem
    from advanced_features import AdvancedFeatureExtractor, create_feature_columns
    from src.spc_integration import SPCIntegration
    from src.spc_data_fetcher import SPCDataFetcher
    from src.spc_feature_integration import SPCFeatureIntegrator
except ImportError:
    logger.error("Failed to import required modules. Make sure you're running from the project root directory.")
    sys.exit(1)

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
    
    # Verify settings
    physical_devices = tf.config.list_physical_devices()
    logger.info(f"Available TensorFlow devices: {physical_devices}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Weather Prediction System")
    
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing weather data')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory to store trained models')
    parser.add_argument('--visualizations-dir', type=str, default='visualizations',
                        help='Directory to store visualizations')
    parser.add_argument('--start-date', type=str,
                        help='Start date in YYYYMMDD format')
    parser.add_argument('--end-date', type=str,
                        help='End date in YYYYMMDD format')
    parser.add_argument('--feature-type', type=str, default='all',
                        choices=['all', 'tornado', 'wind', 'hail'],
                        help='Type of features to include')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Force CPU-only mode even if GPU is available')
    parser.add_argument('--advanced-features', action='store_true',
                        help='Use advanced feature engineering')
    parser.add_argument('--ensemble', action='store_true',
                        help='Create and train ensemble model')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate existing models without training')
    parser.add_argument('--forecast', type=str,
                        help='Generate forecast for specific date (YYYYMMDD)')
    # New SPC integration arguments
    parser.add_argument('--use-spc', action='store_true',
                        help='Integrate with SPC data for forecasts and verification')
    parser.add_argument('--outlook-day', type=int, default=1, choices=range(1, 9),
                        help='SPC outlook day (1-8) to use for verification')
    parser.add_argument('--location', type=str,
                        help='Generate forecast for specific location (lat,lon)')
    parser.add_argument('--spc-cache-dir', type=str, default='data/spc_cache',
                        help='Directory to cache SPC data')
    parser.add_argument('--fetch-spc-only', action='store_true',
                        help='Only fetch SPC data without running predictions')
    parser.add_argument('--spc-features', action='store_true',
                        help='Use SPC data as additional features for training')
    parser.add_argument('--historical-spc-dir', type=str, default='data/historical_spc',
                        help='Directory to store historical SPC data')
    
    return parser.parse_args()

def run_weather_system(args):
    """
    Run the weather prediction system with the specified arguments.
    
    Args:
        args: Command-line arguments
    """
    logger.info("Starting Weather Prediction System")
    
    # Apply CPU optimizations if requested
    if args.cpu_only:
        apply_cpu_optimizations()
    
    # Create the weather prediction system
    system = WeatherPredictionSystem(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        visualizations_dir=args.visualizations_dir
    )
    
    # Initialize advanced feature extractor if requested
    feature_extractor = None
    if args.advanced_features:
        logger.info("Initializing advanced feature extractor")
        feature_extractor = AdvancedFeatureExtractor(data_dir=args.data_dir)
    
    # Initialize SPC integration if requested
    spc_integration = None
    spc_feature_integrator = None
    
    if args.use_spc or args.fetch_spc_only or args.spc_features:
        logger.info("Initializing SPC integration")
        spc_integration = SPCIntegration(
            model_output_dir=args.visualizations_dir,
            spc_cache_dir=args.spc_cache_dir
        )
        
        if args.spc_features:
            logger.info("Initializing SPC feature integration")
            spc_feature_integrator = SPCFeatureIntegrator(
                spc_cache_dir=args.spc_cache_dir,
                historical_data_dir=args.historical_spc_dir
            )
    
    # If only fetching SPC data, do that and exit
    if args.fetch_spc_only and spc_integration:
        fetch_spc_data_only(spc_integration, args.outlook_day)
        return
    
    # Determine date range
    start_date = args.start_date
    end_date = args.end_date
    
    if not start_date or not end_date:
        available_dates = system.data_loader.get_available_dates()
        if available_dates:
            if not start_date:
                # If using SPC features and no start date, use all available dates
                if args.spc_features:
                    logger.info("Using all available data for SPC feature training")
                    start_date = available_dates[0]  # Use earliest date
                else:
                    # Otherwise use the most recent 90 days as default
                    all_dates = sorted(available_dates)
                    if len(all_dates) > 90:
                        start_date = all_dates[-90]
                    else:
                        start_date = all_dates[0]
            
            if not end_date:
                end_date = available_dates[-1]
    
    logger.info(f"Using date range: {start_date} to {end_date}")
    
    # Load and process data
    logger.info("Loading and processing data")
    dataset = system.load_data(start_date, end_date)
    
    # Add SPC features to dataset if requested
    if args.spc_features and spc_feature_integrator:
        logger.info("Adding SPC features to dataset")
        try:
            # Fetch historical SPC data for the date range
            logger.info(f"Fetching historical SPC data for range: {start_date} to {end_date}")
            historical_data = spc_feature_integrator.fetch_historical_spc_data(start_date, end_date)
            # Enhance dataset with SPC features
            dataset = spc_feature_integrator.add_spc_features_to_dataset(dataset, historical_data)
            logger.info("SPC features added to dataset")
        except Exception as e:
            logger.error(f"Error adding SPC features: {e}")
            logger.warning("Continuing with training without SPC features")
    
    # Handle location-specific forecast if requested
    if args.location and spc_integration:
        try:
            lat, lon = map(float, args.location.split(','))
            generate_location_forecast(system, feature_extractor, spc_integration, lat, lon, args.forecast or datetime.now().strftime('%Y%m%d'))
            return
        except ValueError:
            logger.error(f"Invalid location format. Use 'lat,lon' (e.g., '39.7456,-97.0892')")
            return
    
    # Only forecast if requested
    if args.forecast:
        if args.use_spc and spc_integration:
            # Generate forecast with SPC integration
            generate_forecast_with_spc(system, feature_extractor, spc_integration, args.forecast, args.outlook_day)
        else:
            # Generate regular forecast
            generate_forecast(system, feature_extractor, args.forecast)
        return
    
    # Create TensorFlow datasets
    logger.info("Creating TensorFlow datasets")
    if args.advanced_features and feature_extractor:
        # Use advanced feature engineering
        train_dataset, test_dataset, feature_names = create_advanced_datasets(
            system, feature_extractor, dataset, args.feature_type, args.batch_size
        )
    else:
        # Use basic feature engineering
        train_dataset, test_dataset, feature_names = system.create_tf_feature_dataset(
            dataset, args.feature_type
        )
    
    # Evaluate only if requested
    if args.eval_only:
        if args.use_spc and spc_integration:
            # Evaluate with SPC data for verification
            evaluate_models_with_spc(system, test_dataset, spc_integration, args.outlook_day)
        else:
            # Regular evaluation
            evaluate_models(system, test_dataset)
        return
    
    # Build models
    logger.info("Building TensorFlow models")
    system.build_tensorflow_models()
    
    # Train models
    logger.info("Training models")
    results = {}
    
    for model_name in system.models:
        # Skip models if not relevant to the feature type
        if args.feature_type != 'all' and args.feature_type not in model_name:
            continue
            
        logger.info(f"Training {model_name} model")
        
        # Train model
        history = system.train_model(model_name, train_dataset, test_dataset, args.epochs)
        
        # Evaluate model
        metrics = system.evaluate_model(model_name, test_dataset)
        
        # Store results
        results[model_name] = {
            'history': history.history if history else None,
            'metrics': metrics
        }
    
    # Create ensemble model if requested
    if args.ensemble and len(results) > 1:
        logger.info("Creating and training ensemble model")
        
        # Determine input shape from the dataset
        for x, y in train_dataset.take(1):
            input_shape = x.shape[1:]
            break
        
        # Create ensemble model
        model_names = list(results.keys())
        ensemble_model = system.create_ensemble_model(model_names, input_shape)
        
        # Train ensemble model
        if ensemble_model:
            history = system.train_model('ensemble', train_dataset, test_dataset, args.epochs)
            metrics = system.evaluate_model('ensemble', test_dataset)
            
            results['ensemble'] = {
                'history': history.history if history else None,
                'metrics': metrics
            }
    
    # Create visualizations for results
    logger.info("Creating visualizations")
    system.visualizer.plot_training_history(results)
    system.visualizer.plot_model_comparison(results)
    
    logger.info("Weather Prediction System completed successfully")
    
    return results

def create_advanced_datasets(system, feature_extractor, dataset, feature_type='all', batch_size=32):
    """
    Create TensorFlow datasets with advanced feature engineering.
    
    Args:
        system: WeatherPredictionSystem instance
        feature_extractor: AdvancedFeatureExtractor instance
        dataset: Processed dataset
        feature_type: Type of features to include
        batch_size: Batch size for the datasets
        
    Returns:
        tuple: (train_dataset, test_dataset, feature_names)
    """
    logger.info("Creating datasets with advanced feature engineering")
    
    try:
        # Extract advanced features
        all_features = []
        all_labels = []
        spc_features = {}  # Track SPC features if available
        feature_names = None
        
        for date, date_data in tqdm(dataset.items(), desc="Extracting advanced features"):
            for tod, tod_data in date_data.items():
                for file_name, file_data in tod_data.items():
                    # Skip if not relevant to the feature type
                    if feature_type != 'all' and feature_type not in file_name:
                        continue
                    
                    # Extract advanced features
                    features = feature_extractor.extract_advanced_features(file_data, file_name)
                    
                    # Add SPC features if they exist in the dataset
                    spc_feature_names = [k for k in file_data.keys() if k.startswith(('spc_', 'location_', 'season_', 'month', 'day', 'region'))]
                    
                    if spc_feature_names:
                        for feature_name in spc_feature_names:
                            features[feature_name] = file_data[feature_name]
                            if feature_name not in spc_features:
                                spc_features[feature_name] = True
                        logger.debug(f"Added {len(spc_feature_names)} SPC features")
                    
                    if features:
                        # Determine label based on file name
                        label = 0
                        if 'tornado' in file_name:
                            label = 1 if np.any([v > 0.5 for k, v in features.items() if 'max' in k]) else 0
                        elif 'wind' in file_name:
                            label = 1 if np.any([v > 40 for k, v in features.items() if 'max' in k]) else 0
                        elif 'hail' in file_name:
                            label = 1 if np.any([v > 0.3 for k, v in features.items() if 'max' in k]) else 0
                        
                        # Convert dictionary to list in consistent order
                        if feature_names is None:
                            feature_names = list(features.keys())
                        
                        # Extract features in the right order
                        feature_values = [features.get(name, 0.0) for name in feature_names]
                        
                        all_features.append(feature_values)
                        all_labels.append(label)
        
        if not all_features:
            logger.error("No features extracted")
            # Fall back to basic feature engineering
            return system.create_tf_feature_dataset(dataset, feature_type)
        
        # Convert to numpy arrays
        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels, dtype=np.float32)
        
        # Handle NaN values
        X = np.nan_to_num(X)
        
        # Normalize features
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        stds[stds == 0] = 1.0
        X_norm = (X - means) / stds
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_norm, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        
        # Batch and prefetch for performance
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        logger.info(f"Created datasets with {len(feature_names)} advanced features")
        logger.info(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
        
        # Log SPC features if they were used
        if spc_features:
            logger.info(f"Using {len(spc_features)} SPC-related features: {', '.join(sorted(spc_features.keys()))}")
        
        return train_dataset, test_dataset, feature_names
        
    except Exception as e:
        logger.error(f"Error creating advanced datasets: {e}")
        # Fall back to basic feature engineering
        return system.create_tf_feature_dataset(dataset, feature_type)

def evaluate_models(system, test_dataset, return_metrics=False):
    """
    Evaluate all trained models on the test dataset.
    
    Args:
        system: Weather prediction system
        test_dataset: Test dataset
        return_metrics: Whether to return metrics (default: False)
    
    Returns:
        dict: Metrics for each model if return_metrics is True
    """
    logger.info("Evaluating trained models")
    metrics = {}
    
    for model_name in system.models:
        logger.info(f"Evaluating {model_name} model")
        
        # Skip if model doesn't exist
        model_path = os.path.join(system.models_dir, f"{model_name}.h5")
        if not os.path.exists(model_path):
            logger.warning(f"Model {model_name} not found at {model_path}, skipping")
            continue
        
        # Load model
        model = system.models[model_name]
        
        # Evaluate
        eval_metrics = system.evaluate_model(model_name, test_dataset)
        metrics[model_name] = eval_metrics
        
        logger.info(f"Model {model_name} metrics: {eval_metrics}")
    
    if return_metrics:
        return metrics

def generate_forecast(system, feature_extractor, forecast_date, return_results=False):
    """
    Generate forecast for a specific date.
    
    Args:
        system: Weather prediction system
        feature_extractor: Feature extractor
        forecast_date: Date to forecast
        return_results: Whether to return results (default: False)
    
    Returns:
        dict: Forecast results if return_results is True
    """
    logger.info(f"Generating forecast for {forecast_date}")
    
    # Load the forecast date data
    try:
        date_data = system.data_loader.load_date_data(forecast_date)
        if not date_data:
            logger.error(f"No data available for date {forecast_date}")
            return None
    except Exception as e:
        logger.error(f"Error loading date data: {e}")
        return None
    
    # Process the data similar to the training data
    processed_data = system.data_processor.process_raw_data({forecast_date: date_data})
    
    if not processed_data or forecast_date not in processed_data:
        logger.error(f"Failed to process data for date {forecast_date}")
        return None
    
    # Run predictions using all available models
    results = {}
    
    for model_name in system.models:
        logger.info(f"Running {model_name} model prediction")
        
        # Skip if model doesn't exist
        model_path = os.path.join(system.models_dir, f"{model_name}.h5")
        if not os.path.exists(model_path):
            logger.warning(f"Model {model_name} not found at {model_path}, skipping")
            continue
        
        # Load model
        model = system.models[model_name]
        
        # Prepare inputs for this specific model
        if feature_extractor:
            # Use advanced feature extractor
            features = feature_extractor.extract_features(processed_data[forecast_date])
            model_inputs = feature_extractor.prepare_model_inputs(features, model_name)
        else:
            # Use basic feature extraction
            model_inputs = system.prepare_model_inputs(processed_data[forecast_date], model_name)
        
        # Run prediction
        try:
            predictions = model.predict(model_inputs)
            
            # Save the predictions
            results[model_name] = {
                'predictions': {
                    'lat': processed_data[forecast_date].get('lat', []),
                    'lon': processed_data[forecast_date].get('lon', []),
                    'probabilities': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'forecast_date': forecast_date
                }
            }
            
            # Add SPC features to the prediction if available
            try:
                from src.spc_feature_integration import SPCFeatureIntegrator
                spc_feature_integrator = SPCFeatureIntegrator()
                enhanced_predictions = spc_feature_integrator.extract_spc_features_for_prediction(
                    results[model_name]['predictions']
                )
                results[model_name]['predictions'] = enhanced_predictions
                logger.info(f"Added SPC features to prediction for {model_name}")
            except Exception as e:
                logger.warning(f"Could not add SPC features to prediction: {e}")
            
            # Visualize the prediction
            system.visualizer.plot_prediction(
                results[model_name]['predictions'],
                os.path.join(system.visualizations_dir, f"{model_name}_{forecast_date}.png"),
                title=f"{model_name} Forecast for {forecast_date}"
            )
            
            logger.info(f"Completed prediction for {model_name}")
        
        except Exception as e:
            logger.error(f"Error running prediction for {model_name}: {e}")
    
    if not results:
        logger.warning("No forecast results generated")
    
    if return_results:
        return results

def generate_forecast_with_spc(system, feature_extractor, spc_integration, forecast_date, outlook_day):
    """
    Generate forecast with SPC integration for comparison
    
    Args:
        system: Weather prediction system
        feature_extractor: Feature extractor
        spc_integration: SPC integration
        forecast_date: Date to forecast
        outlook_day: SPC outlook day to use for verification
    """
    logger.info(f"Generating forecast for {forecast_date} with SPC verification")
    
    # Generate the regular forecast first
    prediction_results = generate_forecast(system, feature_extractor, forecast_date, return_results=True)
    
    if not prediction_results:
        logger.error("Forecast generation failed, cannot create SPC comparison")
        return
    
    # Fetch SPC outlook data
    logger.info(f"Fetching SPC outlook data for day {outlook_day}")
    outlook_data = spc_integration.get_current_spc_outlook(day=outlook_day)
    
    if not outlook_data:
        logger.warning(f"Could not fetch SPC outlook for day {outlook_day}")
    else:
        logger.info(f"Successfully fetched SPC outlook")
    
    # Create combined visualization with model prediction and SPC outlook
    output_file = os.path.join(
        system.visualizations_dir,
        f"forecast_{forecast_date}_with_spc_day{outlook_day}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    
    # Create a standardized format for the prediction data
    prediction_data = {
        'lat': [],
        'lon': [],
        'probability': []
    }
    
    # Reshape prediction results into the format needed for plotting
    for model_name, model_preds in prediction_results.items():
        if isinstance(model_preds, dict) and 'predictions' in model_preds:
            predictions = model_preds['predictions']
            if isinstance(predictions, dict) and 'lat' in predictions and 'lon' in predictions:
                prediction_data['lat'].extend(predictions['lat'])
                prediction_data['lon'].extend(predictions['lon'])
                
                # Use the first probability dimension if there are multiple
                if 'probabilities' in predictions and len(predictions['probabilities']) > 0:
                    if isinstance(predictions['probabilities'][0], (list, np.ndarray)):
                        # If we have a multi-class output, use the first class
                        probs = [p[0] if isinstance(p, (list, np.ndarray)) else p for p in predictions['probabilities']]
                    else:
                        probs = predictions['probabilities']
                    prediction_data['probability'].extend(probs)
    
    # Create the verification plot
    if prediction_data['lat'] and prediction_data['lon'] and prediction_data['probability']:
        output_path = spc_integration.create_risk_verification_plot(
            prediction_data, 
            outlook_data, 
            output_file=output_file
        )
        logger.info(f"Saved verification plot to {output_path}")
    else:
        logger.warning("Could not create verification plot due to missing prediction data")

def evaluate_models_with_spc(system, test_dataset, spc_integration, outlook_day):
    """
    Evaluate models with SPC data for verification
    
    Args:
        system: Weather prediction system
        test_dataset: Test dataset
        spc_integration: SPC integration
        outlook_day: SPC outlook day to use for verification
    """
    logger.info(f"Evaluating models with SPC verification for day {outlook_day}")
    
    # First, perform regular model evaluation
    model_metrics = evaluate_models(system, test_dataset, return_metrics=True)
    
    # Fetch SPC outlook
    outlook_data = spc_integration.get_current_spc_outlook(day=outlook_day)
    
    if not outlook_data:
        logger.warning(f"Could not fetch SPC outlook for day {outlook_day}, continuing with regular evaluation")
        return
    
    # Fetch recent severe reports for verification
    recent_reports = spc_integration.get_recent_severe_reports()
    
    # Save the evaluation results with SPC data
    output_dir = os.path.join(system.visualizations_dir, 'model_evaluation')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create combined report
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_metrics': model_metrics,
        'spc_outlook': {
            'day': outlook_data.get('day'),
            'valid_time': outlook_data.get('valid_time'),
            'discussion_url': outlook_data.get('discussion_url')
        }
    }
    
    # Add summary of recent reports
    if not recent_reports.empty:
        report['recent_reports'] = {
            'total_count': len(recent_reports),
            'tornado_count': len(recent_reports[recent_reports['type'] == 'TORNADO']) if 'type' in recent_reports.columns else 'unknown',
            'wind_count': len(recent_reports[recent_reports['type'] == 'WIND']) if 'type' in recent_reports.columns else 'unknown',
            'hail_count': len(recent_reports[recent_reports['type'] == 'HAIL']) if 'type' in recent_reports.columns else 'unknown'
        }
    
    # Save the report
    output_file = os.path.join(output_dir, f"evaluation_with_spc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        # Convert numpy values to Python native types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Saved evaluation report with SPC data to {output_file}")
    
    return report

def fetch_spc_data_only(spc_integration, outlook_day):
    """
    Fetch SPC data without running predictions
    
    Args:
        spc_integration: SPC integration
        outlook_day: SPC outlook day to fetch
    """
    logger.info(f"Fetching SPC data for day {outlook_day}")
    
    # Get today's date
    today = datetime.now().strftime('%Y%m%d')
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    
    # Fetch outlook for today
    logger.info(f"Fetching SPC outlook for today ({today})")
    outlook_data = spc_integration.get_current_spc_outlook(day=outlook_day)
    
    if not outlook_data:
        logger.error(f"Could not fetch SPC outlook for day {outlook_day}")
    else:
        logger.info(f"Successfully fetched SPC outlook for day {outlook_day}")
    
    # Also fetch recent severe reports - both yesterday and today
    logger.info(f"Fetching severe weather reports for yesterday ({yesterday}) and today ({today})")
    
    yesterday_reports = spc_integration.spc_fetcher.get_severe_weather_reports(date=yesterday)
    today_reports = spc_integration.spc_fetcher.get_severe_weather_reports(date=today)
    
    # Combine reports
    if yesterday_reports.empty and today_reports.empty:
        logger.warning("No recent severe weather reports found")
    else:
        total_reports = len(yesterday_reports) + len(today_reports)
        logger.info(f"Fetched {total_reports} recent severe weather reports")
    
    # Fetch mesoscale discussions for today
    logger.info(f"Fetching mesoscale discussions for today ({today})")
    discussions = spc_integration.spc_fetcher.get_mesoscale_discussions(start_date=today)
    
    if not discussions:
        logger.info("No mesoscale discussions found for today")
    else:
        logger.info(f"Fetched {len(discussions)} mesoscale discussions for today")
    
    # Fetch watches for today
    logger.info(f"Fetching watches for today ({today})")
    watches = spc_integration.spc_fetcher.get_watches(date=today)
    
    if not watches:
        logger.info("No watches found for today")
    else:
        logger.info(f"Fetched {len(watches)} watches for today")
    
    # Summary output
    logger.info("=== SPC Data Summary ===")
    logger.info(f"Outlook Day {outlook_day}: Valid time: {outlook_data.get('valid_time', 'Unknown')}")
    
    if outlook_data and outlook_data.get('discussion_text'):
        logger.info("Discussion excerpt: " + outlook_data['discussion_text'][:200] + "...")
    
    if not yesterday_reports.empty and 'type' in yesterday_reports.columns:
        logger.info(f"Yesterday's reports: {len(yesterday_reports)} total")
        report_types = yesterday_reports['type'].value_counts().to_dict() if 'type' in yesterday_reports.columns else {}
        for rtype, count in report_types.items():
            logger.info(f"  - {rtype}: {count}")
    
    if not today_reports.empty and 'type' in today_reports.columns:
        logger.info(f"Today's reports: {len(today_reports)} total")
        report_types = today_reports['type'].value_counts().to_dict() if 'type' in today_reports.columns else {}
        for rtype, count in report_types.items():
            logger.info(f"  - {rtype}: {count}")
    
    logger.info(f"Mesoscale discussions: {len(discussions)}")
    logger.info(f"Watches: {len(watches)}")
    
    # Save today's data for easy access
    try:
        cache_dir = os.path.join(spc_integration.spc_cache_dir, 'daily')
        os.makedirs(cache_dir, exist_ok=True)
        
        summary_file = os.path.join(cache_dir, f"{today}_summary.json")
        summary = {
            'date': today,
            'outlook': outlook_data,
            'reports_count': len(today_reports),
            'discussions_count': len(discussions),
            'watches_count': len(watches),
            'fetch_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved today's SPC data summary to {summary_file}")
    except Exception as e:
        logger.warning(f"Error saving SPC data summary: {e}")

def generate_location_forecast(system, feature_extractor, spc_integration, lat, lon, forecast_date):
    """
    Generate forecast for a specific location
    
    Args:
        system: Weather prediction system
        feature_extractor: Feature extractor
        spc_integration: SPC integration
        lat: Latitude of the location
        lon: Longitude of the location
        forecast_date: Date to forecast
    """
    logger.info(f"Generating forecast for location ({lat}, {lon}) on {forecast_date}")
    
    # Generate the regular forecast first
    prediction_results = generate_forecast(system, feature_extractor, forecast_date, return_results=True)
    
    if not prediction_results:
        logger.error("Forecast generation failed, cannot create location forecast")
        return
    
    # Fetch SPC outlook data
    logger.info("Fetching SPC outlook data")
    outlook_data = spc_integration.get_current_spc_outlook(day=1)  # Default to day 1
    
    # Create a standardized format for the prediction data
    prediction_data = {
        'lat': [],
        'lon': [],
        'probability': []
    }
    
    # Reshape prediction results into the format needed
    for model_name, model_preds in prediction_results.items():
        if isinstance(model_preds, dict) and 'predictions' in model_preds:
            predictions = model_preds['predictions']
            if isinstance(predictions, dict) and 'lat' in predictions and 'lon' in predictions:
                prediction_data['lat'].extend(predictions['lat'])
                prediction_data['lon'].extend(predictions['lon'])
                
                # Use the first probability dimension if there are multiple
                if 'probabilities' in predictions and len(predictions['probabilities']) > 0:
                    if isinstance(predictions['probabilities'][0], (list, np.ndarray)):
                        # If we have a multi-class output, use the first class
                        probs = [p[0] if isinstance(p, (list, np.ndarray)) else p for p in predictions['probabilities']]
                    else:
                        probs = predictions['probabilities']
                    prediction_data['probability'].extend(probs)
    
    # Generate the location risk report
    if prediction_data['lat'] and prediction_data['lon'] and prediction_data['probability']:
        report = spc_integration.generate_location_risk_report(
            lat=lat,
            lon=lon,
            prediction_data=prediction_data,
            outlook_data=outlook_data
        )
        
        # Save the report
        report_path = spc_integration.save_location_report(report)
        
        # Print a summary
        location_info = report['location']['info']
        model_risk = report['model_prediction']
        
        print("\n=== Location Forecast Summary ===")
        print(f"Location: ({lat}, {lon}) - {location_info.get('county', 'Unknown')} County, {location_info.get('state', 'Unknown')}")
        print(f"Forecast Date: {forecast_date}")
        print(f"Model Risk: {model_risk.get('category', 'Unknown')} ({model_risk.get('probability', 'Unknown'):.1%})")
        
        if report['recent_reports']['count'] > 0:
            print(f"Recent Severe Weather: {report['recent_reports']['count']} reports within 100km")
        
        if outlook_data:
            print(f"SPC Outlook: Day {outlook_data.get('day', 1)}, Valid {outlook_data.get('valid_time', 'Unknown')}")
            if outlook_data.get('discussion_url'):
                print(f"Discussion: {outlook_data['discussion_url']}")
        
        logger.info(f"Saved location forecast to {report_path}")
    else:
        logger.warning("Could not create location forecast due to missing prediction data")

def main():
    """Main entry point."""
    args = parse_arguments()
    run_weather_system(args)

if __name__ == "__main__":
    main() 