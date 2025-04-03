import os
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeatherModelTrainer:
    """
    Class for training and evaluating weather prediction models
    """
    
    def __init__(self, models_dir='/home/jackson/weather_prediction_bot/models'):
        """
        Initialize the model trainer
        
        Args:
            models_dir (str): Directory to save trained models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Dictionary to store trained models
        self.models = {}
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def prepare_data(self, data_dict, target_col='target_max_hail', task='regression'):
        """
        Prepare data for training
        
        Args:
            data_dict (dict): Dictionary with train, val, test DataFrames
            target_col (str): Target column name
            task (str): 'regression' or 'classification'
            
        Returns:
            dict: Dictionary with X and y for train, val, test
        """
        # Identify feature columns
        date_cols = ['date', 'year', 'month', 'day', 'day_of_year', 'season']
        
        if task == 'classification' and not target_col.endswith('binary'):
            target_col = 'target_hail_binary'  # Default binary target
        
        target_cols = [col for col in data_dict['train'].columns if col.startswith('target_')]
        
        X_cols = [col for col in data_dict['train'].columns 
                 if col not in target_cols and col not in date_cols]
        
        result = {}
        
        for split, df in data_dict.items():
            X = df[X_cols].copy()
            y = df[target_col].copy()
            
            result[f'X_{split}'] = X
            result[f'y_{split}'] = y
            
            # Add date info for later use in visualization
            result[f'dates_{split}'] = df['date'].copy()
        
        return result
    
    def train_random_forest(self, X_train, y_train, X_val, y_val, task='regression', **kwargs):
        """
        Train a Random Forest model
        
        Args:
            X_train (DataFrame): Training features
            y_train (Series): Training targets
            X_val (DataFrame): Validation features
            y_val (Series): Validation targets
            task (str): 'regression' or 'classification'
            **kwargs: Additional parameters for the model
            
        Returns:
            model: Trained model
        """
        logger.info("Training Random Forest model...")
        
        # Set default parameters
        params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Update with any provided kwargs
        params.update(kwargs)
        
        # Create appropriate model based on task
        if task == 'regression':
            model = RandomForestRegressor(**params)
        else:
            model = RandomForestClassifier(**params)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        self._evaluate_model(model, X_val, y_val, task)
        
        return model
    
    def train_gradient_boosting(self, X_train, y_train, X_val, y_val, task='regression', **kwargs):
        """
        Train a Gradient Boosting model
        
        Args:
            X_train (DataFrame): Training features
            y_train (Series): Training targets
            X_val (DataFrame): Validation features
            y_val (Series): Validation targets
            task (str): 'regression' or 'classification'
            **kwargs: Additional parameters for the model
            
        Returns:
            model: Trained model
        """
        logger.info("Training Gradient Boosting model...")
        
        # Set default parameters
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
        # Update with any provided kwargs
        params.update(kwargs)
        
        # Create appropriate model based on task
        if task == 'regression':
            model = GradientBoostingRegressor(**params)
        else:
            model = GradientBoostingClassifier(**params)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        self._evaluate_model(model, X_val, y_val, task)
        
        return model
    
    def train_xgboost(self, X_train, y_train, X_val, y_val, task='regression', **kwargs):
        """
        Train an XGBoost model
        
        Args:
            X_train (DataFrame): Training features
            y_train (Series): Training targets
            X_val (DataFrame): Validation features
            y_val (Series): Validation targets
            task (str): 'regression' or 'classification'
            **kwargs: Additional parameters for the model
            
        Returns:
            model: Trained model
        """
        logger.info("Training XGBoost model...")
        
        # Set default parameters
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'random_state': 42,
            'tree_method': 'hist'  # For faster training
        }
        
        # Update with any provided kwargs
        params.update(kwargs)
        
        # Create appropriate model based on task
        if task == 'regression':
            model = xgb.XGBRegressor(**params)
        else:
            model = xgb.XGBClassifier(**params)
        
        # Train the model
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Evaluate on validation set
        self._evaluate_model(model, X_val, y_val, task)
        
        return model
    
    def train_neural_network(self, X_train, y_train, X_val, y_val, task='regression', **kwargs):
        """
        Train a neural network model using TensorFlow/Keras
        
        Args:
            X_train (DataFrame): Training features
            y_train (Series): Training targets
            X_val (DataFrame): Validation features
            y_val (Series): Validation targets
            task (str): 'regression' or 'classification'
            **kwargs: Additional parameters for the model
            
        Returns:
            model: Trained model
        """
        logger.info("Training Neural Network model...")
        
        # Convert to numpy arrays
        X_train_np = X_train.values
        y_train_np = y_train.values
        X_val_np = X_val.values
        y_val_np = y_val.values
        
        # Get input dimensions
        input_dim = X_train_np.shape[1]
        
        # Set up model architecture
        model = Sequential()
        
        # Input layer
        model.add(Dense(128, input_dim=input_dim, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # Hidden layers
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        
        # Output layer
        if task == 'regression':
            model.add(Dense(1, activation='linear'))
            loss = 'mean_squared_error'
            metrics = ['mae']
        else:
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss=loss,
            metrics=metrics
        )
        
        # Set up callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(self.models_dir, 'nn_model.h5'),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        history = model.fit(
            X_train_np, y_train_np,
            epochs=100,
            batch_size=32,
            validation_data=(X_val_np, y_val_np),
            callbacks=callbacks,
            verbose=0
        )
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        if task == 'regression':
            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'], label='Training MAE')
            plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.xlabel('Epochs')
            plt.ylabel('MAE')
            plt.legend()
        else:
            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_dir, 'nn_training_history.png'))
        
        # Evaluate on validation set
        if task == 'regression':
            y_pred = model.predict(X_val_np).flatten()
            mae = mean_absolute_error(y_val_np, y_pred)
            mse = mean_squared_error(y_val_np, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val_np, y_pred)
            
            logger.info(f"Neural Network Validation - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        else:
            y_pred_proba = model.predict(X_val_np).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            accuracy = accuracy_score(y_val_np, y_pred)
            precision = precision_score(y_val_np, y_pred)
            recall = recall_score(y_val_np, y_pred)
            f1 = f1_score(y_val_np, y_pred)
            
            logger.info(f"Neural Network Validation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return model
    
    def _evaluate_model(self, model, X, y, task='regression'):
        """
        Evaluate a model on validation or test data
        
        Args:
            model: Trained model
            X (DataFrame): Features
            y (Series): Targets
            task (str): 'regression' or 'classification'
        """
        if task == 'regression':
            y_pred = model.predict(X)
            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, y_pred)
            
            logger.info(f"Model Evaluation - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred
            }
        else:
            # For classification
            try:
                y_pred_proba = model.predict_proba(X)[:, 1]
            except:
                y_pred_proba = model.predict(X)
            
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            auc = roc_auc_score(y, y_pred_proba)
            
            logger.info(f"Model Evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                      f"Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
    
    def train_models(self, data_dict, target_col='target_max_hail', task='regression'):
        """
        Train multiple models and evaluate them
        
        Args:
            data_dict (dict): Dictionary with train, val, test DataFrames
            target_col (str): Target column name
            task (str): 'regression' or 'classification'
            
        Returns:
            dict: Dictionary containing trained models and their evaluations
        """
        # Prepare data
        prepared_data = self.prepare_data(data_dict, target_col, task)
        
        X_train = prepared_data['X_train']
        y_train = prepared_data['y_train']
        X_val = prepared_data['X_val']
        y_val = prepared_data['y_val']
        X_test = prepared_data['X_test']
        y_test = prepared_data['y_test']
        
        # Train models
        models = {}
        evaluations = {}
        
        # Random Forest
        rf_model = self.train_random_forest(X_train, y_train, X_val, y_val, task)
        models['random_forest'] = rf_model
        evaluations['random_forest'] = self._evaluate_model(rf_model, X_test, y_test, task)
        
        # Gradient Boosting
        gb_model = self.train_gradient_boosting(X_train, y_train, X_val, y_val, task)
        models['gradient_boosting'] = gb_model
        evaluations['gradient_boosting'] = self._evaluate_model(gb_model, X_test, y_test, task)
        
        # XGBoost
        xgb_model = self.train_xgboost(X_train, y_train, X_val, y_val, task)
        models['xgboost'] = xgb_model
        evaluations['xgboost'] = self._evaluate_model(xgb_model, X_test, y_test, task)
        
        # Neural Network
        nn_model = self.train_neural_network(X_train, y_train, X_val, y_val, task)
        models['neural_network'] = nn_model
        
        # Store models
        self.models = models
        
        # Determine best model
        if task == 'regression':
            best_model_name = min(evaluations, key=lambda m: evaluations[m]['mae'])
            best_metric = 'mae'
        else:
            best_model_name = max(evaluations, key=lambda m: evaluations[m]['f1'])
            best_metric = 'f1'
        
        logger.info(f"Best model: {best_model_name} based on {best_metric}")
        
        # Save best model
        best_model = models[best_model_name]
        model_path = os.path.join(self.models_dir, f'best_model_{task}.joblib')
        joblib.dump(best_model, model_path)
        logger.info(f"Saved best model to {model_path}")
        
        # Save model info
        model_info = {
            'best_model': best_model_name,
            'task': task,
            'target_column': target_col,
            'feature_columns': list(X_train.columns),
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(os.path.join(self.models_dir, 'model_info.txt'), 'w') as f:
            for key, value in model_info.items():
                if isinstance(value, list):
                    f.write(f"{key}: {len(value)} features\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        # Plot feature importance for the best model (if available)
        if hasattr(best_model, 'feature_importances_'):
            self._plot_feature_importance(best_model, X_train.columns)
        
        return {
            'models': models,
            'evaluations': evaluations,
            'best_model': best_model_name,
            'prepared_data': prepared_data
        }
    
    def _plot_feature_importance(self, model, feature_names, top_n=20):
        """
        Plot feature importances
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: Names of features
            top_n (int): Number of top features to plot
        """
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot top N features
        top_indices = indices[:top_n]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances')
        plt.barh(range(len(top_features)), top_importances, align='center')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_dir, 'feature_importance.png'))
        
        # Save to CSV for reference
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        importance_df.to_csv(os.path.join(self.models_dir, 'feature_importance.csv'), index=False)
    
    def save_models(self):
        """
        Save all trained models
        
        Returns:
            dict: Dictionary with paths to saved models
        """
        paths = {}
        
        for model_name, model in self.models.items():
            # Skip saving neural network as it's saved during training
            if model_name == 'neural_network':
                paths[model_name] = os.path.join(self.models_dir, 'nn_model.h5')
                continue
                
            file_path = os.path.join(self.models_dir, f'{model_name}.joblib')
            joblib.dump(model, file_path)
            paths[model_name] = file_path
            logger.info(f"Saved {model_name} model to {file_path}")
        
        return paths 