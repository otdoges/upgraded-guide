import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
import logging
from PIL import Image, ImageDraw, ImageFont
import matplotlib.dates as mdates

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeatherVisualizer:
    """
    Class for creating visualizations of weather predictions
    """
    
    def __init__(self, output_dir='/home/jackson/weather_prediction_bot/visualizations'):
        """
        Initialize the visualizer
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up color maps
        self.setup_colormaps()
        
        # Set plot style
        sns.set_style('darkgrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    def setup_colormaps(self):
        """Set up custom colormaps for visualizations"""
        # Probability colormap (white to dark red)
        self.prob_cmap = LinearSegmentedColormap.from_list(
            'probability', 
            ['#FFFFFF', '#FFFFCC', '#FFEDA0', '#FED976', '#FEB24C', 
             '#FD8D3C', '#FC4E2A', '#E31A1C', '#BD0026', '#800026']
        )
        
        # Temperature colormap (cold to hot)
        self.temp_cmap = LinearSegmentedColormap.from_list(
            'temperature',
            ['#313695', '#4575B4', '#74ADD1', '#ABD9E9', '#E0F3F8', 
             '#FFFFBF', '#FEE090', '#FDAE61', '#F46D43', '#D73027', '#A50026']
        )
        
        # Precipitation colormap (white to blue)
        self.precip_cmap = LinearSegmentedColormap.from_list(
            'precipitation',
            ['#FFFFFF', '#EDF8FB', '#CCECE6', '#99D8C9', '#66C2A4', 
             '#41AE76', '#238B45', '#006D2C', '#00441B']
        )
        
        # Wind colormap
        self.wind_cmap = LinearSegmentedColormap.from_list(
            'wind',
            ['#FFFFFF', '#E6F5D0', '#B8E186', '#7FBC41', '#4D9221', 
             '#276419', '#000000']
        )
    
    def plot_spatial_prediction(self, lats, lons, values, title, forecast_date, 
                               colormap='probability', save_path=None):
        """
        Create a spatial plot of weather predictions
        
        Args:
            lats (array): Latitude coordinates
            lons (array): Longitude coordinates
            values (array): Prediction values
            title (str): Plot title
            forecast_date (str): Date of the forecast
            colormap (str): Name of colormap to use
            save_path (str): Path to save the plot
            
        Returns:
            str: Path to the saved plot
        """
        # Set up colormap
        if colormap == 'probability':
            cmap = self.prob_cmap
            vmin, vmax = 0, 1
        elif colormap == 'temperature':
            cmap = self.temp_cmap
            vmin, vmax = None, None
        elif colormap == 'precipitation':
            cmap = self.precip_cmap
            vmin, vmax = 0, None
        elif colormap == 'wind':
            cmap = self.wind_cmap
            vmin, vmax = 0, None
        else:
            cmap = 'viridis'
            vmin, vmax = None, None
        
        # Create figure with cartopy projection
        fig = plt.figure(figsize=(15, 10))
        ax = plt.axes(projection=ccrs.LambertConformal(
            central_longitude=-95, central_latitude=35))
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        
        # Set extent for continental US
        ax.set_extent([-125, -66, 24, 50], ccrs.PlateCarree())
        
        # Plot data
        mesh = ax.pcolormesh(
            lons, lats, values, 
            transform=ccrs.PlateCarree(),
            cmap=cmap, vmin=vmin, vmax=vmax
        )
        
        # Add colorbar
        cbar = plt.colorbar(mesh, ax=ax, pad=0.01, shrink=0.8)
        
        # Add title and timestamp
        if isinstance(forecast_date, str):
            if len(forecast_date) == 8:  # YYYYMMDD format
                date_obj = datetime.strptime(forecast_date, '%Y%m%d')
                date_str = date_obj.strftime('%Y-%m-%d')
            else:
                date_str = forecast_date
        else:
            date_str = forecast_date.strftime('%Y-%m-%d')
            
        plt.title(f"{title}\nForecast Date: {date_str}", fontsize=16)
        
        # Add timestamp
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        plt.annotate(f"Generated: {now}", (0.01, 0.01), xycoords='figure fraction', fontsize=8)
        
        # Save or show plot
        if save_path is None:
            date_part = date_str.replace('-', '')
            filename = f"{date_part}_{title.replace(' ', '_').lower()}.png"
            save_path = os.path.join(self.output_dir, filename)
        
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved spatial prediction plot to {save_path}")
        
        return save_path
    
    def plot_time_series(self, dates, values, model_name, target_name, title=None, save_path=None):
        """
        Create a time series plot of predictions
        
        Args:
            dates (list): List of dates
            values (list): List of prediction values
            model_name (str): Name of the model
            target_name (str): Name of the predicted target
            title (str): Plot title
            save_path (str): Path to save the plot
            
        Returns:
            str: Path to the saved plot
        """
        # Convert string dates to datetime if needed
        if isinstance(dates[0], str):
            if len(dates[0]) == 8:  # YYYYMMDD format
                date_objects = [datetime.strptime(d, '%Y%m%d') for d in dates]
            else:
                date_objects = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
        else:
            date_objects = dates
            
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot time series
        ax.plot(date_objects, values, marker='o', linestyle='-', linewidth=2, label=model_name)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Weekly ticks
        plt.xticks(rotation=45)
        
        # Set plot title and labels
        if title:
            plt.title(title, fontsize=16)
        else:
            plt.title(f"{target_name} Predictions over Time", fontsize=16)
            
        plt.xlabel("Date", fontsize=12)
        plt.ylabel(target_name, fontsize=12)
        plt.legend()
        
        # Add timestamp
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        plt.annotate(f"Generated: {now}", (0.01, 0.01), xycoords='figure fraction', fontsize=8)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show plot
        if save_path is None:
            filename = f"time_series_{target_name.replace(' ', '_').lower()}.png"
            save_path = os.path.join(self.output_dir, filename)
        
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved time series plot to {save_path}")
        
        return save_path
    
    def plot_model_comparison(self, dates, predictions_dict, actual=None, 
                             target_name='Hail Probability', save_path=None):
        """
        Create a time series plot comparing different models
        
        Args:
            dates (list): List of dates
            predictions_dict (dict): Dictionary with model names as keys and predictions as values
            actual (list): List of actual values (optional)
            target_name (str): Name of the predicted target
            save_path (str): Path to save the plot
            
        Returns:
            str: Path to the saved plot
        """
        # Convert string dates to datetime if needed
        if isinstance(dates[0], str):
            if len(dates[0]) == 8:  # YYYYMMDD format
                date_objects = [datetime.strptime(d, '%Y%m%d') for d in dates]
            else:
                date_objects = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
        else:
            date_objects = dates
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot predictions for each model
        for model_name, values in predictions_dict.items():
            ax.plot(date_objects, values, marker='o', linestyle='-', linewidth=2, 
                   label=f"{model_name} Prediction")
        
        # Plot actual values if available
        if actual is not None:
            ax.plot(date_objects, actual, marker='s', linestyle='-', linewidth=3, 
                   color='black', label='Actual')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Weekly ticks
        plt.xticks(rotation=45)
        
        # Set plot title and labels
        plt.title(f"Model Comparison: {target_name} Predictions", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel(target_name, fontsize=12)
        plt.legend()
        
        # Add timestamp
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        plt.annotate(f"Generated: {now}", (0.01, 0.01), xycoords='figure fraction', fontsize=8)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show plot
        if save_path is None:
            filename = f"model_comparison_{target_name.replace(' ', '_').lower()}.png"
            save_path = os.path.join(self.output_dir, filename)
        
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved model comparison plot to {save_path}")
        
        return save_path
    
    def plot_evaluation_metrics(self, metrics_dict, chart_type='bar', save_path=None):
        """
        Create a plot of model evaluation metrics
        
        Args:
            metrics_dict (dict): Dictionary with model names as keys and metric dictionaries as values
            chart_type (str): Type of chart ('bar' or 'radar')
            save_path (str): Path to save the plot
            
        Returns:
            str: Path to the saved plot
        """
        # Extract metrics for each model
        models = list(metrics_dict.keys())
        
        # Get metric names from the first model
        first_model = next(iter(metrics_dict.values()))
        metrics = [k for k in first_model.keys() if k not in ['predictions', 'probabilities']]
        
        # Create dataframe for plotting
        metrics_data = []
        for model in models:
            model_metrics = metrics_dict[model]
            row = {'Model': model}
            for metric in metrics:
                if metric in model_metrics:
                    row[metric] = model_metrics[metric]
            metrics_data.append(row)
        
        df = pd.DataFrame(metrics_data)
        
        # Create figure
        if chart_type == 'bar':
            # Melt dataframe for seaborn
            df_melted = pd.melt(df, id_vars=['Model'], value_vars=metrics, 
                               var_name='Metric', value_name='Value')
            
            plt.figure(figsize=(14, 8))
            sns.barplot(x='Model', y='Value', hue='Metric', data=df_melted)
            plt.title('Model Evaluation Metrics Comparison', fontsize=16)
            plt.ylabel('Score', fontsize=12)
            plt.xticks(rotation=45)
            plt.legend(title='Metric')
            
        else:  # radar chart
            # Needs matplotlib's radar chart
            # Number of metrics (variables)
            N = len(metrics)
            
            # Create angles for radar chart
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Create figure
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111, polar=True)
            
            # Draw one axis per variable and add labels
            plt.xticks(angles[:-1], metrics, fontsize=12)
            
            # Draw ylabels
            ax.set_rlabel_position(0)
            plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], fontsize=10)
            plt.ylim(0, 1)
            
            # Plot each model
            for model in models:
                values = [metrics_dict[model].get(metric, 0) for metric in metrics]
                values += values[:1]  # Close the loop
                
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
                ax.fill(angles, values, alpha=0.1)
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title('Model Evaluation Metrics Comparison', fontsize=16)
        
        # Add timestamp
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        plt.annotate(f"Generated: {now}", (0.01, 0.01), xycoords='figure fraction', fontsize=8)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show plot
        if save_path is None:
            filename = f"model_metrics_comparison_{chart_type}.png"
            save_path = os.path.join(self.output_dir, filename)
        
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved metrics comparison plot to {save_path}")
        
        return save_path
    
    def create_weather_forecast_image(self, date, predictions, region_data=None, 
                                     title="Weather Forecast", save_path=None):
        """
        Create a professional weather forecast image
        
        Args:
            date (str or datetime): Forecast date
            predictions (dict): Dictionary with prediction values
            region_data (dict): Dictionary with geographical data (optional)
            title (str): Image title
            save_path (str): Path to save the image
            
        Returns:
            str: Path to the saved image
        """
        # Convert date to string if it's a datetime
        if isinstance(date, datetime):
            date_str = date.strftime('%Y-%m-%d')
        else:
            if len(date) == 8:  # YYYYMMDD format
                date_obj = datetime.strptime(date, '%Y%m%d')
                date_str = date_obj.strftime('%Y-%m-%d')
            else:
                date_str = date
        
        # Create a background image
        width, height = 1200, 900
        image = Image.new('RGB', (width, height), (240, 240, 250))
        draw = ImageDraw.Draw(image)
        
        # Try to load a font
        try:
            title_font = ImageFont.truetype('DejaVuSans-Bold.ttf', 36)
            header_font = ImageFont.truetype('DejaVuSans-Bold.ttf', 24)
            text_font = ImageFont.truetype('DejaVuSans.ttf', 18)
        except IOError:
            # Fallback to default font
            title_font = ImageFont.load_default()
            header_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
        
        # Add title and date
        draw.text((width//2, 30), title, fill=(30, 30, 100), font=title_font, anchor="ms")
        draw.text((width//2, 70), f"Forecast for {date_str}", fill=(50, 50, 120), font=header_font, anchor="ms")
        
        # Add horizontal line
        draw.line([(50, 100), (width-50, 100)], fill=(180, 180, 220), width=2)
        
        # Add prediction values in a nice format
        y_pos = 120
        for pred_name, pred_value in predictions.items():
            # Format the prediction name
            formatted_name = pred_name.replace('_', ' ').title()
            
            # Format the prediction value
            if isinstance(pred_value, float):
                formatted_value = f"{pred_value:.2f}"
            else:
                formatted_value = str(pred_value)
            
            draw.text((100, y_pos), formatted_name, fill=(50, 50, 120), font=text_font)
            draw.text((width-100, y_pos), formatted_value, fill=(50, 50, 120), font=text_font, anchor="rs")
            
            y_pos += 40
        
        # Add a disclaimer
        disclaimer = "This forecast is based on machine learning models and may not reflect actual weather conditions."
        draw.text((width//2, height-30), disclaimer, fill=(100, 100, 150), font=text_font, anchor="ms")
        
        # Add timestamp
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        draw.text((width-50, height-10), f"Generated: {now}", fill=(100, 100, 150), font=ImageFont.load_default(), anchor="rs")
        
        # Save the image
        if save_path is None:
            date_part = date_str.replace('-', '')
            filename = f"forecast_{date_part}.png"
            save_path = os.path.join(self.output_dir, filename)
        
        image.save(save_path, quality=95)
        
        logger.info(f"Saved forecast image to {save_path}")
        
        return save_path
    
    def create_forecast_comparison(self, date, model_predictions, actual=None, 
                                 title="Model Forecast Comparison", save_path=None):
        """
        Create an image comparing forecasts from different models
        
        Args:
            date (str or datetime): Forecast date
            model_predictions (dict): Dictionary with model names as keys and prediction dicts as values
            actual (dict): Dictionary with actual values (optional)
            title (str): Image title
            save_path (str): Path to save the image
            
        Returns:
            str: Path to the saved image
        """
        # Convert date to string if it's a datetime
        if isinstance(date, datetime):
            date_str = date.strftime('%Y-%m-%d')
        else:
            if len(date) == 8:  # YYYYMMDD format
                date_obj = datetime.strptime(date, '%Y%m%d')
                date_str = date_obj.strftime('%Y-%m-%d')
            else:
                date_str = date
        
        # Create a background image
        width, height = 1200, 900
        image = Image.new('RGB', (width, height), (240, 240, 250))
        draw = ImageDraw.Draw(image)
        
        # Try to load a font
        try:
            title_font = ImageFont.truetype('DejaVuSans-Bold.ttf', 36)
            header_font = ImageFont.truetype('DejaVuSans-Bold.ttf', 24)
            text_font = ImageFont.truetype('DejaVuSans.ttf', 18)
            model_font = ImageFont.truetype('DejaVuSans-Bold.ttf', 20)
        except IOError:
            # Fallback to default font
            title_font = ImageFont.load_default()
            header_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
            model_font = ImageFont.load_default()
        
        # Add title and date
        draw.text((width//2, 30), title, fill=(30, 30, 100), font=title_font, anchor="ms")
        draw.text((width//2, 70), f"Forecast for {date_str}", fill=(50, 50, 120), font=header_font, anchor="ms")
        
        # Add horizontal line
        draw.line([(50, 100), (width-50, 100)], fill=(180, 180, 220), width=2)
        
        # Get all prediction keys
        all_keys = set()
        for model_preds in model_predictions.values():
            all_keys.update(model_preds.keys())
        
        # Sort keys
        sorted_keys = sorted(all_keys)
        
        # Calculate column width
        num_columns = len(model_predictions) + (1 if actual else 0)
        column_width = (width - 300) // num_columns
        
        # Draw header row
        y_pos = 130
        x_pos = 300
        
        # Draw prediction names column
        draw.text((150, y_pos), "Prediction", fill=(30, 30, 100), font=model_font, anchor="ms")
        
        # Draw model names row
        for model_name in model_predictions.keys():
            draw.text((x_pos + column_width//2, y_pos), model_name, fill=(30, 30, 100), font=model_font, anchor="ms")
            x_pos += column_width
        
        # Draw actual column header if provided
        if actual:
            draw.text((x_pos + column_width//2, y_pos), "Actual", fill=(30, 30, 100), font=model_font, anchor="ms")
        
        # Add horizontal line
        y_pos += 30
        draw.line([(50, y_pos), (width-50, y_pos)], fill=(180, 180, 220), width=2)
        y_pos += 20
        
        # Draw rows for each prediction
        for key in sorted_keys:
            # Format the prediction name
            formatted_key = key.replace('_', ' ').title()
            
            # Draw prediction name
            draw.text((150, y_pos), formatted_key, fill=(50, 50, 120), font=text_font, anchor="ms")
            
            # Draw values for each model
            x_pos = 300
            for model_name, preds in model_predictions.items():
                if key in preds:
                    value = preds[key]
                    # Format the value
                    if isinstance(value, float):
                        formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    
                    draw.text((x_pos + column_width//2, y_pos), formatted_value, 
                             fill=(50, 50, 120), font=text_font, anchor="ms")
                
                x_pos += column_width
            
            # Draw actual value if provided
            if actual and key in actual:
                value = actual[key]
                # Format the value
                if isinstance(value, float):
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                
                draw.text((x_pos + column_width//2, y_pos), formatted_value, 
                         fill=(0, 100, 0), font=text_font, anchor="ms")
            
            y_pos += 40
        
        # Add a disclaimer
        disclaimer = "This comparison is based on machine learning models and may not reflect actual weather conditions."
        draw.text((width//2, height-30), disclaimer, fill=(100, 100, 150), font=text_font, anchor="ms")
        
        # Add timestamp
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        draw.text((width-50, height-10), f"Generated: {now}", fill=(100, 100, 150), font=ImageFont.load_default(), anchor="rs")
        
        # Save the image
        if save_path is None:
            date_part = date_str.replace('-', '')
            filename = f"model_comparison_{date_part}.png"
            save_path = os.path.join(self.output_dir, filename)
        
        image.save(save_path, quality=95)
        
        logger.info(f"Saved model comparison image to {save_path}")
        
        return save_path 