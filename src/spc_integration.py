import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.pyplot import cm
import requests
from io import BytesIO
from PIL import Image
import geopandas as gpd
import json
from shapely.geometry import Point, Polygon

from .spc_data_fetcher import SPCDataFetcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SPCIntegration:
    """
    Class to integrate SPC data with the prediction system
    """
    
    def __init__(self, model_output_dir='./output', spc_cache_dir='./data/spc_cache'):
        """
        Initialize the SPC integration
        
        Args:
            model_output_dir (str): Directory for storing model outputs
            spc_cache_dir (str): Directory for caching SPC data
        """
        self.model_output_dir = model_output_dir
        self.spc_fetcher = SPCDataFetcher(cache_dir=spc_cache_dir)
        self.states_gdf = None
        self.counties_gdf = None
        
        # Ensure output directory exists
        os.makedirs(self.model_output_dir, exist_ok=True)
        
        # Try to load US state and county boundaries for mapping
        try:
            self._load_us_boundaries()
        except Exception as e:
            logger.warning(f"Could not load US boundaries: {e}")
    
    def _load_us_boundaries(self):
        """
        Load US state and county boundaries for mapping
        """
        try:
            # Check if we have cached boundaries
            states_file = os.path.join(self.spc_fetcher.cache_dir, 'us_states.geojson')
            counties_file = os.path.join(self.spc_fetcher.cache_dir, 'us_counties.geojson')
            
            if os.path.exists(states_file) and os.path.exists(counties_file):
                self.states_gdf = gpd.read_file(states_file)
                self.counties_gdf = gpd.read_file(counties_file)
                logger.info("Loaded US boundaries from cache")
                return
            
            # If not cached, download from Natural Earth or Census
            self.states_gdf = gpd.read_file('https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_state_500k.zip')
            self.counties_gdf = gpd.read_file('https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_500k.zip')
            
            # Cache for future use
            os.makedirs(os.path.dirname(states_file), exist_ok=True)
            self.states_gdf.to_file(states_file, driver='GeoJSON')
            self.counties_gdf.to_file(counties_file, driver='GeoJSON')
            
            logger.info("Downloaded and cached US boundaries")
        except Exception as e:
            logger.error(f"Error loading US boundaries: {e}")
            raise
    
    def get_current_spc_outlook(self, day=1):
        """
        Get the current SPC convective outlook
        
        Args:
            day (int): Day number (1 for today, 2 for tomorrow, etc.)
            
        Returns:
            dict: Outlook data
        """
        return self.spc_fetcher.get_current_convective_outlook(day=day)
    
    def get_recent_severe_reports(self, days_back=7):
        """
        Get severe weather reports for the last n days
        
        Args:
            days_back (int): Number of days to look back
            
        Returns:
            pd.DataFrame: Combined reports from the last n days
        """
        all_reports = []
        
        for i in range(days_back):
            report_date = (datetime.now() - timedelta(days=i+1)).strftime('%Y%m%d')
            daily_reports = self.spc_fetcher.get_severe_weather_reports(date=report_date)
            
            if not daily_reports.empty:
                daily_reports['date'] = report_date
                all_reports.append(daily_reports)
        
        if all_reports:
            return pd.concat(all_reports, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def download_outlook_image(self, url):
        """
        Download an outlook image from a URL
        
        Args:
            url (str): URL of the image
            
        Returns:
            PIL.Image: Downloaded image
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {e}")
            return None
    
    def create_risk_verification_plot(self, prediction_data, outlook_data, output_file=None):
        """
        Create a plot comparing model prediction with SPC outlook
        
        Args:
            prediction_data (dict): Model prediction data with lat/lon coordinates
            outlook_data (dict): SPC outlook data
            output_file (str): Path to save the output file, if None will show the plot
            
        Returns:
            str: Path to the saved plot if output_file is provided, None otherwise
        """
        # Default output file if not provided
        if output_file is None:
            output_file = os.path.join(
                self.model_output_dir, 
                f"verification_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
        
        # Download SPC outlook image
        spc_image = None
        if outlook_data and outlook_data.get('categorical'):
            spc_image = self.download_outlook_image(outlook_data['categorical'])
        
        # Plot setup
        fig, axes = plt.subplots(1, 2 if spc_image else 1, figsize=(20, 10))
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # Plot model prediction
        ax = axes[0]
        
        # If we have loaded US boundaries, use them
        if self.states_gdf is not None:
            self.states_gdf.boundary.plot(ax=ax, linewidth=1, color='black')
        
        # Create custom colormap for tornado risk levels
        tornado_cmap = plt.cm.get_cmap('YlOrRd')
        
        # Example prediction data plot (assumes prediction_data has lat, lon, and probability)
        # In a real implementation, this would use the actual prediction format
        if prediction_data and 'lat' in prediction_data and 'lon' in prediction_data and 'probability' in prediction_data:
            sc = ax.scatter(
                prediction_data['lon'], 
                prediction_data['lat'], 
                c=prediction_data['probability'], 
                cmap=tornado_cmap, 
                s=20, 
                alpha=0.7,
                vmin=0, 
                vmax=1
            )
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Tornado Risk Probability')
            
            # Add risk level annotations to the colorbar
            risk_levels = [
                (0.01, 'Marginal (1%)'),
                (0.05, 'Slight (5%)'),
                (0.15, 'Enhanced (15%)'),
                (0.30, 'Moderate (30%)'),
                (0.60, 'High (60%)')
            ]
            
            for level, label in risk_levels:
                cbar.ax.axhline(y=level, color='black', linestyle='-', linewidth=1, alpha=0.6)
                cbar.ax.text(1.1, level, label, va='center', ha='left', fontsize=8, color='black')
        
        ax.set_title('Model Prediction')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Add rectangular key for tornado risk levels in the bottom right
        # Create a legend box in the bottom right corner
        legend_ax = fig.add_axes([0.82, 0.05, 0.15, 0.25]) if not spc_image else fig.add_axes([0.45, 0.05, 0.15, 0.25])
        legend_ax.axis('off')
        legend_ax.set_title('Tornado Risk Levels', fontsize=10)
        
        # Create colored boxes for each risk level
        risk_categories = [
            ('Marginal', 'Marginal Risk (1-2%)', tornado_cmap(0.1)),
            ('Slight', 'Slight Risk (5-10%)', tornado_cmap(0.3)),
            ('Enhanced', 'Enhanced Risk (15-20%)', tornado_cmap(0.5)),
            ('Moderate', 'Moderate Risk (30-45%)', tornado_cmap(0.7)),
            ('High', 'High Risk (60%+)', tornado_cmap(0.9))
        ]
        
        for i, (name, label, color) in enumerate(risk_categories):
            y_pos = 0.8 - (i * 0.15)
            legend_ax.add_patch(plt.Rectangle((0.05, y_pos), 0.2, 0.1, facecolor=color, edgecolor='black'))
            legend_ax.text(0.3, y_pos + 0.05, label, va='center', fontsize=8)
        
        # Plot SPC outlook if available
        if spc_image and len(axes) > 1:
            ax = axes[1]
            ax.imshow(spc_image)
            ax.set_title('SPC Convective Outlook')
            ax.axis('off')
        
        # Add timestamp
        plt.figtext(
            0.5, 0.01, 
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ha='center'
        )
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved verification plot to {output_file}")
        return output_file
    
    def generate_location_risk_report(self, lat, lon, prediction_data, outlook_data=None):
        """
        Generate a risk report for a specific location
        
        Args:
            lat (float): Latitude of the location
            lon (float): Longitude of the location
            prediction_data (dict): Model prediction data
            outlook_data (dict): SPC outlook data, if available
            
        Returns:
            dict: Risk report for the location
        """
        # Get SPC outlook if not provided
        if outlook_data is None:
            outlook_data = self.get_current_spc_outlook()
        
        # Get location information
        location_info = self._get_location_info(lat, lon)
        
        # Extract risk from prediction data for this location
        # In a real implementation, this would use actual prediction format
        model_risk = self._extract_model_risk_at_point(lat, lon, prediction_data)
        
        # Get recent reports near this location
        recent_reports = self._get_nearby_reports(lat, lon, radius_km=100)
        
        # Build the report
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'location': {
                'lat': lat,
                'lon': lon,
                'info': location_info
            },
            'model_prediction': model_risk,
            'spc_outlook': {
                'day': outlook_data.get('day') if outlook_data else None,
                'valid_time': outlook_data.get('valid_time') if outlook_data else None,
                'discussion_url': outlook_data.get('discussion_url') if outlook_data else None
            },
            'recent_reports': {
                'count': len(recent_reports),
                'reports': recent_reports[:10]  # Limit to 10 reports
            }
        }
        
        return report
    
    def _get_location_info(self, lat, lon):
        """
        Get location information for a point
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            
        Returns:
            dict: Location information
        """
        location_info = {
            'state': None,
            'county': None,
            'country': 'United States'  # Default assumption
        }
        
        # If we don't have boundary data, try to get from a reverse geocoding service
        if self.states_gdf is None or self.counties_gdf is None:
            try:
                # Example using the free Nominatim service (should use with delay in production)
                response = requests.get(
                    f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}",
                    headers={'User-Agent': 'WeatherPredictionApp/1.0'},
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    address = data.get('address', {})
                    location_info['state'] = address.get('state')
                    location_info['county'] = address.get('county')
                    location_info['country'] = address.get('country')
            except Exception as e:
                logger.warning(f"Error getting location info from geocoding: {e}")
        else:
            # Use our loaded boundary data
            point = Point(lon, lat)
            
            # Find state
            state_match = self.states_gdf[self.states_gdf.geometry.contains(point)]
            if not state_match.empty:
                location_info['state'] = state_match.iloc[0]['NAME']
            
            # Find county
            county_match = self.counties_gdf[self.counties_gdf.geometry.contains(point)]
            if not county_match.empty:
                location_info['county'] = county_match.iloc[0]['NAME']
        
        return location_info
    
    def _extract_model_risk_at_point(self, lat, lon, prediction_data):
        """
        Extract model risk at a specific point
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            prediction_data (dict): Model prediction data
            
        Returns:
            dict: Risk information at the point
        """
        # This is a simple example - actual implementation would depend on prediction format
        if (not prediction_data or 'lat' not in prediction_data or 
            'lon' not in prediction_data or 'probability' not in prediction_data):
            return {
                'probability': None,
                'category': None
            }
        
        # Find the closest point in the prediction data
        pred_lats = np.array(prediction_data['lat'])
        pred_lons = np.array(prediction_data['lon'])
        pred_probs = np.array(prediction_data['probability'])
        
        # Calculate distances to all points
        distances = np.sqrt((pred_lats - lat)**2 + (pred_lons - lon)**2)
        
        # Find the closest point
        closest_idx = np.argmin(distances)
        probability = pred_probs[closest_idx]
        
        # Categorize the risk
        if probability < 0.02:
            category = 'Marginal'
        elif probability < 0.05:
            category = 'Slight'
        elif probability < 0.15:
            category = 'Enhanced'
        elif probability < 0.30:
            category = 'Moderate'
        else:
            category = 'High'
        
        return {
            'probability': float(probability),
            'category': category
        }
    
    def _get_nearby_reports(self, lat, lon, radius_km=100, days_back=7):
        """
        Get recent severe weather reports near a location
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            radius_km (float): Radius in kilometers
            days_back (int): Number of days to look back
            
        Returns:
            list: List of nearby reports
        """
        # Get recent reports
        reports_df = self.get_recent_severe_reports(days_back=days_back)
        
        if reports_df.empty:
            return []
        
        # Filter reports by distance
        # Convert lat/lon columns to numeric if they aren't already
        for col in ['lat', 'lon']:
            if col in reports_df.columns:
                reports_df[col] = pd.to_numeric(reports_df[col], errors='coerce')
        
        # Calculate distance to each report (approximate using Haversine formula)
        earth_radius_km = 6371.0
        
        def haversine(lat1, lon1, lat2, lon2):
            """Calculate the Haversine distance between two points"""
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = (np.sin(dlat/2)**2 + 
                 np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            return earth_radius_km * c
        
        # Assuming the reports DataFrame has 'lat' and 'lon' columns
        if 'lat' in reports_df.columns and 'lon' in reports_df.columns:
            reports_df['distance_km'] = reports_df.apply(
                lambda row: haversine(lat, lon, row['lat'], row['lon']), 
                axis=1
            )
            
            # Filter by radius
            nearby_reports = reports_df[reports_df['distance_km'] <= radius_km].sort_values('distance_km')
            
            # Convert to list of dictionaries
            return nearby_reports.to_dict('records')
        
        return []
    
    def save_location_report(self, report, output_file=None):
        """
        Save a location risk report to a file
        
        Args:
            report (dict): Report data
            output_file (str): Path to save the report, if None will generate a path
            
        Returns:
            str: Path to the saved report
        """
        if output_file is None:
            lat = report['location']['lat']
            lon = report['location']['lon']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(
                self.model_output_dir, 
                f"location_report_{lat:.2f}_{lon:.2f}_{timestamp}.json"
            )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved location report to {output_file}")
        return output_file 