import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import re
from tqdm import tqdm
from shapely.geometry import Point, Polygon
import geopandas as gpd

from .spc_data_fetcher import SPCDataFetcher
from .spc_integration import SPCIntegration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SPCFeatureIntegrator:
    """
    Class for integrating SPC data as features for model training
    
    This class adds additional features from SPC data to enhance model training.
    """
    
    def __init__(self, spc_cache_dir='./data/spc_cache', historical_data_dir='./data/historical_spc'):
        """
        Initialize the SPC feature integrator
        
        Args:
            spc_cache_dir (str): Path to cache SPC data
            historical_data_dir (str): Path to store historical SPC data
        """
        self.spc_cache_dir = spc_cache_dir
        self.historical_data_dir = historical_data_dir
        self.spc_fetcher = SPCDataFetcher(cache_dir=spc_cache_dir)
        self.spc_integration = SPCIntegration(spc_cache_dir=spc_cache_dir)
        
        # Ensure directories exist
        os.makedirs(self.historical_data_dir, exist_ok=True)
        
        # Load US boundaries if available
        self.states_gdf = None
        self.counties_gdf = None
        try:
            self._load_us_boundaries()
        except Exception as e:
            logger.warning(f"Could not load US boundaries for feature integration: {e}")
    
    def _load_us_boundaries(self):
        """
        Load US state and county boundaries for spatial feature calculation
        """
        try:
            # Check if we have cached boundaries
            states_file = os.path.join(self.spc_cache_dir, 'us_states.geojson')
            counties_file = os.path.join(self.spc_cache_dir, 'us_counties.geojson')
            
            if os.path.exists(states_file) and os.path.exists(counties_file):
                self.states_gdf = gpd.read_file(states_file)
                self.counties_gdf = gpd.read_file(counties_file)
                logger.info("Loaded US boundaries from cache")
                return
            
            # If not cached, defer to SPC integration to download them
            self.spc_integration._load_us_boundaries()
            self.states_gdf = self.spc_integration.states_gdf
            self.counties_gdf = self.spc_integration.counties_gdf
            
        except Exception as e:
            logger.error(f"Error loading US boundaries: {e}")
            raise
    
    def fetch_historical_spc_data(self, start_date, end_date):
        """
        Fetch and store historical SPC data for a date range
        
        Args:
            start_date (str): Start date in YYYYMMDD format
            end_date (str): End date in YYYYMMDD format
            
        Returns:
            dict: Historical SPC data by date
        """
        logger.info(f"Fetching historical SPC data from {start_date} to {end_date}")
        
        # Convert dates to datetime objects
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        
        historical_data = {}
        
        # Iterate through each date
        current_dt = start_dt
        while current_dt <= end_dt:
            current_date = current_dt.strftime('%Y%m%d')
            logger.info(f"Fetching data for {current_date}")
            
            # Check if we already have this date cached
            cache_file = os.path.join(self.historical_data_dir, f"spc_data_{current_date}.json")
            
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        historical_data[current_date] = json.load(f)
                    logger.info(f"Loaded cached SPC data for {current_date}")
                    current_dt += timedelta(days=1)
                    continue
                except Exception as e:
                    logger.warning(f"Error reading cached data for {current_date}: {e}")
            
            # Fetch severe weather reports for this date
            reports = self.spc_fetcher.get_severe_weather_reports(date=current_date)
            
            # Fetch mesoscale discussions
            discussions = self.spc_fetcher.get_mesoscale_discussions(start_date=current_date, end_date=current_date)
            
            # Fetch watches
            watches = self.spc_fetcher.get_watches(date=current_date)
            
            # Store the data
            date_data = {
                'date': current_date,
                'reports': reports.to_dict('records') if not reports.empty else [],
                'discussions': discussions,
                'watches': watches
            }
            
            # Cache the data
            try:
                with open(cache_file, 'w') as f:
                    json.dump(date_data, f)
            except Exception as e:
                logger.warning(f"Error caching SPC data for {current_date}: {e}")
            
            historical_data[current_date] = date_data
            current_dt += timedelta(days=1)
        
        logger.info(f"Fetched historical SPC data for {len(historical_data)} dates")
        return historical_data
    
    def extract_spc_features(self, date, lat, lon, historical_data=None):
        """
        Extract SPC-based features for a specific date and location
        
        Args:
            date (str): Date in YYYYMMDD format
            lat (float): Latitude
            lon (float): Longitude
            historical_data (dict): Historical SPC data if available
            
        Returns:
            dict: SPC features for the given date and location
        """
        # If no historical data provided, try to load from cache
        if historical_data is None or date not in historical_data:
            cache_file = os.path.join(self.historical_data_dir, f"spc_data_{date}.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        date_data = json.load(f)
                except Exception:
                    # If can't load cached data, fetch it directly
                    reports = self.spc_fetcher.get_severe_weather_reports(date=date)
                    discussions = self.spc_fetcher.get_mesoscale_discussions(start_date=date, end_date=date)
                    watches = self.spc_fetcher.get_watches(date=date)
                    
                    date_data = {
                        'date': date,
                        'reports': reports.to_dict('records') if not reports.empty else [],
                        'discussions': discussions,
                        'watches': watches
                    }
            else:
                # If no cached data, fetch it directly
                reports = self.spc_fetcher.get_severe_weather_reports(date=date)
                discussions = self.spc_fetcher.get_mesoscale_discussions(start_date=date, end_date=date)
                watches = self.spc_fetcher.get_watches(date=date)
                
                date_data = {
                    'date': date,
                    'reports': reports.to_dict('records') if not reports.empty else [],
                    'discussions': discussions,
                    'watches': watches
                }
        else:
            date_data = historical_data[date]
        
        # Calculate spatial features based on SPC data
        features = {}
        
        # Get reports within different radiuses
        for radius_km in [50, 100, 250]:
            nearby_reports = self._get_nearby_reports_from_data(
                lat, lon, date_data.get('reports', []), radius_km=radius_km
            )
            
            # Count by type
            tornado_count = sum(1 for r in nearby_reports if 'type' in r and r['type'] == 'TORNADO')
            wind_count = sum(1 for r in nearby_reports if 'type' in r and r['type'] == 'WIND')
            hail_count = sum(1 for r in nearby_reports if 'type' in r and r['type'] == 'HAIL')
            
            features[f'spc_tornado_reports_{radius_km}km'] = tornado_count
            features[f'spc_wind_reports_{radius_km}km'] = wind_count
            features[f'spc_hail_reports_{radius_km}km'] = hail_count
            features[f'spc_total_reports_{radius_km}km'] = len(nearby_reports)
        
        # Check if location was in any watches
        in_tornado_watch = False
        in_severe_watch = False
        
        for watch in date_data.get('watches', []):
            # Note: This is a simplified check - ideally we would parse watch geometry
            # from the watch image or text. This would require more complex parsing.
            if 'text' in watch:
                # Basic check - see if state abbreviation or county name is in watch text
                point = Point(lon, lat)
                state_match = self._get_state_for_point(point)
                county_match = self._get_county_for_point(point)
                
                state_in_text = state_match and state_match in watch['text']
                county_in_text = county_match and county_match in watch['text']
                
                if state_in_text or county_in_text:
                    if watch.get('type') == 'Tornado':
                        in_tornado_watch = True
                    elif watch.get('type') == 'Severe Thunderstorm':
                        in_severe_watch = True
        
        features['spc_in_tornado_watch'] = 1 if in_tornado_watch else 0
        features['spc_in_severe_watch'] = 1 if in_severe_watch else 0
        features['spc_in_any_watch'] = 1 if (in_tornado_watch or in_severe_watch) else 0
        
        # Check if location was in any mesoscale discussions
        in_md = False
        md_count = 0
        
        for md in date_data.get('discussions', []):
            # Simple check - see if state or county is mentioned in the MD text
            if 'text' in md:
                point = Point(lon, lat)
                state_match = self._get_state_for_point(point)
                county_match = self._get_county_for_point(point)
                
                state_in_text = state_match and state_match in md['text']
                county_in_text = county_match and county_match in md['text']
                
                if state_in_text or county_in_text:
                    in_md = True
                    md_count += 1
        
        features['spc_in_mesoscale_discussion'] = 1 if in_md else 0
        features['spc_mesoscale_discussion_count'] = md_count
        
        # Location information
        location_info = self._get_location_info(lat, lon)
        features['location_state'] = location_info.get('state')
        features['location_county'] = location_info.get('county')
        
        # Calculate some climatological features
        features.update(self._calculate_climatological_features(date, lat, lon))
        
        return features
    
    def _get_nearby_reports_from_data(self, lat, lon, reports, radius_km=100):
        """
        Get nearby reports from report data within a radius
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            reports (list): List of report dictionaries
            radius_km (float): Radius in kilometers
            
        Returns:
            list: Nearby reports
        """
        if not reports:
            return []
        
        nearby = []
        
        for report in reports:
            if 'lat' in report and 'lon' in report:
                try:
                    report_lat = float(report['lat'])
                    report_lon = float(report['lon'])
                    
                    # Calculate distance (approximate using Haversine formula)
                    distance = self._haversine(lat, lon, report_lat, report_lon)
                    
                    if distance <= radius_km:
                        report_copy = report.copy()
                        report_copy['distance_km'] = distance
                        nearby.append(report_copy)
                        
                except (ValueError, TypeError):
                    continue
        
        return sorted(nearby, key=lambda x: x.get('distance_km', float('inf')))
    
    def _haversine(self, lat1, lon1, lat2, lon2):
        """
        Calculate the Haversine distance between two points
        
        Args:
            lat1 (float): Latitude of point 1
            lon1 (float): Longitude of point 1
            lat2 (float): Latitude of point 2
            lon2 (float): Longitude of point 2
            
        Returns:
            float: Distance in kilometers
        """
        # Earth radius in kilometers
        R = 6371.0
        
        # Convert degrees to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # Differences
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Haversine formula
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = R * c
        
        return distance
    
    def _get_location_info(self, lat, lon):
        """
        Get location information for a point
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            
        Returns:
            dict: Location information
        """
        point = Point(lon, lat)
        
        state = self._get_state_for_point(point)
        county = self._get_county_for_point(point)
        
        return {
            'state': state,
            'county': county,
            'country': 'United States'
        }
    
    def _get_state_for_point(self, point):
        """
        Get state name for a point
        
        Args:
            point (Point): Shapely Point object
            
        Returns:
            str: State name or None
        """
        if self.states_gdf is None:
            return None
        
        state_match = self.states_gdf[self.states_gdf.geometry.contains(point)]
        if not state_match.empty:
            return state_match.iloc[0]['NAME']
        
        return None
    
    def _get_county_for_point(self, point):
        """
        Get county name for a point
        
        Args:
            point (Point): Shapely Point object
            
        Returns:
            str: County name or None
        """
        if self.counties_gdf is None:
            return None
        
        county_match = self.counties_gdf[self.counties_gdf.geometry.contains(point)]
        if not county_match.empty:
            return county_match.iloc[0]['NAME']
        
        return None
    
    def _calculate_climatological_features(self, date, lat, lon):
        """
        Calculate climatological features based on historical data
        
        Args:
            date (str): Date in YYYYMMDD format
            lat (float): Latitude
            lon (float): Longitude
            
        Returns:
            dict: Climatological features
        """
        features = {}
        
        # Extract month and day information
        dt = datetime.strptime(date, '%Y%m%d')
        month = dt.month
        day = dt.day
        doy = dt.timetuple().tm_yday  # Day of year
        
        features['month'] = month
        features['day'] = day
        features['day_of_year'] = doy
        
        # Seasonality features - using sine and cosine transforms to capture cyclical nature
        features['season_sin'] = np.sin(2 * np.pi * doy / 365)
        features['season_cos'] = np.cos(2 * np.pi * doy / 365)
        
        # Peak tornado season indicator (March-June)
        features['tornado_season'] = 1 if 3 <= month <= 6 else 0
        
        # Peak hail season indicator (March-September)
        features['hail_season'] = 1 if 3 <= month <= 9 else 0
        
        # Some regions have specific severe weather seasons
        # These are simplified regional indicators - more sophisticated regions would be better
        region = self._determine_region(lat, lon)
        features['region'] = region
        
        if region == 'plains':
            # Plains tornado season (April-June)
            features['region_peak_season'] = 1 if 4 <= month <= 6 else 0
        elif region == 'southeast':
            # Southeast tornado season (March-May) and secondary (November)
            features['region_peak_season'] = 1 if (3 <= month <= 5) or month == 11 else 0
        elif region == 'midwest':
            # Midwest tornado season (April-July)
            features['region_peak_season'] = 1 if 4 <= month <= 7 else 0
        else:
            features['region_peak_season'] = 0
        
        return features
    
    def _determine_region(self, lat, lon):
        """
        Determine the US region for a lat/lon point
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            
        Returns:
            str: Region name
        """
        # Very simplified region determination
        # Plains (Tornado Alley)
        if 30 <= lat <= 48 and -105 <= lon <= -90:
            return 'plains'
        # Southeast
        elif 25 <= lat <= 36.5 and -90 <= lon <= -75:
            return 'southeast'
        # Midwest
        elif 36.5 <= lat <= 49 and -90 <= lon <= -75:
            return 'midwest'
        # Northeast
        elif 37 <= lat <= 47 and -80 <= lon <= -67:
            return 'northeast'
        # West
        elif 30 <= lat <= 49 and -125 <= lon <= -105:
            return 'west'
        # Other
        else:
            return 'other'
    
    def add_spc_features_to_dataset(self, dataset, historical_data=None):
        """
        Add SPC features to an existing dataset
        
        Args:
            dataset (dict): Dataset organized by date and time of day
            historical_data (dict): Historical SPC data by date
            
        Returns:
            dict: Enhanced dataset with SPC features
        """
        logger.info("Adding SPC features to dataset")
        
        # Fetch historical data if not provided
        if historical_data is None:
            # Get all unique dates in the dataset
            all_dates = list(dataset.keys())
            if not all_dates:
                logger.warning("No dates in dataset")
                return dataset
                
            start_date = min(all_dates)
            end_date = max(all_dates)
            
            historical_data = self.fetch_historical_spc_data(start_date, end_date)
        
        # Create enhanced dataset
        enhanced_dataset = {}
        
        for date, date_data in tqdm(dataset.items(), desc="Adding SPC features"):
            enhanced_dataset[date] = {}
            
            for tod, tod_data in date_data.items():
                enhanced_dataset[date][tod] = {}
                
                for file_name, file_data in tod_data.items():
                    # Get coordinates if available
                    lat_array = file_data.get('lat', [])
                    lon_array = file_data.get('lon', [])
                    
                    # Create a copy of the file data
                    enhanced_file_data = file_data.copy()
                    
                    # If we have coordinates, add SPC features for each point
                    if len(lat_array) > 0 and len(lon_array) > 0:
                        # Initialize SPC feature arrays
                        # For simplicity, we'll use the center point for SPC features
                        # In a production system, you'd calculate these for each grid point
                        
                        # Get center point
                        center_lat = np.mean(lat_array)
                        center_lon = np.mean(lon_array)
                        
                        # Extract SPC features for the center point
                        spc_features = self.extract_spc_features(date, center_lat, center_lon, historical_data)
                        
                        # Add SPC features to the file data
                        for feature_name, feature_value in spc_features.items():
                            enhanced_file_data[feature_name] = feature_value
                    
                    enhanced_dataset[date][tod][file_name] = enhanced_file_data
        
        logger.info("Added SPC features to dataset")
        return enhanced_dataset
    
    def extract_spc_features_for_prediction(self, prediction_data):
        """
        Extract SPC features for prediction data
        
        Args:
            prediction_data (dict): Prediction data with lat/lon points
            
        Returns:
            dict: Prediction data with added SPC features
        """
        # Get current date
        current_date = datetime.now().strftime('%Y%m%d')
        
        # Get center point for simplicity
        if 'lat' in prediction_data and 'lon' in prediction_data:
            lat_array = prediction_data['lat']
            lon_array = prediction_data['lon']
            
            if len(lat_array) > 0 and len(lon_array) > 0:
                center_lat = np.mean(lat_array)
                center_lon = np.mean(lon_array)
                
                # Extract SPC features
                spc_features = self.extract_spc_features(current_date, center_lat, center_lon)
                
                # Add SPC features to prediction data
                enhanced_prediction = prediction_data.copy()
                enhanced_prediction['spc_features'] = spc_features
                
                return enhanced_prediction
        
        return prediction_data 