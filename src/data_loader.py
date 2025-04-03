import os
import glob
import pygrib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeatherDataLoader:
    """
    A class for loading and processing weather forecast data in GRIB2 format
    """
    
    def __init__(self, data_dir='/home/jackson/Downloads/forecasts'):
        """
        Initialize the data loader with the path to the forecast data
        
        Args:
            data_dir (str): Path to the directory containing forecast data
        """
        self.data_dir = data_dir
        self.forecast_types = ['hail', 'sig_hail', 'hail_abs_calib', 'sig_hail_abs_calib']
        
    def get_available_dates(self, start_date=None, end_date=None):
        """
        Get list of available dates in the data directory
        
        Args:
            start_date (str): Start date in YYYYMMDD format
            end_date (str): End date in YYYYMMDD format
            
        Returns:
            list: List of available dates
        """
        all_dirs = sorted(glob.glob(os.path.join(self.data_dir, '??????')))
        
        # Extract year-month directories
        ym_dirs = [os.path.basename(d) for d in all_dirs]
        
        available_dates = []
        
        for ym_dir in ym_dirs:
            year_month_path = os.path.join(self.data_dir, ym_dir)
            day_dirs = sorted(glob.glob(os.path.join(year_month_path, '????????')))
            available_dates.extend([os.path.basename(d) for d in day_dirs])
        
        if start_date:
            available_dates = [d for d in available_dates if d >= start_date]
        
        if end_date:
            available_dates = [d for d in available_dates if d <= end_date]
            
        logger.info(f"Found {len(available_dates)} available dates")
        return available_dates
    
    def load_grib_data(self, grib_file):
        """
        Load data from a GRIB2 file
        
        Args:
            grib_file (str): Path to the GRIB2 file
            
        Returns:
            dict: Dictionary containing the data
        """
        try:
            grbs = pygrib.open(grib_file)
            data = {}
            
            for grb in grbs:
                data_values, lats, lons = grb.data()
                name = grb.name
                level = grb.level
                forecast_time = grb.forecastTime
                
                key = f"{name}_{level}_{forecast_time}"
                data[key] = {
                    'values': data_values,
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
            logger.error(f"Error loading GRIB file {grib_file}: {e}")
            return None
    
    def load_date_data(self, date_str, time_of_day='t0z'):
        """
        Load all data for a specific date and time
        
        Args:
            date_str (str): Date string in YYYYMMDD format
            time_of_day (str): Time of day (t0z, t12z, etc.)
            
        Returns:
            dict: Dictionary containing all data for the date
        """
        year_month = date_str[:6]
        date_path = os.path.join(self.data_dir, year_month, date_str, time_of_day)
        
        if not os.path.exists(date_path):
            logger.warning(f"Path does not exist: {date_path}")
            return None
        
        data = {}
        
        for forecast_type in self.forecast_types:
            # Search for files matching the pattern
            pattern = f"*{forecast_type}*{date_str}*{time_of_day}*.grib2"
            files = glob.glob(os.path.join(date_path, pattern))
            
            if not files:
                continue
                
            for file in files:
                file_basename = os.path.basename(file)
                grib_data = self.load_grib_data(file)
                
                if grib_data:
                    data[file_basename] = grib_data
        
        return data
    
    def create_dataset(self, start_date, end_date, times_of_day=['t0z', 't12z']):
        """
        Create a dataset from a range of dates
        
        Args:
            start_date (str): Start date in YYYYMMDD format
            end_date (str): End date in YYYYMMDD format
            times_of_day (list): List of times of day to include
            
        Returns:
            dict: Dictionary containing dataset
        """
        dates = self.get_available_dates(start_date, end_date)
        dataset = {}
        
        for date_str in tqdm(dates, desc="Loading dates"):
            date_data = {}
            
            for tod in times_of_day:
                tod_data = self.load_date_data(date_str, tod)
                
                if tod_data:
                    date_data[tod] = tod_data
            
            if date_data:
                dataset[date_str] = date_data
        
        logger.info(f"Created dataset with {len(dataset)} dates")
        return dataset
    
    def get_image_paths(self, date_str, time_of_day='t0z', img_type='png'):
        """
        Get paths to image files for a specific date and time
        
        Args:
            date_str (str): Date string in YYYYMMDD format
            time_of_day (str): Time of day (t0z, t12z, etc.)
            img_type (str): Image type (png or pdf)
            
        Returns:
            list: List of image paths
        """
        year_month = date_str[:6]
        date_path = os.path.join(self.data_dir, year_month, date_str, time_of_day)
        
        if not os.path.exists(date_path):
            return []
        
        pattern = f"*{date_str}*{time_of_day}*.{img_type}"
        return glob.glob(os.path.join(date_path, pattern)) 