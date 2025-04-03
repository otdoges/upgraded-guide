import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from bs4 import BeautifulSoup
import json
import re
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SPCDataFetcher:
    """
    A class for fetching and processing data from the Storm Prediction Center (SPC) NOAA website
    """
    
    def __init__(self, cache_dir='./data/spc_cache'):
        """
        Initialize the SPC data fetcher with cache directory
        
        Args:
            cache_dir (str): Path to the directory for caching SPC data
        """
        self.cache_dir = cache_dir
        self.base_url = "https://www.spc.noaa.gov"
        self.outlook_url = f"{self.base_url}/products/outlook"
        self.md_url = f"{self.base_url}/products/md"
        self.watch_url = f"{self.base_url}/products/watch"
        self.reports_url = f"{self.base_url}/climo/reports"
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _fetch_url(self, url, timeout=10):
        """
        Fetch data from a URL with error handling
        
        Args:
            url (str): URL to fetch
            timeout (int): Timeout in seconds
            
        Returns:
            str: Content of the URL or None if error
        """
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def get_current_convective_outlook(self, day=1, cache=True):
        """
        Get the current convective outlook for a specific day
        
        Args:
            day (int): Day number (1 for today, 2 for tomorrow, etc.)
            cache (bool): Whether to use cached data if available
            
        Returns:
            dict: Convective outlook data including probability maps URLs
        """
        if not 1 <= day <= 8:
            logger.error(f"Day must be between 1 and 8, got {day}")
            return None
        
        cache_file = os.path.join(self.cache_dir, f"convective_outlook_day{day}_{datetime.now().strftime('%Y%m%d')}.json")
        
        # Check cache first if enabled
        if cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    if (datetime.now() - datetime.strptime(data['fetch_time'], "%Y-%m-%d %H:%M:%S")).total_seconds() < 3600:
                        logger.info(f"Using cached convective outlook for day {day}")
                        return data
            except Exception as e:
                logger.warning(f"Error reading cache file: {e}")
        
        # Fetch the data
        if day <= 3:
            url = f"{self.outlook_url}/day{day}_otlk_latest.html"
        else:
            url = f"{self.outlook_url}/day4-8/day{day}_otlk_latest.html"
        
        html = self._fetch_url(url)
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract outlook information
        outlook_data = {
            'day': day,
            'fetch_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'outlook_time': None,
            'valid_time': None,
            'categorical': None,
            'tornado': None,
            'wind': None,
            'hail': None,
            'prob_maps': {},
            'discussion_url': None,
            'discussion_text': None
        }
        
        # Extract image URLs for probability maps
        img_tags = soup.find_all('img')
        for img in img_tags:
            src = img.get('src', '')
            if 'probabilistic' in src.lower() or 'categorical' in src.lower():
                if 'torn' in src.lower():
                    outlook_data['tornado'] = f"{self.base_url}{src}"
                    outlook_data['prob_maps']['tornado'] = f"{self.base_url}{src}"
                elif 'wind' in src.lower():
                    outlook_data['wind'] = f"{self.base_url}{src}"
                    outlook_data['prob_maps']['wind'] = f"{self.base_url}{src}"
                elif 'hail' in src.lower():
                    outlook_data['hail'] = f"{self.base_url}{src}"
                    outlook_data['prob_maps']['hail'] = f"{self.base_url}{src}"
                elif 'cat' in src.lower():
                    outlook_data['categorical'] = f"{self.base_url}{src}"
                    outlook_data['prob_maps']['categorical'] = f"{self.base_url}{src}"
        
        # Get discussion URL
        discussion_links = soup.find_all('a', href=re.compile(r'day\d_discussion'))
        if discussion_links:
            discussion_url = discussion_links[0].get('href')
            if not discussion_url.startswith('http'):
                discussion_url = f"{self.outlook_url}/{discussion_url}"
            outlook_data['discussion_url'] = discussion_url
            
            # Fetch discussion text
            discussion_html = self._fetch_url(discussion_url)
            if discussion_html:
                discussion_soup = BeautifulSoup(discussion_html, 'html.parser')
                pre_tags = discussion_soup.find_all('pre')
                if pre_tags:
                    outlook_data['discussion_text'] = pre_tags[0].get_text()
        
        # Extract valid time information
        time_pattern = re.compile(r'Valid (\d{6}Z) - (\d{6}Z)')
        text = soup.get_text()
        time_match = time_pattern.search(text)
        if time_match:
            outlook_data['valid_time'] = f"{time_match.group(1)} - {time_match.group(2)}"
        
        # Save to cache
        if cache:
            try:
                with open(cache_file, 'w') as f:
                    json.dump(outlook_data, f)
            except Exception as e:
                logger.warning(f"Error writing cache file: {e}")
        
        return outlook_data
    
    def get_severe_weather_reports(self, date=None, report_type='all', cache=True):
        """
        Get severe weather reports for a specific date
        
        Args:
            date (str): Date in YYYYMMDD format, defaults to yesterday
            report_type (str): Type of report ('all', 'tornado', 'wind', 'hail')
            cache (bool): Whether to use cached data if available
            
        Returns:
            pd.DataFrame: DataFrame containing severe weather reports
        """
        # Default to yesterday if no date provided
        if date is None:
            yesterday = datetime.now() - timedelta(days=1)
            date = yesterday.strftime('%Y%m%d')
        
        cache_file = os.path.join(self.cache_dir, f"severe_reports_{date}_{report_type}.csv")
        
        # Check cache first if enabled
        if cache and os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file)
                logger.info(f"Using cached severe weather reports for {date}")
                return df
            except Exception as e:
                logger.warning(f"Error reading cache file: {e}")
        
        # Map report type to SPC code
        type_map = {
            'all': 'all',
            'tornado': 'torn',
            'wind': 'wind',
            'hail': 'hail'
        }
        
        report_code = type_map.get(report_type.lower(), 'all')
        
        # Construct URL
        url = f"{self.reports_url}/{date.split('-')[0]}/{date}_rpts_{report_code}.csv"
        
        # Fetch the data
        try:
            df = pd.read_csv(url)
            
            # Save to cache
            if cache:
                try:
                    df.to_csv(cache_file, index=False)
                except Exception as e:
                    logger.warning(f"Error writing cache file: {e}")
            
            return df
        except Exception as e:
            logger.error(f"Error fetching severe weather reports: {e}")
            return pd.DataFrame()
    
    def get_mesoscale_discussions(self, start_date=None, end_date=None, cache=True):
        """
        Get mesoscale discussions for a date range
        
        Args:
            start_date (str): Start date in YYYYMMDD format, defaults to today
            end_date (str): End date in YYYYMMDD format, defaults to today
            cache (bool): Whether to use cached data if available
            
        Returns:
            list: List of mesoscale discussion data
        """
        # Default to today if no dates provided
        if start_date is None:
            start_date = datetime.now().strftime('%Y%m%d')
        
        if end_date is None:
            end_date = start_date
        
        cache_file = os.path.join(self.cache_dir, f"md_{start_date}_{end_date}.json")
        
        # Check cache first if enabled
        if cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    if (datetime.now() - datetime.strptime(data['fetch_time'], "%Y-%m-%d %H:%M:%S")).total_seconds() < 3600:
                        logger.info(f"Using cached mesoscale discussions")
                        return data['discussions']
            except Exception as e:
                logger.warning(f"Error reading cache file: {e}")
        
        # Convert dates to datetime objects
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        
        all_discussions = []
        
        # Iterate through each date in the range
        current_dt = start_dt
        while current_dt <= end_dt:
            current_date = current_dt.strftime('%Y%m%d')
            
            # Construct URL for the date
            url = f"{self.md_url}/md_{current_date}.html"
            
            html = self._fetch_url(url)
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find all links to individual mesoscale discussions
                md_links = soup.find_all('a', href=re.compile(r'md\d{4}.html'))
                
                for link in md_links:
                    md_url = link.get('href')
                    if not md_url.startswith('http'):
                        md_url = f"{self.md_url}/{md_url}"
                    
                    md_html = self._fetch_url(md_url)
                    if md_html:
                        md_soup = BeautifulSoup(md_html, 'html.parser')
                        
                        # Extract discussion text
                        pre_tags = md_soup.find_all('pre')
                        text = pre_tags[0].get_text() if pre_tags else ""
                        
                        # Extract image URL if available
                        img_tags = md_soup.find_all('img')
                        img_url = None
                        for img in img_tags:
                            src = img.get('src', '')
                            if 'md' in src.lower() and not 'noimage' in src.lower():
                                img_url = f"{self.md_url}/{src}" if not src.startswith('http') else src
                                break
                        
                        discussion_data = {
                            'url': md_url,
                            'id': os.path.basename(md_url).replace('.html', ''),
                            'date': current_date,
                            'text': text,
                            'image_url': img_url
                        }
                        
                        all_discussions.append(discussion_data)
            
            current_dt += timedelta(days=1)
        
        # Save to cache
        if cache:
            try:
                cache_data = {
                    'fetch_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'discussions': all_discussions
                }
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f)
            except Exception as e:
                logger.warning(f"Error writing cache file: {e}")
        
        return all_discussions
    
    def get_watches(self, date=None, cache=True):
        """
        Get severe weather watches for a specific date
        
        Args:
            date (str): Date in YYYYMMDD format, defaults to today
            cache (bool): Whether to use cached data if available
            
        Returns:
            list: List of watch data
        """
        # Default to today if no date provided
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        
        cache_file = os.path.join(self.cache_dir, f"watches_{date}.json")
        
        # Check cache first if enabled
        if cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    if (datetime.now() - datetime.strptime(data['fetch_time'], "%Y-%m-%d %H:%M:%S")).total_seconds() < 3600:
                        logger.info(f"Using cached watches for {date}")
                        return data['watches']
            except Exception as e:
                logger.warning(f"Error reading cache file: {e}")
        
        # Construct URL for the date
        url = f"{self.watch_url}/watch_{date}.html"
        
        html = self._fetch_url(url)
        if not html:
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find all links to individual watches
        watch_links = soup.find_all('a', href=re.compile(r'ww\d{4}.html'))
        
        all_watches = []
        
        for link in watch_links:
            watch_url = link.get('href')
            if not watch_url.startswith('http'):
                watch_url = f"{self.watch_url}/{watch_url}"
            
            watch_html = self._fetch_url(watch_url)
            if watch_html:
                watch_soup = BeautifulSoup(watch_html, 'html.parser')
                
                # Extract watch text
                pre_tags = watch_soup.find_all('pre')
                text = pre_tags[0].get_text() if pre_tags else ""
                
                # Extract image URL if available
                img_tags = watch_soup.find_all('img')
                img_url = None
                for img in img_tags:
                    src = img.get('src', '')
                    if 'wwatlas' in src.lower():
                        img_url = f"{self.watch_url}/{src}" if not src.startswith('http') else src
                        break
                
                # Parse watch type from text
                watch_type = "Unknown"
                if "TORNADO WATCH" in text:
                    watch_type = "Tornado"
                elif "SEVERE THUNDERSTORM WATCH" in text:
                    watch_type = "Severe Thunderstorm"
                
                watch_data = {
                    'url': watch_url,
                    'id': os.path.basename(watch_url).replace('.html', ''),
                    'date': date,
                    'type': watch_type,
                    'text': text,
                    'image_url': img_url
                }
                
                all_watches.append(watch_data)
        
        # Save to cache
        if cache:
            try:
                cache_data = {
                    'fetch_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'watches': all_watches
                }
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f)
            except Exception as e:
                logger.warning(f"Error writing cache file: {e}")
        
        return all_watches 