#!/usr/bin/env python3
"""
Weather Prediction System Setup
==============================
Script to set up the environment for the weather prediction system,
install dependencies using uv pip, and optimize for CPU performance.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define required packages
REQUIRED_PACKAGES = [
    "numpy==1.24.3",
    "pandas==2.0.3",
    "matplotlib==3.7.2",
    "scikit-learn==1.3.0",
    "xgboost==1.7.6",
    "pygrib==2.1.4",
    "tensorflow-cpu==2.13.0",  # CPU-specific TensorFlow
    "pillow==10.0.0",
    "cartopy==0.21.1",
    "seaborn==0.12.2",
    "joblib==1.3.1",
    "tqdm==4.65.0",
    "geopandas==0.14.0",
    "scipy==1.11.3",
    "requests==2.31.0",
    "netCDF4==1.6.4",
    "s3fs==2023.9.2",
    "dask==2023.9.3",
    "metpy==1.5.1",
    "pyproj==3.6.1",
    "h5py==3.10.0",
    "statsmodels==0.14.0",
    "opencv-python-headless==4.8.1.78",
]

# Define extra data sources
DATA_SOURCES = {
    "us_population": "https://www2.census.gov/geo/tiger/TIGER2020/COUNTY/tl_2020_us_county.zip",
    "noaa_indices": "https://www.ncei.noaa.gov/pub/data/cirs/climdiv/climdiv-pcpndv-v1.0.0-20230805",
    "global_temp": "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
}

def check_system():
    """Check system configuration and requirements."""
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Operating system: {platform.platform()}")
    
    # Check CPU info
    if platform.system() == "Linux":
        try:
            cpu_info = subprocess.check_output("lscpu", shell=True).decode("utf-8")
            logger.info(f"CPU information:\n{cpu_info}")
        except subprocess.SubprocessError:
            logger.warning("Could not retrieve CPU information")
    
    # Check memory
    try:
        total_memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        total_memory_gb = total_memory / (1024.**3)
        logger.info(f"Total system memory: {total_memory_gb:.2f} GB")
        
        if total_memory_gb < 4:
            logger.warning("Low memory detected. Performance may be affected.")
    except:
        logger.warning("Could not determine system memory")
    
    # Create required directories
    for directory in ["data", "models", "visualizations"]:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def install_uv():
    """Install uv package manager if not already installed."""
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        logger.info("uv is already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.info("Installing uv package manager")
        
        try:
            # Different installation methods based on platform
            if platform.system() == "Linux" or platform.system() == "Darwin":
                subprocess.run("curl -fsSL https://astral.sh/uv/install.sh | sh", shell=True, check=True)
            elif platform.system() == "Windows":
                subprocess.run("powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"", shell=True, check=True)
            else:
                logger.error(f"Unsupported platform for uv installation: {platform.system()}")
                return False
            
            logger.info("uv installed successfully")
            return True
        except subprocess.SubprocessError as e:
            logger.error(f"Failed to install uv: {e}")
            return False
    
    return True

def install_dependencies():
    """Install all required dependencies using uv pip."""
    logger.info("Installing dependencies with uv pip")
    
    # Create requirements.txt
    with open("requirements.txt", "w") as f:
        f.write("\n".join(REQUIRED_PACKAGES))
    
    try:
        # Install dependencies with uv pip
        subprocess.run(["uv", "pip", "install", "-r", "requirements.txt"], check=True)
        logger.info("Dependencies installed successfully using uv pip")
        return True
    except subprocess.SubprocessError as e:
        logger.error(f"Error installing dependencies with uv pip: {e}")
        logger.info("Falling back to regular pip")
        
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            logger.info("Dependencies installed successfully using regular pip")
            return True
        except subprocess.SubprocessError as e2:
            logger.error(f"Error installing dependencies with pip: {e2}")
            return False

def download_additional_data():
    """Download additional data sources for improved accuracy."""
    data_dir = Path("data")
    extra_data_dir = data_dir / "extra"
    extra_data_dir.mkdir(exist_ok=True)
    
    logger.info(f"Downloading additional data sources to {extra_data_dir}")
    
    for name, url in DATA_SOURCES.items():
        target_file = extra_data_dir / f"{name}.dat"
        if not target_file.exists():
            try:
                logger.info(f"Downloading {name} data from {url}")
                subprocess.run(["curl", "-L", "-o", str(target_file), url], check=True)
                logger.info(f"Successfully downloaded {name} data")
            except subprocess.SubprocessError as e:
                logger.error(f"Failed to download {name} data: {e}")

def configure_cpu_optimizations():
    """Configure system for CPU-optimized processing."""
    # Create TensorFlow configuration file
    tf_config = """
# TensorFlow CPU configuration
import tensorflow as tf
import os

# Set TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Configure threading
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Enable CPU optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPU_TRANSFER_GUARD'] = '0'

# Set memory growth
physical_devices = tf.config.list_physical_devices()
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

print("TensorFlow CPU optimizations configured")
"""
    
    with open("tf_cpu_config.py", "w") as f:
        f.write(tf_config)
    
    logger.info("Created TensorFlow CPU configuration file")
    
    # Create .env file with optimization environment variables
    env_vars = """
# Environment variables for CPU optimization
OPENBLAS_NUM_THREADS=4
MKL_NUM_THREADS=4
NUMEXPR_NUM_THREADS=4
OMP_NUM_THREADS=4
VECLIB_MAXIMUM_THREADS=4
JOBLIB_DEFAULT_N_JOBS=4
"""
    
    with open(".env", "w") as f:
        f.write(env_vars)
    
    logger.info("Created environment configuration file")

def main():
    """Main setup function."""
    logger.info("Starting Weather Prediction System setup")
    
    # Check system requirements
    check_system()
    
    # Install uv package manager
    if not install_uv():
        logger.warning("Continuing setup without uv")
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Failed to install dependencies. Setup incomplete.")
        return False
    
    # Download additional data
    download_additional_data()
    
    # Configure CPU optimizations
    configure_cpu_optimizations()
    
    logger.info("Weather Prediction System setup completed successfully")
    logger.info("Run 'python __init__.py' to start the system")
    
    return True

if __name__ == "__main__":
    main() 