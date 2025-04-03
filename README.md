# Weather Prediction Bot

An advanced machine learning-based weather prediction bot that utilizes historical forecast data to predict weather conditions with high accuracy. The bot specializes in hail prediction based on GRIB2 meteorological data.

## Features

- **Data Processing**: Loads and processes GRIB2 weather forecast data
- **Multiple ML Models**: Trains and compares various models (Random Forest, Gradient Boosting, XGBoost, Neural Networks)
- **High Accuracy**: Selects the best performing model for predictions
- **Visualizations**: Generates beautiful visualizations of forecasts and model performance
- **Time Series Analysis**: Shows prediction trends over time
- **Model Comparison**: Compares different model predictions
- **Custom Forecast Images**: Creates professional forecast images
- **SPC Integration**: Fetches and integrates data from NOAA's Storm Prediction Center for verification and enhanced forecasting
- **Location-based Forecasts**: Generate risk reports for specific geographic locations with SPC data integration
- **Severe Weather Reporting**: Incorporates recent severe weather reports into the forecasting system

## Project Structure

```
weather_prediction_bot/
├── data/                   # Processed data storage
│   └── spc_cache/          # Cached SPC data
├── models/                 # Trained models storage
├── visualizations/         # Generated visualizations
├── src/
│   ├── data_loader.py      # Module for loading GRIB2 weather data
│   ├── data_processor.py   # Data preprocessing and feature extraction
│   ├── model_trainer.py    # ML model training and evaluation
│   ├── visualizer.py       # Visualization generation
│   ├── weather_bot.py      # Main bot functionality
│   ├── spc_data_fetcher.py # Module for fetching data from the SPC
│   └── spc_integration.py  # Integration of SPC data with predictions
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd weather_prediction_bot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have the required data in the appropriate directory or update the data path in the configuration.

## Usage

### Command Line Interface

Run the bot with default settings:

```bash
python run_weather_system.py
```

Customize with arguments:

```bash
python run_weather_system.py --data-dir /path/to/data --models-dir /path/to/models --visualizations-dir /path/to/visualizations --start-date 20230101 --end-date 20230131 --feature-type all --epochs 20 --batch-size 32 --advanced-features
```

Using SPC Integration:

```bash
# Generate a forecast with SPC verification
python run_weather_system.py --forecast 20240401 --use-spc --outlook-day 1

# Generate a location-specific forecast
python run_weather_system.py --location 35.2220,-97.4395 --use-spc

# Only fetch SPC data (useful for verification)
python run_weather_system.py --fetch-spc-only --outlook-day 1
```

### API Usage

You can also use the bot programmatically in your Python code:

```python
from src.weather_bot import WeatherPredictionBot

# Initialize the bot
bot = WeatherPredictionBot(
    data_dir='/path/to/data',
    output_dir='/path/to/output'
)

# Run the full pipeline
results = bot.run_pipeline(
    start_date='20230101',
    end_date='20230131',
    data_limit=100,
    forecast_horizon=24,
    task='regression',
    target_col='target_max_hail'
)

# Access results
if results['status'] == 'success':
    print(f"Best model: {results['best_model']}")
    print(f"Created {len(results['visualizations'])} visualizations")
```

## Data Format

The bot expects weather forecast data in GRIB2 format organized in a specific directory structure:

```
/data_dir/
├── YYYYMM/
│   ├── YYYYMMDD/
│   │   ├── t0z/
│   │   │   ├── weather_data_file.grib2
│   │   │   ├── weather_data_file.png
│   │   │   └── ...
│   │   ├── t12z/
│   │   └── ...
│   └── ...
└── ...
```

## Models Used

The bot trains and evaluates several machine learning models:

1. **Random Forest**: Good for handling non-linear relationships
2. **Gradient Boosting**: High performance for structured data
3. **XGBoost**: Advanced implementation of gradient boosting
4. **Neural Network**: Deep learning approach with multiple layers

## Visualizations

The bot generates several types of visualizations:

1. **Model Performance Metrics**: Comparison of model accuracy, precision, etc.
2. **Time Series Predictions**: Shows predictions over time with comparison to actual values
3. **Feature Importance**: Visual representation of the most important weather parameters
4. **Forecast Images**: Professional weather forecast images for specific dates

## SPC Integration

The new SPC (Storm Prediction Center) integration allows you to:

1. **Fetch Real-time SPC Data**: Access the latest convective outlooks, mesoscale discussions, and severe weather reports
2. **Compare Model Predictions**: Compare your model's predictions with official SPC outlooks
3. **Generate Location Reports**: Create detailed risk reports for specific locations
4. **Visualize Verification**: Create side-by-side visualizations of model predictions vs. SPC outlooks
5. **Access Historical Data**: Retrieve and analyze historical severe weather reports

### SPC Data Types

The system can fetch and integrate the following SPC data:

- **Convective Outlooks**: Day 1-8 categorical and probabilistic outlooks
- **Mesoscale Discussions**: Detailed meteorological discussions for areas of severe weather potential
- **Watches**: Tornado and severe thunderstorm watches
- **Severe Weather Reports**: Recent reports of tornadoes, severe wind, and hail

### Location-Based Risk Reports

Generate comprehensive risk reports for specific locations that include:

- Location information (state, county)
- Model prediction risk level
- Recent severe weather reports nearby
- Current SPC outlook category
- Links to official SPC discussions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 