@echo off
setlocal enabledelayedexpansion

echo Weather Prediction System Runner
echo ===============================
echo.

:: Set colors for console output
color 0B

:: Set default values
set DATA_DIR=data
set MODELS_DIR=models
set VIS_DIR=visualizations
set FEATURE_TYPE=all
set EPOCHS=20
set BATCH_SIZE=32
set OUTLOOK_DAY=1

:: Check for Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found. Please install Python and try again.
    exit /b 1
)

:: Check if requirements are installed
echo Checking dependencies...
pip show beautifulsoup4 >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to install dependencies.
        exit /b 1
    )
)

:: Create necessary directories
if not exist %DATA_DIR% mkdir %DATA_DIR%
if not exist %MODELS_DIR% mkdir %MODELS_DIR%
if not exist %VIS_DIR% mkdir %VIS_DIR%
if not exist %DATA_DIR%\spc_cache mkdir %DATA_DIR%\spc_cache

:: Get current date in YYYYMMDD format
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (
    set MM=%%a
    set DD=%%b
    set YYYY=%%c
)
set TODAY=%YYYY%%MM%%DD%

echo.
echo Choose an option:
echo 1. Train models with default settings
echo 2. Train models with advanced features
echo 3. Generate forecast for today
echo 4. Generate forecast with SPC verification
echo 5. Generate location-specific forecast
echo 6. Fetch SPC data only
echo 7. Run full system with SPC integration
echo 8. Evaluate existing models
echo 9. Train models with SPC features (enhanced)
echo 10. Custom command
echo 0. Exit
echo.

set /p OPTION="Enter option (0-10): "

if "%OPTION%"=="1" (
    echo Running model training with default settings...
    python run_weather_system.py --data-dir %DATA_DIR% --models-dir %MODELS_DIR% --visualizations-dir %VIS_DIR%
    
) else if "%OPTION%"=="2" (
    echo Running model training with advanced features...
    python run_weather_system.py --data-dir %DATA_DIR% --models-dir %MODELS_DIR% --visualizations-dir %VIS_DIR% --advanced-features --epochs 30 --batch-size 64
    
) else if "%OPTION%"=="3" (
    echo Generating forecast for today...
    python run_weather_system.py --forecast %TODAY% --data-dir %DATA_DIR% --models-dir %MODELS_DIR% --visualizations-dir %VIS_DIR%
    
) else if "%OPTION%"=="4" (
    set /p FORECAST_DATE="Enter forecast date (YYYYMMDD, default: today): "
    if "!FORECAST_DATE!"=="" set FORECAST_DATE=%TODAY%
    
    set /p OUTLOOK_DAY="Enter SPC outlook day (1-8, default: 1): "
    if "!OUTLOOK_DAY!"=="" set OUTLOOK_DAY=1
    
    echo Generating forecast with SPC verification...
    python run_weather_system.py --forecast !FORECAST_DATE! --use-spc --outlook-day !OUTLOOK_DAY! --data-dir %DATA_DIR% --models-dir %MODELS_DIR% --visualizations-dir %VIS_DIR%
    
) else if "%OPTION%"=="5" (
    set /p LOCATION="Enter location as lat,lon (e.g., 35.2220,-97.4395): "
    
    if "!LOCATION!"=="" (
        echo ERROR: Location is required.
        exit /b 1
    )
    
    echo Generating location-specific forecast...
    python run_weather_system.py --location !LOCATION! --use-spc --data-dir %DATA_DIR% --models-dir %MODELS_DIR% --visualizations-dir %VIS_DIR%
    
) else if "%OPTION%"=="6" (
    set /p OUTLOOK_DAY="Enter SPC outlook day (1-8, default: 1): "
    if "!OUTLOOK_DAY!"=="" set OUTLOOK_DAY=1
    
    echo Fetching SPC data only...
    python run_weather_system.py --fetch-spc-only --outlook-day !OUTLOOK_DAY! --spc-cache-dir %DATA_DIR%\spc_cache
    
) else if "%OPTION%"=="7" (
    echo Running full system with SPC integration...
    python run_weather_system.py --data-dir %DATA_DIR% --models-dir %MODELS_DIR% --visualizations-dir %VIS_DIR% --use-spc --advanced-features
    
) else if "%OPTION%"=="8" (
    echo Evaluating existing models...
    python run_weather_system.py --eval-only --use-spc --data-dir %DATA_DIR% --models-dir %MODELS_DIR% --visualizations-dir %VIS_DIR%
    
) else if "%OPTION%"=="9" (
    echo Training models with SPC features (RECOMMENDED FOR BEST RESULTS)...
    
    set /p START_DATE="Enter start date for training (YYYYMMDD): "
    set /p END_DATE="Enter end date for training (YYYYMMDD): "
    
    echo This will fetch SPC data for the date range and use it for enhanced model training.
    echo This method produces the most accurate models by combining weather data with SPC data.
    
    python run_weather_system.py --data-dir %DATA_DIR% --models-dir %MODELS_DIR% --visualizations-dir %VIS_DIR% --advanced-features --spc-features --use-spc --start-date !START_DATE! --end-date !END_DATE! --epochs 50 --batch-size 32
    
) else if "%OPTION%"=="10" (
    echo Enter custom command options (without 'python run_weather_system.py'):
    set /p CUSTOM_OPTS=""
    
    echo Running custom command...
    python run_weather_system.py %CUSTOM_OPTS%
    
) else if "%OPTION%"=="0" (
    echo Exiting...
    exit /b 0
    
) else (
    echo Invalid option. Please try again.
    exit /b 1
)

echo.
echo Command completed.
echo Results and visualizations are in the '%VIS_DIR%' directory.

pause 