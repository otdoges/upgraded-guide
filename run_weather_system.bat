@echo off
setlocal enabledelayedexpansion

title Weather Prediction System Runner

:start
cls
echo ================================================
echo        Weather Prediction System Runner         
echo ================================================
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
echo Checking for Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo ERROR: Python not found. Please install Python and try again.
    echo Press any key to exit...
    pause >nul
    exit /b 1
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

:menu
cls
echo ================================================
echo        Weather Prediction System Runner         
echo ================================================
echo.
echo Please select an option by typing the number and pressing ENTER:
echo.
echo [1] Train models with default settings
echo [2] Train models with advanced features (safe mode)
echo [3] Generate forecast for today
echo [4] Generate forecast with SPC verification
echo [5] Generate location-specific forecast
echo [6] Fetch SPC data only
echo [7] Run full system with SPC integration
echo [8] Evaluate existing models
echo [9] Train models with SPC features (RECOMMENDED)
echo [0] Exit
echo.
echo Type a number (0-9) and press ENTER: 

:: Use choice command to capture a single key press
set OPTION=
set /p OPTION=">"

if "%OPTION%"=="" goto menu

:: Validate input
set "valid="
for %%v in (0 1 2 3 4 5 6 7 8 9) do if "%OPTION%"=="%%v" set "valid=1"
if not defined valid (
    echo.
    echo Invalid option. Please enter a number between 0 and 9.
    echo.
    echo Press any key to continue...
    pause >nul
    goto menu
)

:: Process selected option
if "%OPTION%"=="0" goto exit_program

:: First check dependencies if needed
if "%OPTION%"=="1" goto check_deps
if "%OPTION%"=="2" goto check_deps
if "%OPTION%"=="7" goto check_deps
if "%OPTION%"=="9" goto check_deps

:: Otherwise go directly to the appropriate section
goto option_%OPTION%

:check_deps
:: Check if requirements are installed
echo.
echo Checking dependencies...
pip show beautifulsoup4 >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Installing dependencies...
    pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        color 0C
        echo ERROR: Failed to install dependencies.
        echo.
        echo Press any key to return to menu...
        pause >nul
        goto menu
    )
)
goto option_%OPTION%

:option_1
cls
echo ================================================
echo          Train models with default settings
echo ================================================
echo.
echo This will train the model with default settings.
echo.
echo Press ENTER to start training or ESC to return to menu.
echo.
choice /c EC /n /m "Press E to start or C to cancel: "
if %ERRORLEVEL%==2 goto menu

echo.
echo Starting training with default settings...
echo.
python run_weather_system.py --data-dir %DATA_DIR% --models-dir %MODELS_DIR% --visualizations-dir %VIS_DIR%
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo.
    echo ERROR: Command failed with exit code %ERRORLEVEL%
    color 0B
)
goto command_finished

:option_2
cls
echo ================================================
echo      Train models with advanced features
echo ================================================
echo.
echo This will train models with advanced features using a safe batch size.
echo.
choice /c EC /n /m "Press E to start or C to cancel: "
if %ERRORLEVEL%==2 goto menu

echo.
echo Starting training with advanced features...
echo.
python run_weather_system.py --data-dir %DATA_DIR% --models-dir %MODELS_DIR% --visualizations-dir %VIS_DIR% --advanced-features --epochs 30 --batch-size 32
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo.
    echo ERROR: Command failed with exit code %ERRORLEVEL%
    color 0B
)
goto command_finished

:option_3
cls
echo ================================================
echo           Generate forecast for today
echo ================================================
echo.
echo This will generate a forecast for today (%TODAY%).
echo.
choice /c EC /n /m "Press E to start or C to cancel: "
if %ERRORLEVEL%==2 goto menu

echo.
echo Generating forecast for today...
echo.
python run_weather_system.py --forecast %TODAY% --data-dir %DATA_DIR% --models-dir %MODELS_DIR% --visualizations-dir %VIS_DIR%
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo.
    echo ERROR: Command failed with exit code %ERRORLEVEL%
    color 0B
)
goto command_finished

:option_4
cls
echo ================================================
echo      Generate forecast with SPC verification
echo ================================================
echo.
set FORECAST_DATE=
set /p FORECAST_DATE="Enter forecast date (YYYYMMDD, default: today): "
if "!FORECAST_DATE!"=="" set FORECAST_DATE=%TODAY%

set OUTLOOK_DAY=
set /p OUTLOOK_DAY="Enter SPC outlook day (1-8, default: 1): "
if "!OUTLOOK_DAY!"=="" set OUTLOOK_DAY=1
echo.
echo Will generate forecast for !FORECAST_DATE! with SPC outlook day !OUTLOOK_DAY!
echo.
choice /c EC /n /m "Press E to start or C to cancel: "
if %ERRORLEVEL%==2 goto menu

echo.
echo Generating forecast with SPC verification...
echo.
python run_weather_system.py --forecast !FORECAST_DATE! --use-spc --outlook-day !OUTLOOK_DAY! --data-dir %DATA_DIR% --models-dir %MODELS_DIR% --visualizations-dir %VIS_DIR%
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo.
    echo ERROR: Command failed with exit code %ERRORLEVEL%
    color 0B
)
goto command_finished

:option_5
cls
echo ================================================
echo       Generate location-specific forecast
echo ================================================
echo.
set LOCATION=
set /p LOCATION="Enter location as lat,lon (e.g., 35.2220,-97.4395): "

if "!LOCATION!"=="" (
    color 0C
    echo.
    echo ERROR: Location is required.
    color 0B
    echo.
    echo Press any key to return to menu...
    pause >nul
    goto menu
)

echo.
echo Will generate forecast for location: !LOCATION!
echo.
choice /c EC /n /m "Press E to start or C to cancel: "
if %ERRORLEVEL%==2 goto menu

echo.
echo Generating location-specific forecast...
echo.
python run_weather_system.py --location !LOCATION! --use-spc --data-dir %DATA_DIR% --models-dir %MODELS_DIR% --visualizations-dir %VIS_DIR%
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo.
    echo ERROR: Command failed with exit code %ERRORLEVEL%
    color 0B
)
goto command_finished

:option_6
cls
echo ================================================
echo               Fetch SPC data only
echo ================================================
echo.
echo This will automatically fetch today's SPC data.
echo The data will be used for forecasting and verification.
echo.
choice /c EC /n /m "Press E to start or C to cancel: "
if %ERRORLEVEL%==2 goto menu

echo.
echo Fetching today's SPC data...
echo.
python run_weather_system.py --fetch-spc-only --outlook-day 1 --spc-cache-dir %DATA_DIR%\spc_cache
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo.
    echo ERROR: Command failed with exit code %ERRORLEVEL%
    color 0B
)
goto command_finished

:option_7
cls
echo ================================================
echo      Run full system with SPC integration
echo ================================================
echo.
echo This will run the full system with SPC integration.
echo.
choice /c EC /n /m "Press E to start or C to cancel: "
if %ERRORLEVEL%==2 goto menu

echo.
echo Running full system with SPC integration...
echo.
python run_weather_system.py --data-dir %DATA_DIR% --models-dir %MODELS_DIR% --visualizations-dir %VIS_DIR% --use-spc --advanced-features
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo.
    echo ERROR: Command failed with exit code %ERRORLEVEL%
    color 0B
)
goto command_finished

:option_8
cls
echo ================================================
echo            Evaluate existing models
echo ================================================
echo.
echo This will evaluate the existing trained models.
echo.
choice /c EC /n /m "Press E to start or C to cancel: "
if %ERRORLEVEL%==2 goto menu

echo.
echo Evaluating existing models...
echo.
python run_weather_system.py --eval-only --use-spc --data-dir %DATA_DIR% --models-dir %MODELS_DIR% --visualizations-dir %VIS_DIR%
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo.
    echo ERROR: Command failed with exit code %ERRORLEVEL%
    color 0B
)
goto command_finished

:option_9
cls
echo ================================================
echo    Train models with SPC features (ENHANCED)
echo ================================================
echo.
echo This option produces the most accurate models by combining weather data 
echo with Storm Prediction Center (SPC) data for enhanced training.
echo.
echo The system will automatically fetch all available SPC data and use it
echo for training without requiring specific date ranges.
echo.
choice /c EC /n /m "Press E to start or C to cancel: "
if %ERRORLEVEL%==2 goto menu

echo.
echo Training models with SPC features...
echo This may take a while. Please be patient.
echo Fetching all available SPC data...
echo.
python run_weather_system.py --data-dir %DATA_DIR% --models-dir %MODELS_DIR% --visualizations-dir %VIS_DIR% --advanced-features --spc-features --use-spc --epochs 50 --batch-size 32
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo.
    echo ERROR: Command failed with exit code %ERRORLEVEL%
    color 0B
)
goto command_finished

:command_finished
echo.
echo ================================================
echo.
echo Command execution completed.
echo Results and visualizations are saved in the '%VIS_DIR%' directory.
echo.
echo ================================================
echo.
echo Press any key to return to the main menu...
pause >nul
goto menu

:exit_program
cls
echo.
echo Thank you for using the Weather Prediction System!
echo.
echo Press any key to exit...
pause >nul
exit /b 0 