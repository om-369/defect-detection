@echo off

:: Set environment variables
set FLASK_APP=src/defect_detection/app.py
set FLASK_ENV=development
set FLASK_DEBUG=1

:: Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

:: Install requirements if needed
if exist requirements.txt (
    pip install -r requirements.txt
)

:: Run the Flask application
python -m flask run --host=0.0.0.0 --port=5000

pause
