#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set environment variables
export FLASK_APP=src/defect_detection/app.py
export FLASK_ENV=development
export FLASK_DEBUG=1

# Install requirements if needed
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Run the Flask application
python -m flask run --host=0.0.0.0 --port=5000
