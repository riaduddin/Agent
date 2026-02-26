#!/bin/bash



# Activate virtual environment
source venv/bin/activate

# Set Google application credentials
export GOOGLE_APPLICATION_CREDENTIALS=./key.json

# Run the worker script
# python worker.py
python run.py

# export GOOGLE_APPLICATION_CREDENTIALS=./key.json && python run.py