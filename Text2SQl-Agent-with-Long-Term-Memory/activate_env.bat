@echo off
echo Activating Text2SQL Agent virtual environment...
call .venv\Scripts\activate.bat
echo Virtual environment activated!
echo.
echo To run the memory agent: python memory_agent.py
echo To run the frontend: python memory_frontend.py
echo.
cmd /k 