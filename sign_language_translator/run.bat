@echo off
echo Starting Sign Language Translator...

:: Start the backend server
echo Starting backend server...
cd backend
start cmd /k "python src/api.py"

:: Wait for backend to start
timeout /t 5

:: Start the frontend server
echo Starting frontend server...
cd ../frontend
start cmd /k "npm start"

echo Both servers are starting...
echo Backend will be available at http://localhost:5000
echo Frontend will be available at http://localhost:3000
echo.
echo Press Ctrl+C in each command window to stop the servers. 