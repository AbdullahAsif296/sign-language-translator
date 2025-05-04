#!/bin/bash

# Start the backend server
echo "Starting backend server..."
cd backend
python src/api.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Start the frontend server
echo "Starting frontend server..."
cd ../frontend
npm start &
FRONTEND_PID=$!

# Function to handle cleanup
cleanup() {
    echo "Stopping servers..."
    kill $BACKEND_PID
    kill $FRONTEND_PID
    exit
}

# Trap Ctrl+C and call cleanup
trap cleanup SIGINT

# Keep the script running
wait 