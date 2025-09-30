@echo off
echo Starting Parkinson's Detection PWA...
echo.

echo [1/2] Starting Frontend Server...
cd frontend
start "Frontend Server" cmd /k "python -m http.server 8080 || npx serve -s . -l 8080"
echo Frontend will be available at: http://localhost:8080
echo.

echo [2/2] Starting Backend Server...
cd ..\backend
start "Backend Server" cmd /k "python app.py"
echo Backend will be available at: http://localhost:5000
echo.

echo Both servers are starting in separate windows.
echo Open http://localhost:8080 in your browser to use the app.
echo.
echo Press any key to exit this window...
pause >nul