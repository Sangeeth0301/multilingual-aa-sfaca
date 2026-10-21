@echo off
rem Double-click to share the app: starts the model API and a free tunnel, then publishes the URL.
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File scripts\start_public.ps1
pause
