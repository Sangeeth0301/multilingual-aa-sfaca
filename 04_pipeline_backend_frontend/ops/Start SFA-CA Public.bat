@echo off
rem Double-click to share the app: starts the model API and a free tunnel, then publishes the URL.
cd /d "%~dp0\..\.."
powershell -ExecutionPolicy Bypass -File 04_pipeline_backend_frontend\ops\start_public.ps1
pause
