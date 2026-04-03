@echo off
cd /d "f:\Project\4_1_9_final"
.venv\Scripts\python.exe generate_paper_figures.py
echo Exit code: %ERRORLEVEL%
pause
