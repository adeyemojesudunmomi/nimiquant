@echo off
cd /d "C:\Users\Mrs Dickson\Desktop\NimiQuant-Crypto intelligence Engine"
echo Running NimiQuant Daily Update...
py data.py
py scanner_signals.py
pause