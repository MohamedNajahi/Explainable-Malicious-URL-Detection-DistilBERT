@echo off
cd /d "c:\Users\MSii\Desktop\NIBM\3rd Year Matrerial\Project\PhishingDetection"
"c:\Users\MSii\Desktop\NIBM\3rd Year Matrerial\Project\PhishingDetection\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8765 --timeout-keep-alive 120
