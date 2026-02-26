source venv/Scripts/activate
export GOOGLE_APPLICATION_CREDENTIALS=service-account.json
uvicorn main:app --port 8070