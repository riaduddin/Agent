source .venv/Scripts/activate
export GOOGLE_APPLICATION_CREDENTIALS=service-account.json
uv run uvicorn main:app --port 8060 --host 0.0.0.0


# # Activate venv if exists and not active
# if [[ -z "$VIRTUAL_ENV" ]]; then
#     if [[ -f "venv/Scripts/activate" ]]; then
#         source venv/Scripts/activate
#     elif [[ -f ".venv/Scripts/activate" ]]; then
#         source .venv/Scripts/activate
#     fi
# fi

# if ! command -v uvicorn &> /dev/null; then
#     echo "Error: uvicorn is not installed. Run 'pip install -r requirements.txt'"
#     exit 1
# fi

# python -m uvicorn main:app --port 8060 --host 0.0.0.0 --reload
# #gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind localhost:8060 --timeout 1200 --graceful-timeout 120 --keep-alive 120