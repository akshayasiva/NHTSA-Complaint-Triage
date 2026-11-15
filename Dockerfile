FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app_api.py .
COPY models/comp_lr.joblib models/danger_lr.joblib models/config.json models/
EXPOSE 8000
CMD ["uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "8000"]
