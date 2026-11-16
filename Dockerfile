FROM python:3.11-slim
WORKDIR /app
ENV MLFLOW_DISABLE_AGENT_HINT=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py ./
COPY model/ ./model/
RUN useradd --create-home api && chown -R api:api /app/model
USER api
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
