FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY backend backend
COPY ml ml
COPY frontend/dist frontend/dist
COPY samples samples
# HF Spaces runs the container unprivileged; only /tmp is writable.
ENV MODEL_DIR=ml/models SAMPLES_DIR=samples FRONTEND_DIST=frontend/dist \
    STORAGE_DIR=/tmp/storage DATABASE_URL=sqlite:////tmp/knee.db \
    TRAIN_CONFIG=ml/training/config.yaml
EXPOSE 7860
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
