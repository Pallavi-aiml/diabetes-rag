FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN mkdir -p data/raw data/processed data/chroma_db

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -fs http://localhost:8000/health || exit 1

# Default = API. docker-compose overrides the command for the Streamlit frontend.
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
