FROM python:3.11-slim


WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ONNX_PATH=deployment/onnx/model.onnx
ENV CONFIDENCE_THRESHOLD=0.65

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY deployment ./deployment

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "deployment.api.main:app", "--host", "0.0.0.0", "--port", "8000"]