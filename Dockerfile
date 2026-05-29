FROM python:3.12-slim

WORKDIR /app

# Prevent Python from writing .pyc files to disk inside the container
ENV PYTHONDONTWRITEBYTECODE=1
# Prevent Python from buffering stdout/stderr (good for real-time Docker logs)
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for heavy Python packages (C-extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/uploads data/chroma

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
