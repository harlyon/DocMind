FROM python:3.11-slim

WORKDIR /app

# System deps for building Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Use --only-binary=:all: to avoid source compilation issues
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt \
    || pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directories
RUN mkdir -p data/uploads data/chroma

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]