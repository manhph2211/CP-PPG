FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git wget curl build-essential libgl1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
CMD ["python", "src/experiments/tools/deploy.py"]
