# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Hugging Face Spaces sets $PORT automatically
ENV PORT=7860
EXPOSE 7860

CMD ["bash","-lc","gunicorn -w 1 -k gthread --threads 4 -t 120 -b 0.0.0.0:${PORT} app:app"]



