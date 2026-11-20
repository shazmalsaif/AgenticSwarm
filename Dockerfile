FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    HOME=/home/user

WORKDIR /app

# Install Python dependencies first for better layer caching
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/backend/requirements.txt \
    && playwright install --with-deps chromium \
    && chmod -R o+rx /ms-playwright

# Copy application source
COPY backend /app/backend
COPY frontend /app/frontend

# Hugging Face Spaces runs containers as UID 1000
RUN useradd -m -u 1000 user \
    && chown -R user:user /app

USER user

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-7860}"]
