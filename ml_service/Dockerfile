FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_DIR=/app/model

WORKDIR /app

# All dependencies ship as binary wheels, so no compiler toolchain is needed
COPY requirements.txt .
RUN pip install -r requirements.txt

# Run as an unprivileged user
RUN useradd --create-home --uid 1000 appuser
COPY --chown=appuser:appuser . .
USER appuser

# Bake the model into the image. If you COPY a model/ folder trained on real data,
# this step is skipped; otherwise the synthetic demo model is trained at build time.
RUN [ -f model/model.pkl ] || python train.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/health' % os.getenv('PORT','8000'), timeout=3)"

# Render injects $PORT; defaults to 8000 locally
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WEB_CONCURRENCY:-1}"]
