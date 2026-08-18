FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

# The model artifact is not tracked in git, so `python src/train.py` must have
# been run before building. If models/ is empty the image still builds and the
# container still starts, but /health reports unhealthy and returns 503, which
# fails the health check below for the correct reason.
COPY models ./models

# Drop privileges: nothing in the service needs root.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
