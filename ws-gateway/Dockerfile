FROM python:3.11-slim

WORKDIR /app

# Add /app to PYTHONPATH for imports
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
#COPY src/ ./src/
#COPY migrations/ ./migrations/
#COPY tests/ ./tests/
#COPY compare_balances.py ./

# Expose port (non-standard, configured via env)
EXPOSE 4400

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${WS_GATEWAY_PORT:-4400}/health || exit 1

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "4400"]

