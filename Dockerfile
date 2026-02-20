FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (for psycopg2, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy configuration
COPY pyproject.toml .

# Install dependencies
RUN pip install --no-cache-dir .

# Copy source code
COPY . .

# Expose Gradio port
EXPOSE 7860

# Run the application
CMD ["python", "run.py"]
