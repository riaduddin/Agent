# Use Python slim image (3.11) for compatibility with MCP and better perf
FROM python:3.11-slim-bullseye

# Configure apt to retry on network failures
RUN echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::http::Timeout "20";' >> /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::ftp::Timeout "20";' >> /etc/apt/apt.conf.d/80-retries

# Install Node.js (v20) and system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    gcc \
    g++ \
    libc6-dev \
    # Chromium dependencies for Playwright
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    libatspi2.0-0 \
    libxshmfence1 \
    fonts-liberation \
    libappindicator3-1 \
    xdg-utils && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    npm install -g npm@latest && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Set Playwright path and environment variables for production
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create a non-root user and prepare Playwright directory
RUN groupadd -r appuser && useradd -r -m -g appuser appuser && \
    mkdir -p /ms-playwright && \
    chown -R appuser:appuser /ms-playwright && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Install Playwright browsers (Chromium) into the specific path as appuser
RUN playwright install chromium && \
    (playwright install-deps chromium || echo "Skipping deps")

# Copy application code (will be owned by appuser due to switch)
COPY --chown=appuser:appuser . .

# Install conversion service dependencies
WORKDIR /app/conversion_service
RUN npm install && \
    npx puppeteer browsers install chrome
WORKDIR /app

# Default port for the application
ENV PORT=8010

# IMPORTANT: Cloud Run uses PORT environment variable
EXPOSE 8010

# Use uvicorn directly for Cloud Run
CMD exec uvicorn main:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1 \
    --timeout-keep-alive 65 \
    --ws-ping-interval 20 \
    --ws-ping-timeout 60 \
    --log-level info \
    --access-log \
    --use-colors