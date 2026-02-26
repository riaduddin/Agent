# Builder stage with a more unique name
FROM python:3.12 AS backend-builder

# Install build dependencies and Python packages via apt-get where possible
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libpython3-dev \
    libxml2-dev \
    libxslt1-dev \
    libxmlsec1-dev \
    libxmlsec1-openssl \
    pkg-config \
    libffi-dev \
    xmlsec1 \
    cargo \
    rustc \
    nodejs \
    npm && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create and activate a virtual environment
RUN python3 -m venv .venv
ENV PATH="/app/.venv/bin:${PATH}"

COPY requirements.txt .

# Install uv and use it for faster, more robust dependency resolution
RUN .venv/bin/pip install --upgrade pip uv && \
    .venv/bin/uv pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.12

# Install runtime dependencies
# poppler-utils is required for pdf2image (multimodal entity extraction)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libxml2 \
    libxslt1.1 \
    libxmlsec1-openssl \
    libffi8 \
    xmlsec1 \
    poppler-utils && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=backend-builder /app/.venv ./.venv
ENV PATH="/app/.venv/bin:${PATH}"
ENV VIRTUAL_ENV="/app/.venv"\nENV PYTHONPATH="/app"
ENV XMLSEC_ROOT="/usr"
ENV LIBXML2_VERSION="2.9"

COPY . .

EXPOSE 5000
CMD ["python", "run.py"]
