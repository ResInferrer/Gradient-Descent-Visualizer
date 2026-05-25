FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    python3-dev \
    pybind11-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN mkdir build && cd build && \
    cmake .. && \
    make && \
    SO_FILE=$(find . -name "*.so" | head -1) && \
    echo "Found SO file: $SO_FILE" && \
    cp "$SO_FILE" .. && \
    cd .. && \
    rm -rf build

RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser
ENV PYTHONPATH="/app:${PYTHONPATH}"

CMD ["python", "main.py"]