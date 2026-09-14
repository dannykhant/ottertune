FROM --platform=linux/amd64 python:3.7-slim

WORKDIR /ottertune

# System dependencies for psycopg2, docker CLI, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libpq-dev docker.io \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-tuner.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH="/ottertune/server"

ENTRYPOINT ["python", "ottertune_tuner.py"]
