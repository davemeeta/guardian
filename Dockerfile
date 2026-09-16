# Single shared image for all three services (ml, agent, dashboard) —
# docker-compose.yml differentiates them only by CMD. They share the same
# heavy scientific-Python base (lightgbm, torch, scikit-learn) anyway since
# the agent service retrains a GBM baseline in-process to build evidence, so
# splitting into separate images would duplicate that weight for no benefit.
#
# Ollama itself is deliberately NOT containerized here — its model weights
# are large and Docker on macOS can't pass through Metal GPU acceleration,
# so it runs as a normal process on the host; the agent service reaches it
# via OLLAMA_HOST=http://host.docker.internal:11434 (see docker-compose.yml).
FROM python:3.11-slim

WORKDIR /app

# lightgbm needs libgomp at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src

# Install the CPU-only torch build first, from PyTorch's own CPU index —
# none of these services use GPU (containers can't reach macOS's Metal
# backend anyway, and the host doesn't have CUDA), so the default PyPI
# torch wheel would otherwise pull several GB of unused CUDA libraries.
# Once torch is already satisfied, installing the rest of the package
# won't re-resolve it against the CUDA-enabled default.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -e .

COPY configs ./configs
COPY scripts ./scripts

# Bakes the dataset into the image so the container is self-contained —
# no host prep beyond `docker compose up` is required.
RUN python scripts/download_data.py

ENV PYTHONPATH=/app/src
ENV MLFLOW_DISABLE_TELEMETRY=true

EXPOSE 8000
