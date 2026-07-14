# CPU-only training image for RTRRL / PPO baseline.
# Multi-stage: build deps into a venv with uv, then ship a slim runtime with the
# AWS CLI for S3 checkpoint sync. GPU is intentionally NOT installed (see README).

# ---- builder: resolve + install dependencies with uv -------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# git: for the popjym git dependency. build-essential (g++): some deps have no
# prebuilt wheel and compile C/C++ extensions from source (e.g. pytinyrenderer
# via brax). These tools stay in the builder; the runtime only copies /opt/venv.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy

WORKDIR /app

# Only the dependency manifests first, so this layer caches across code changes.
COPY pyproject.toml uv.lock ./
# pyproject has [tool.uv] package = false, so this installs deps only (no project).
RUN uv sync --frozen --no-dev

# ---- runtime -----------------------------------------------------------------
FROM python:3.12-slim AS runtime

# AWS CLI v2 for S3 checkpoint sync from the container (uses the Batch job role).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl unzip ca-certificates \
    && curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/awscliv2.zip /tmp/aws \
    && apt-get purge -y unzip \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Bring in the prebuilt virtualenv and put it on PATH (no uv needed at runtime).
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    JAX_PLATFORM_NAME=cpu \
    XLA_FLAGS="--xla_force_host_platform_device_count=1"

WORKDIR /app

# Application code (build context is the repo root; see .dockerignore).
# config/, docs/, figures/, logs/ are excluded — configs are injected at runtime.
COPY . /app

# Entrypoint decodes the submit-time CONFIG_B64 into /tmp/run-config.yml, then exec's.
COPY infra/docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# Default command is a no-op help; real command comes from the Batch job definition.
CMD ["python", "rtrrl.py", "--help"]
