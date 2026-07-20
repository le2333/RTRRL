ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update \
    && apt-get install --yes --no-install-recommends time \
    && rm -rf /var/lib/apt/lists/*
COPY training-sdk /workspace/training-sdk
COPY memo /app
RUN ln -s /app /app/memo
RUN /opt/venv/bin/python -m ensurepip
RUN /opt/venv/bin/python -m pip install /workspace/training-sdk pytest

WORKDIR /app
ENV PYTHONPATH=/workspace/training-sdk/src:/app \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MALLOC_ARENA_MAX=2

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
