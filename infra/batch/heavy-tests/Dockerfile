ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY training-sdk /workspace/training-sdk
COPY memo /app
RUN /opt/venv/bin/python -m ensurepip
RUN /opt/venv/bin/python -m pip install /workspace/training-sdk pytest

WORKDIR /app
ENV PYTHONPATH=/workspace/training-sdk/src:/app \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MALLOC_ARENA_MAX=2

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
