# No `# syntax=` line: it fetches the Dockerfile frontend from Docker Hub on every build, which
# hangs when Docker Hub is slow. The built-in frontend already supports RUN --mount.
# AgentCore Runtime needs linux/arm64:  make image
# (docker buildx build --platform linux/arm64 -t ticket-agent:dev --load .)

ARG PYTHON_VERSION=3.12
# The uv the image is built with. Locally any uv >= [tool.uv] required-version works.
ARG UV_VERSION=0.13.1

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---- build stage: resolve nothing, install exactly what uv.lock pins ----
FROM python:${PYTHON_VERSION}-slim AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app

# Dependencies first: this layer only rebuilds when uv.lock or pyproject.toml change.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project

# Then the project, installed non-editable so the runtime image needs no source tree.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# ---- runtime stage: the virtualenv only, no uv, no source, non-root ----
FROM python:${PYTHON_VERSION}-slim AS runtime
RUN useradd --system --uid 10001 --no-create-home app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
# Non-secret settings; override any value with an environment variable (TICKETING__URL=...).
WORKDIR /app
COPY --chown=app:app agent.toml ./
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 8080
CMD ["uvicorn", "ticket_agent.main:app", "--host", "0.0.0.0", "--port", "8080"]
