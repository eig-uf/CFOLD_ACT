# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.9.11@sha256:5aa820129de0a600924f166aec9cb51613b15b68f1dcd2a02f31a500d2ede568 AS uv
FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d

# FFmpeg 5 supplies the shared libraries used by TorchCodec 0.5.
# OpenCV is headless; no desktop or separate OpenCV installation is needed.
# v4l-utils supports scripts/find_cameras.sh; git supports development in VS Code.
# evdev (via LeRobot/pynput) builds a C extension against Linux input headers.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates ffmpeg gcc git libc6-dev libgomp1 linux-libc-dev v4l-utils \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid "${USER_GID}" vscode \
    && useradd --uid "${USER_UID}" --gid vscode --create-home --shell /bin/bash vscode \
    && mkdir -p /workspaces/CFOLD_ACT \
    && chown vscode:vscode /workspaces/CFOLD_ACT

# Keep dependencies outside the bind mount, including any host .venv.
# Locating the environment in HOME also lets Dev Containers remap its ownership
# when the host Linux UID differs from 1000.
ENV UV_PROJECT_ENVIRONMENT=/home/vscode/.venv \
    VIRTUAL_ENV=/home/vscode/.venv \
    PATH="/home/vscode/.venv/bin:$PATH" \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspaces/CFOLD_ACT
USER vscode

# Cache third-party dependencies independently of source edits.
COPY --chown=vscode:vscode pyproject.toml uv.lock ./
RUN uv sync --locked --extra test --no-install-project

COPY --chown=vscode:vscode . .
RUN uv sync --locked --extra test

CMD ["bash"]
