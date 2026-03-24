# Use a slim Python base image
FROM python:3.12.7-slim-bookworm


# Install uv globally
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    TP_SOURCE="CE" \
    MODE="prod"

COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock

# Install dependencies
RUN uv sync

# Copy the frontend needed files
COPY src/ /app/src
COPY images/front/start_services.sh /app/docker/start_services.sh

# Make the wrapper script executable
RUN chmod +x /app/docker/start_services.sh

# Expose the frontend port
EXPOSE 6030

# Set the entrypoint to the start script
CMD [ "/app/docker/start_services.sh" ]
