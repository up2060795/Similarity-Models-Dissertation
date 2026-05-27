# ARG for Python version (e.g., 3.11, 3.12) to be used for the base image tag. .env file will override this.
ARG PYTHON_VERSION=3.12
# ARG for UV version to be installed (e.g., 0.2.0, or your desired version). .env file will override this.
ARG UV_VERSION=0.2.0

# --- Stage 0: Fetch UV binaries ---
# This stage uses the specified UV_VERSION to pull the correct astral/uv image
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv_fetcher

# --- Stage 1: Base Image Setup ---
FROM python:${PYTHON_VERSION}-slim-bookworm AS base

# Define application home and virtual environment path first
ENV APP_HOME=/app
ENV VENV_PATH_IN_IMAGE=${APP_HOME}/.venv

# Set other environment variables
ENV PYTHONUNBUFFERED=1 \
    # Tells uv to compile .pyc files for potentially faster imports
    UV_COMPILE_BYTECODE=1 \
    # When using mounted caches, uv should copy files instead of trying to link
    UV_LINK_MODE=copy \
    # Ensure apt runs non-interactively
    DEBIAN_FRONTEND=noninteractive \
    # Define UV cache directory (uv commands will run as root initially)
    UV_CACHE_DIR_DOCKER=/root/.cache/uv

# Add the virtual environment's bin directory to the PATH
# This ensures subsequent RUN commands and the final CMD use the venv python
ENV PATH="${VENV_PATH_IN_IMAGE}/bin:${PATH}"

# Install UV from the uv_fetcher stage.
COPY --from=uv_fetcher /uv /uvx /usr/local/bin/

# Install git (for pip installs from git),
# build-essential (for C extensions), and any other essential system dependencies.
# These operations are done as root.
RUN \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        # Add any other system dependencies your application might need here
    && \
    # Clean up downloaded packages and apt cache
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR ${APP_HOME}

# Create the virtual environment using the Python from the base image
# The PATH is already set, so `python` should resolve correctly,
# but being explicit with base python can sometimes avoid issues if PATH was complex.
# However, with the PATH set above, `python -m venv` should use the base python.
RUN python -m venv ${VENV_PATH_IN_IMAGE}

# --- Stage 2: Builder Stage (Install Dependencies) ---
# Use a separate stage to install dependencies to leverage Docker layer caching
FROM base AS builder

# WORKDIR is inherited from base (${APP_HOME})

# Copy only essential files for dependency installation
COPY pyproject.toml uv.lock ./

# Install project dependencies (excluding the project itself and dev dependencies) into the virtual environment.
# This RUN command executes as root. The `uv` command will use the venv Python
# because ${VENV_PATH_IN_IMAGE}/bin is already in PATH.
RUN \
    --mount=type=cache,target=${UV_CACHE_DIR_DOCKER} \
    uv sync --locked --no-install-project --no-dev --python ${VENV_PATH_IN_IMAGE}/bin/python

# --- Stage 3: Final Runtime Stage ---
# Start from the base image again to keep the final image lean
FROM base AS final

# WORKDIR is inherited from base (${APP_HOME})
# PATH is inherited from base, already pointing to the venv bin.

# Copy the virtual environment with installed dependencies from the builder stage
COPY --from=builder ${VENV_PATH_IN_IMAGE} ${VENV_PATH_IN_IMAGE}

# Set PYTHONPATH to include the app directory if your app uses absolute imports from APP_HOME
ENV PYTHONPATH=${APP_HOME}:\$PYTHONPATH

# Copy the entire application source code into the working directory.
# This comes after venv copy to optimize Docker layer caching.
COPY . ${APP_HOME}/

# Install the project itself into the virtual environment,
# along with its non-dev dependencies, using the lock file.
# This RUN command also executes as root.
# pyproject.toml and uv.lock are now in ${APP_HOME} from the `COPY . .`
RUN \
    --mount=type=cache,target=${UV_CACHE_DIR_DOCKER} \
    uv sync --locked --no-dev --python ${VENV_PATH_IN_IMAGE}/bin/python


# Reset entrypoint in case the base Python image had one.
ENTRYPOINT []

# Define the default command to run your application.
# Adjust this CMD to how your specific application should be started.
CMD ["python", "-m", "your_module.main"]

# Example for a FastAPI application (if applicable):
# CMD ["uvicorn", "your_app.main:app", "--host", "0.0.0.0", "--port", "80"]
