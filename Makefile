# ==============================================================================
# Main Makefile
#
# Provides targets for common development tasks including environment setup,
# dependency management, linting, testing, running the application,
# Docker image management, and cleanup.
# ==============================================================================
# --- Configuration Variables ---
# Load environment variables from .env file if it exists.
-include .env

# --- Entry point for the application ---
# Note: we run as a module (e.g., 'python -m your_module.main') to ensure correct import resolution and package discovery.
# This approach sets up the package context properly, allowing relative imports and dependencies to work as intended.
APP_ENTRYPOINT := your_module.main

# --- Shell Configuration ---
SHELL       := /bin/bash
# -e: exit immediately if a command exits with a non-zero status.
# -o pipefail: the return value of a pipeline is the status of the last command.
SHELLFLAGS  := -e -o pipefail -c
# --- OS Detection ---
OS             := $(shell uname -s)

# --- Environment Configuration ---
# Add common local bin directories to PATH for this make session
# This helps find 'uv' if it was just installed but shell isn't reloaded.
export PATH := $(HOME)/.local/bin:$(HOME)/.cargo/bin:$(PATH)

# Add Homebrew paths on macOS
ifeq ($(OS), Darwin)
export PATH := /opt/homebrew/bin:/usr/local/bin:$(PATH)
endif

# Extract python version from pyproject.toml, strictly enforcing the format:
# requires-python = "==<major>.<minor>.<patch>"
PYTHON_VERSION := $(shell grep 'requires-python' pyproject.toml | \
                      sed -n 's/.*"==\([0-9]\{1,\}\.[0-9]\{1,\}\.[0-9]\{1,\}\)".*/\1/p')

# Check if PYTHON_VERSION was extracted.
ifeq ($(PYTHON_VERSION),)
$(error ERROR: 'pyproject.toml' must contain 'requires-python' in the format "==<version>". Example: requires-python = "==3.11.4". Please correct the file.)
endif

# help: UV_VERSION     - UV (Python packager) version (e.g., 0.9.9)
UV_VERSION     ?= 0.9.9
# help: IMAGE_NAME     - Docker image name (Default: my-python-app)
IMAGE_NAME     ?= my-python-app
# help: IMAGE_TAG      - Docker image tag (Default: latest)
IMAGE_TAG      ?= latest
# help: PYPROJECT_FILE - Path to pyproject.toml (Default: pyproject.toml)
PYPROJECT_FILE ?= pyproject.toml
# help: DOCKER_CMD     - Command to run Docker (Default: docker)
# (Use 'sudo docker' only if your user is not in the 'docker' group)
DOCKER_CMD          ?= docker
DOCKER_BUILDKIT_VAR ?= DOCKER_BUILDKIT=1
# --- Tool Commands ---
# UV command, assumes uv is in PATH after setup-uv target.
UV_CMD         := uv


# ==============================================================================
# Help Target
# ==============================================================================
.PHONY: help
help: ## ✨ Display this help message
	@echo "Makefile for $(IMAGE_NAME)"
	@echo "--------------------------------"
	@echo "Python Version:  $(PYTHON_VERSION)"
	@echo "App Entry Point: $(APP_ENTRYPOINT)"
	@echo ""
	@grep -h -E '^[a-zA-Z_0-9.-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-25s %s\n", $$1, $$2}' | sort
	@echo ""
	@echo "Configuration Variables (override with 'make TARGET VAR=value'):"
	@grep -E '^# help: [A-Z_]\+' $(firstword $(MAKEFILE_LIST)) | \
		sed -e 's/^# help: //' | column -t -s '-' | sed -e 's/^/  /'

# ==============================================================================
# Core Setup Targets
# ==============================================================================
.PHONY: check-curl
check-curl: ## ⚙️ (Internal) Check if curl is installed
	@if ! command -v curl &> /dev/null; then \
		echo "Error: 'curl' is not installed or not in PATH."; \
		echo "Please install 'curl' to proceed with 'setup-uv'."; \
		exit 1; \
	fi
	@echo "--- curl is installed ---"

.PHONY: setup-uv
setup-uv: check-curl ## 🛠️ Install/Ensure correct UV version
	@echo "--- Ensuring UV $(UV_VERSION) ---"
	@if ! command -v $(UV_CMD) &> /dev/null || ! ($(UV_CMD) --version 2>/dev/null | grep -q "uv $(UV_VERSION)"); then \
		echo "Installing/Updating UV to $(UV_VERSION)..."; \
		curl -LsSf https://astral.sh/uv/$(UV_VERSION)/install.sh | sh; \
	else \
		echo "UV $(UV_VERSION) is already installed."; \
	fi

# ==============================================================================
# Project Workflow Targets
# ==============================================================================
.PHONY: install
install: setup-uv  ## 🚀 Install dependencies & package
	@echo "--- Installing project dependencies and package ---"
	@# uv sync handles Python download (if missing), venv creation, and editable install
	$(UV_CMD) sync --all-extras --project $(PYPROJECT_FILE) || exit 1

	@echo "Checking for pre-commit hooks..."
	@if $(UV_CMD) run --quiet -- python -m pre_commit --version &> /dev/null; then \
		echo "-> Installing pre-commit hooks..."; \
		$(UV_CMD) run pre-commit install -t pre-commit -t pre-push; \
	else \
		echo "-> pre-commit not found, skipping hooks."; \
	fi
	@echo "✅ Installation complete."

.PHONY: lint
lint: install ## 🧹 Lint code using pre-commit
	@echo "--- Linting code ---"
	$(UV_CMD) run pre-commit run --all-files
	@echo "✅ Linting finished."

.PHONY: test
test: install ## 🧪 Run tests using pytest
	@echo "--- Running tests ---"
	$(UV_CMD) run pytest
	@echo "✅ Tests finished."

.PHONY: run
run: install ## ▶️ Run the main application (main.py)
	@echo "--- Running application (main.py) ---"
	$(UV_CMD) run python -m $(APP_ENTRYPOINT)
	@echo "✅ Application finished."

# ==============================================================================
# Docker Targets
# ==============================================================================

.PHONY: check-docker-installed
check-docker-installed: ## ⚙️ (Internal) Check if Docker is installed
	@if ! command -v $(DOCKER_CMD) &> /dev/null; then \
		echo "Error: Docker client is not installed or not in PATH."; \
		echo "Please install Docker to proceed with Docker-related targets."; \
		echo "Instructions are present in the README.md file."; \
		exit 1; \
	else \
		echo "--- Docker client is installed ---"; \
	fi

.PHONY: check-colima-installed
check-colima-installed: check-docker-installed ## ⚙️ (Internal) Check if Colima is installed on macOS
	@if [ "$(OS)" = "Darwin" ]; then \
		if ! command -v colima &> /dev/null; then \
			echo "Error: Colima is not installed."; \
			echo "Please install Colima to proceed with Docker-related targets on macOS."; \
			echo "Instructions are present in the README.md file."; \
			exit 1; \
		else \
			echo "--- Colima is installed ---"; \
		fi; \
	else \
		echo "Not macOS, skipping Colima installation check."; \
	fi


.PHONY: check-colima-running
check-colima-running: ## ⚙️ (Internal) Check if Colima is running on macOS
	@if [ "$(OS)" = "Darwin" ]; then \
		if ! colima status &> /dev/null; then \
			echo "Colima is not running. Starting Colima..."; \
			colima start || { echo "Error: Failed to start Colima."; exit 1; }; \
			echo "Colima started successfully."; \
		else \
			echo "Colima is already running."; \
		fi; \
	else \
		echo "Not macOS, skipping Colima running check."; \
	fi

.PHONY: build-docker
build-docker: check-colima-running ## 🐳 Build the Docker image
	@echo "--- Building Docker image $(IMAGE_NAME):$(IMAGE_TAG) ---"
	$(DOCKER_BUILDKIT_VAR) $(DOCKER_CMD) build \
	  --build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
	  --build-arg UV_VERSION=$(UV_VERSION) \
	  -t $(IMAGE_NAME):$(IMAGE_TAG) .
	@echo "✅ Docker image $(IMAGE_NAME):$(IMAGE_TAG) built."

.PHONY: run-docker
run-docker: ## 🚢 Build and run the Docker container interactively (passes .env variables if present)
	@make build-docker # Ensure image is built
	@echo "--- Running Docker container $(IMAGE_NAME):$(IMAGE_TAG) ---"
	@if [ -f .env ]; then \
		echo "Running command: $(DOCKER_CMD) run --rm -it --env-file .env $(IMAGE_NAME):$(IMAGE_TAG)"; \
		$(DOCKER_CMD) run --rm -it --env-file .env $(IMAGE_NAME):$(IMAGE_TAG); \
	else \
		echo "No .env file found, running without environment variables"; \
		echo "Running command: $(DOCKER_CMD) run --rm -it $(IMAGE_NAME):$(IMAGE_TAG)"; \
		$(DOCKER_CMD) run --rm -it $(IMAGE_NAME):$(IMAGE_TAG); \
	fi

.PHONY: push-docker
push-docker: ## ⬆️ Push the Docker image to a registry (requires login)
	@make build-docker # Ensure image is built
	@echo "--- Pushing Docker image $(IMAGE_NAME):$(IMAGE_TAG) ---"
	$(DOCKER_CMD) push $(IMAGE_NAME):$(IMAGE_TAG)
	@echo "✅ Docker image $(IMAGE_NAME):$(IMAGE_TAG) pushed."

# ==============================================================================
# Cleanup Target
# ==============================================================================
.PHONY: clean
clean: ## 🗑️ Remove virtual environment and __pycache__ directories
	@echo "--- Removing virtual environment: .venv ---"
	@rm -rf .venv
	@echo "--- Removing __pycache__ directories ---"
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "🧹 Cleanup complete."
	@echo "Note: Globally installed tools (like uv or system Python) are NOT removed."

# ==============================================================================
# Phony Targets Declaration
# ==============================================================================
.PHONY: \
	help \
	check-curl setup-uv \
	install lint test run \
	check-docker-installed check-colima-installed check-colima-running \
	build-docker run-docker push-docker \
	clean

# --- Default Goal ---
# If 'make' is run without arguments, run the 'help' target.
.DEFAULT_GOAL := help
