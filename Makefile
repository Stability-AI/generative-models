# Contains helpful make targets for development
UNAME_S?=$(shell uname -s)
REQUIREMENTS_FILE?=requirements.txt
CUDA_DOCKER_VERSION?=11.8.0
ifeq ($(UNAME_S), Darwin)
	REQUIREMENTS_FILE=requirements-macos.txt
endif

.venv: requirements/requirements.in ## Create a virtual environment and install dependencies
	python3 -m venv --clear .venv
	.venv/bin/pip install wheel pip-tools
	.venv/bin/pip-compile requirements/requirements.in --output-file=requirements/$(REQUIREMENTS_FILE)
	.venv/bin/pip install -r $(REQUIREMENTS_FILE)

.PHONY: compile-requirements
compile-requirements: .venv ## Compile requirements.in to requirements.txt
		.venv/bin/pip-compile requirements/requirements.in --output-file=requirements/$(REQUIREMENTS_FILE)

.PHONY: compile-requirements-docker
compile-requirements-docker: ## Compile requirements.in to requirements.txt (in a docker container)
	# Build the docker image
	docker build --platform=linux/amd64 \
		--build-arg CUDA_DOCKER_VERSION=$(CUDA_DOCKER_VERSION) \
		--target final \
		-t sd-compile-requirements \
		-f scripts/Dockerfile.compile-requirements \
		.
	# Run the docker image (to copy the requirements.txt file out)
	docker run --platform=linux/amd64 \
		--gpus all \
		-v $(PWD):/app \
		-t sd-compile-requirements \
		cp /tmp/requirements.txt requirements/$(REQUIREMENTS_FILE)

.PHONY: test
test: test-inference ## Run tests

.PHONY: test-inference
test-inference: .venv ## Run inference tests
	.venv/bin/pytest -v tests/inference/test_inference.py

.PHONY: test-inference-docker
test-inference-docker: ## Run inference tests (in a docker container)
	# Build the docker image
	docker build --platform=linux/amd64 \
		--build-arg CUDA_DOCKER_VERSION=$(CUDA_DOCKER_VERSION) \
		--target test-inference \
		-t sd-test-inference \
		-f scripts/Dockerfile.compile-requirements \
		.
	# Run the docker image
	docker run --platform=linux/amd64 \
		-v $(PWD):/app \
		-t sd-test-inference

.PHONY: clean
clean: ## Remove the virtual environment
	@rm -rf .venv

.DELETE_ON_ERROR: ## Configure make to delete the target of a rule if it has an error

