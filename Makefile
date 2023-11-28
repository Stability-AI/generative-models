# Contains helpful make targets for development

.venv: requirements.txt ## Create a virtual environment and install dependencies
	python3 -m venv --clear .venv
	.venv/bin/pip install wheel pip-tools
	.venv/bin/pip-compile requirements.in
	.venv/bin/pip install -r requirements.txt

.PHONY: compile-requirements
compile-requirements: .venv ## Compile requirements.in to requirements.txt
	.venv/bin/pip-compile requirements.in
	.venv/bin/pip install -r requirements.txt

.PHONY: compile-requirements-linux
compile-requirements-linux: ## Compile requirements.in to requirements.txt (in a linux container)
	# Build the docker image
	docker build --platform=linux/amd64 \
		-t sd-compile-requirements \
		-f scripts/Dockerfile.compile-requirements \
		.
	# Run the docker image (to copy the requirements.txt file out)
	docker run --platform=linux/amd64 \
		-v $(PWD):/app \
		-t sd-compile-requirements \
		cp /tmp/requirements.txt requirements.txt

.PHONY: clean
clean: ## Remove the virtual environment
	@rm -rf .venv
