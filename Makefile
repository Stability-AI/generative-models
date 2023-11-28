# Contains helpful make targets for development

.venv: requirements.txt ## Create a virtual environment and install dependencies
	python3 -m venv --clear .venv
	.venv/bin/pip install -r requirements.txt

.PHONY: clean
clean: ## Remove the virtual environment
	@rm -rf .venv
