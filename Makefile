.ENV := .venv
.VERSION := 1.0
.TAG := $(VERSION).$(shell date +'%Y%m%d%H%M')
.DEFAULT_GOAL := help
.PHONY: help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

run: ## Run the server
	python -m src.main

train: ## Train the model
	python training.py

test: ## Run tests
	pytest -v
