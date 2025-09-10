.PHONY: install clean download-data train evaluate help

# Variables
PYTHON := python3
PIP := pip3
VENV := venv
PROJECT_ROOT := $(shell pwd)

help: ## Mostrar ayuda
	@echo "Comandos disponibles:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $1, $2}'

install: ## Instalar dependencias
	$(PIP) install -r requirements.txt

install-dev: ## Instalar dependencias de desarrollo
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

setup-venv: ## Crear y configurar entorno virtual
	$(PYTHON) -m venv $(VENV)
	@echo "Activa el entorno virtual con: source $(VENV)/bin/activate"

check: ## Verificar instalación
	PYTHONPATH=$(PROJECT_ROOT) $(PYTHON) check_installation.py

download-data: ## Descargar dataset desde HuggingFace
	PYTHONPATH=$(PROJECT_ROOT) $(PYTHON) src/download_data.py

train: ## Entrenar modelo NER
	PYTHONPATH=$(PROJECT_ROOT) $(PYTHON) src/train.py

evaluate: ## Evaluar modelo entrenado
	PYTHONPATH=$(PROJECT_ROOT) $(PYTHON) src/test_model.py

clean: ## Limpiar archivos temporales
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

clean-models: ## Limpiar modelos guardados
	rm -rf models/*
	touch models/.gitkeep

clean-data: ## Limpiar datos descargados
	rm -rf data/*
	touch data/.gitkeep

clean-all: clean clean-models clean-data ## Limpiar todo