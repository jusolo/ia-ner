#!/usr/bin/env python3
"""
Script para descargar y preparar datasets desde HuggingFace
"""

import os
import yaml
from datasets import load_dataset
from pathlib import Path

def load_config():
    """Cargar configuración desde config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def download_dataset(config):
    """
    Descargar dataset desde HuggingFace
    """
    print(f"Descargando dataset: {config['dataset']['name']}")
    
    # Crear directorios si no existen
    os.makedirs(config['data_paths']['raw_data'], exist_ok=True)
    os.makedirs(config['data_paths']['processed_data'], exist_ok=True)
    
    try:
        # Cargar dataset desde HuggingFace
        dataset = load_dataset(config['dataset']['name'])
        
        print(f"Dataset descargado exitosamente!")
        print(f"Splits disponibles: {list(dataset.keys())}")
        
        # Mostrar información del dataset
        for split_name, split_data in dataset.items():
            print(f"{split_name}: {len(split_data)} ejemplos")
            
        # Guardar dataset localmente
        dataset_path = os.path.join(config['data_paths']['raw_data'], config['dataset']['name'])
        dataset.save_to_disk(dataset_path)
        print(f"Dataset guardado en: {dataset_path}")
        
        # Mostrar ejemplo de datos
        train_example = dataset[config['dataset']['train_split']][0]
        print(f"\nEjemplo de datos:")
        print(f"Tokens: {train_example[config['dataset']['text_column']][:10]}...")
        print(f"Labels: {train_example[config['dataset']['label_column']][:10]}...")
        
        return dataset
        
    except Exception as e:
        print(f"Error descargando dataset: {e}")
        return None

def main():
    """Función principal"""
    config = load_config()
    dataset = download_dataset(config)
    
    if dataset:
        print("¡Dataset preparado exitosamente!")
    else:
        print("Error preparando el dataset")

if __name__ == "__main__":
    main()