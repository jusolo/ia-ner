#!/usr/bin/env python3
"""
Script para descargar y preparar datasets desde HuggingFace
"""

import os
import sys
import yaml

# Configurar rutas correctamente
def setup_paths():
    """Configurar las rutas del proyecto"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    os.chdir(project_root)
    return project_root

# Configurar rutas antes de imports
project_root = setup_paths()

from datasets import load_dataset, Dataset, DatasetDict
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
        dataset_name = config['dataset']['name']
        
        print(f"Cargando dataset español de NER: {dataset_name}")
        dataset = load_dataset(dataset_name, trust_remote_code=True)
        
        print(f"Dataset descargado exitosamente!")
        print(f"Splits disponibles: {list(dataset.keys())}")
        
        # Mostrar información del dataset
        for split_name, split_data in dataset.items():
            print(f"{split_name}: {len(split_data)} ejemplos")
        
        # Inspeccionar la estructura del dataset
        train_split = dataset[config['dataset']['train_split']]
        print(f"\nColumnas disponibles: {train_split.column_names}")
        
        # Mostrar ejemplo de datos
        train_example = train_split[0]
        print(f"\nEjemplo de datos:")
        
        # Verificar qué columnas tiene realmente el dataset
        for col_name in train_split.column_names:
            if col_name in train_example:
                example_data = train_example[col_name]
                if isinstance(example_data, list):
                    print(f"{col_name}: {example_data[:10]}..." if len(example_data) > 10 else f"{col_name}: {example_data}")
                else:
                    print(f"{col_name}: {example_data}")
        
        # Examinar los labels si están disponibles
        if 'ner_tags' in train_example or 'labels' in train_example:
            label_column = 'ner_tags' if 'ner_tags' in train_example else 'labels'
            unique_labels = set()
            
            # Obtener muestra de labels únicos
            for i in range(min(100, len(train_split))):
                example = train_split[i]
                if label_column in example:
                    unique_labels.update(example[label_column])
            
            print(f"\nLabels únicos encontrados: {sorted(unique_labels)}")
            
            # Si el dataset tiene información de features
            if hasattr(train_split, 'features') and label_column in train_split.features:
                feature = train_split.features[label_column]
                if hasattr(feature, 'feature') and hasattr(feature.feature, 'names'):
                    print(f"Nombres de labels: {feature.feature.names}")
        
        # Guardar dataset localmente
        dataset_path = os.path.join(config['data_paths']['raw_data'], config['dataset']['name'].replace('/', '_'))
        dataset.save_to_disk(dataset_path)
        print(f"Dataset guardado en: {dataset_path}")
        
        return dataset
        
    except Exception as e:
        print(f"Error descargando dataset: {e}")
        print("Detalles del error:")
        import traceback
        traceback.print_exc()
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