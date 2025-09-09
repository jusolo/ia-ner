#!/usr/bin/env python3
"""
Script para evaluar modelos NER entrenados
"""

import os
import yaml
import torch
from datasets import load_from_disk, load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    pipeline
)
import evaluate
from utils.data_utils import tokenize_and_align_labels, get_label_names
from utils.model_utils import compute_metrics
import pandas as pd

def load_config():
    """Cargar configuración"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def evaluate_model(config):
    """Evaluar modelo entrenado"""
    print("Evaluando modelo...")
    
    model_path = config['training']['output_dir']
    
    # Verificar que el modelo existe
    if not os.path.exists(model_path):
        print(f"Error: No se encontró modelo en {model_path}")
        return None
    
    # Cargar modelo y tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    # Cargar dataset de test
    try:
        dataset_path = os.path.join(config['data_paths']['raw_data'], config['dataset']['name'])
        if os.path.exists(dataset_path):
            dataset = load_from_disk(dataset_path)
        else:
            dataset = load_dataset(config['dataset']['name'])
    except Exception as e:
        print(f"Error cargando dataset: {e}")
        return None
    
    # Obtener nombres de labels
    label_names = get_label_names(dataset[config['dataset']['train_split']])
    
    # Preparar dataset de test
    def tokenize_function(examples):
        return tokenize_and_align_labels(examples, tokenizer, config)
    
    test_dataset = dataset[config['dataset']['test_split']].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset[config['dataset']['test_split']].column_names
    )
    
    # Crear pipeline para predicciones
    ner_pipeline = pipeline(
        "token-classification", 
        model=model, 
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )
    
    print("Ejecutando evaluación en dataset de test...")
    
    # Evaluar en algunos ejemplos
    test_examples = dataset[config['dataset']['test_split']][:10]  # Primeros 10 ejemplos
    
    results = []
    for i, example in enumerate(test_examples):
        tokens = example[config['dataset']['text_column']]
        text = " ".join(tokens)
        
        # Obtener predicciones
        predictions = ner_pipeline(text)
        
        result = {
            'example': i,
            'text': text[:100] + "..." if len(text) > 100 else text,
            'predictions': predictions
        }
        results.append(result)
        
        print(f"\nEjemplo {i+1}:")
        print(f"Texto: {text[:100]}...")
        print("Entidades encontradas:")
        for pred in predictions:
            print(f"  - {pred['word']}: {pred['entity_group']} (score: {pred['score']:.3f})")
    
    return results

def test_custom_text(config, text_samples):
    """Probar modelo con texto personalizado"""
    print("\nProbando con texto personalizado...")
    
    model_path = config['training']['output_dir']
    
    # Cargar modelo y tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    # Crear pipeline
    ner_pipeline = pipeline(
        "token-classification", 
        model=model, 
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )
    
    for i, text in enumerate(text_samples):
        print(f"\nTexto de prueba {i+1}: {text}")
        predictions = ner_pipeline(text)
        
        if predictions:
            print("Entidades encontradas:")
            for pred in predictions:
                print(f"  - {pred['word']}: {pred['entity_group']} (score: {pred['score']:.3f})")
        else:
            print("  No se encontraron entidades")

def main():
    """Función principal"""
    config = load_config()
    
    # Evaluar modelo
    results = evaluate_model(config)
    
    if results:
        print("\n" + "="*50)
        print("EVALUACIÓN COMPLETADA")
        print("="*50)
    
    # Textos de ejemplo para probar
    test_texts = [
        "Mi nombre es Juan Pérez y vivo en Madrid, España.",
        "Apple Inc. fue fundada por Steve Jobs en California.",
        "La reunión es el próximo lunes a las 3:00 PM."
    ]
    
    test_custom_text(config, test_texts)

if __name__ == "__main__":
    main()