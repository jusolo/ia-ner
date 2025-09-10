#!/usr/bin/env python3
"""
Script principal para entrenar modelos NER
"""

import os
import yaml
import torch
from datasets import load_from_disk, load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import evaluate
from utils.data_utils import tokenize_and_align_labels, get_label_names
from utils.model_utils import compute_metrics

def load_config():
    """Cargar configuración"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def prepare_dataset(config):
    """Preparar dataset para entrenamiento"""
    print("Preparando dataset español...")
    
    # Cargar dataset
    try:
        # Normalizar el nombre del dataset para el path local
        dataset_name_path = config['dataset']['name'].replace('/', '_')
        dataset_path = os.path.join(config['data_paths']['raw_data'], dataset_name_path)
        
        if os.path.exists(dataset_path):
            print(f"Cargando dataset desde: {dataset_path}")
            dataset = load_from_disk(dataset_path)
        else:
            print(f"Descargando dataset: {config['dataset']['name']}")
            dataset = load_dataset(config['dataset']['name'], trust_remote_code=True)
            
    except Exception as e:
        print(f"Error cargando dataset: {e}")
        return None, None, None
    
    # Inspeccionar el dataset
    train_split = dataset[config['dataset']['train_split']]
    print(f"Columnas disponibles: {train_split.column_names}")
    
    # Verificar y ajustar nombres de columnas si es necesario
    actual_columns = train_split.column_names
    
    # Para el dataset PlanTL-GOB-ES/CoNLL-NERC-es, verificar las columnas reales
    text_column = config['dataset']['text_column']
    label_column = config['dataset']['label_column']
    
    if text_column not in actual_columns:
        # Buscar columnas que contengan tokens/words
        possible_text_cols = [col for col in actual_columns if 'token' in col.lower() or 'word' in col.lower() or 'text' in col.lower()]
        if possible_text_cols:
            text_column = possible_text_cols[0]
            print(f"Usando columna de texto: {text_column}")
            config['dataset']['text_column'] = text_column
    
    if label_column not in actual_columns:
        # Buscar columnas que contengan labels/tags
        possible_label_cols = [col for col in actual_columns if 'tag' in col.lower() or 'label' in col.lower() or 'ner' in col.lower()]
        if possible_label_cols:
            label_column = possible_label_cols[0]
            print(f"Usando columna de labels: {label_column}")
            config['dataset']['label_column'] = label_column
    
    # Obtener nombres de labels
    label_names = get_label_names(train_split)
    num_labels = len(label_names)
    
    print(f"Labels encontrados: {label_names}")
    print(f"Número de labels: {num_labels}")
    
    # Cargar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Tokenizar dataset
    def tokenize_function(examples):
        return tokenize_and_align_labels(examples, tokenizer, config)
    
    try:
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=train_split.column_names
        )
        print("Dataset tokenizado exitosamente")
    except Exception as e:
        print(f"Error tokenizando dataset: {e}")
        return None, None, None
    
    return tokenized_dataset, tokenizer, label_names

def train_model(config):
    """Entrenar modelo NER"""
    print("Iniciando entrenamiento...")
    
    # Preparar datos
    dataset, tokenizer, label_names = prepare_dataset(config)
    if dataset is None:
        return None
    
    # Cargar modelo preentrenado
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            config['model']['name'],
            num_labels=len(label_names),
            id2label={i: label for i, label in enumerate(label_names)},
            label2id={label: i for i, label in enumerate(label_names)}
        )
        print(f"Modelo cargado: {config['model']['name']}")
    except Exception as e:
        print(f"Error cargando modelo {config['model']['name']}: {e}")
        print("Usando modelo alternativo...")
        model = AutoModelForTokenClassification.from_pretrained(
            "bert-base-multilingual-cased",
            num_labels=len(label_names),
            id2label={i: label for i, label in enumerate(label_names)},
            label2id={label: i for i, label in enumerate(label_names)}
        )
    
    # Configurar argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        eval_steps=config['training']['eval_steps'],
        save_total_limit=config['training']['save_total_limit'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        greater_is_better=config['training']['greater_is_better'],
        push_to_hub=False,
        report_to="wandb" if config['logging']['use_wandb'] else None,
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True
    )
    
    # Verificar splits disponibles para entrenamiento y evaluación
    available_splits = list(dataset.keys())
    train_split = config['dataset']['train_split']
    val_split = config['dataset']['validation_split']
    
    # Asegurar que tenemos split de entrenamiento
    if train_split not in available_splits:
        train_split = available_splits[0]
        print(f"⚠️ Split de entrenamiento no encontrado, usando: {train_split}")
    
    # Aplicar límites al dataset de entrenamiento
    train_dataset = dataset[train_split]
    max_train_samples = config['dataset'].get('max_train_samples')
    if max_train_samples and len(train_dataset) > max_train_samples:
        print(f"🔢 Limitando dataset de entrenamiento: {len(train_dataset)} → {max_train_samples} ejemplos")
        train_dataset = train_dataset.select(range(max_train_samples))
    else:
        print(f"📊 Dataset de entrenamiento: {len(train_dataset)} ejemplos")
    
    # Verificar split de validación
    eval_dataset = None
    if val_split in available_splits:
        eval_dataset = dataset[val_split]
        print(f"✅ Usando split de validación: {val_split}")
        
        # Aplicar límites al dataset de evaluación
        max_eval_samples = config['dataset'].get('max_eval_samples')
        if max_eval_samples and len(eval_dataset) > max_eval_samples:
            print(f"🔢 Limitando dataset de evaluación: {len(eval_dataset)} → {max_eval_samples} ejemplos")
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        else:
            print(f"📊 Dataset de evaluación: {len(eval_dataset)} ejemplos")
            
    else:
        print(f"⚠️ Split de validación '{val_split}' no encontrado")
        # Dividir el dataset de entrenamiento
        train_size = len(train_dataset)
        if train_size > 100:  # Solo si hay suficientes datos
            val_size = min(int(0.1 * train_size), 200)  # 10% o máximo 200
            print(f"Creando split de validación con {val_size} ejemplos del entrenamiento")
            
            # Crear un split de validación
            eval_dataset = train_dataset.select(range(val_size))
            train_dataset = train_dataset.select(range(val_size, train_size))
            
            print(f"📊 Después de dividir:")
            print(f"   Entrenamiento: {len(train_dataset)} ejemplos")
            print(f"   Evaluación: {len(eval_dataset)} ejemplos")
        else:
            print("Dataset muy pequeño, evaluando en el conjunto de entrenamiento")
            eval_dataset = train_dataset
    
    # Crear trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, label_names),
    )
    
    # Entrenar
    print("Comenzando entrenamiento...")
    trainer.train()
    
    # Guardar modelo final
    print("Guardando modelo...")
    trainer.save_model()
    tokenizer.save_pretrained(config['training']['output_dir'])
    
    print("¡Entrenamiento completado!")
    return trainer

def main():
    """Función principal"""
    config = load_config()
    
    # Crear directorios necesarios
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    
    # Entrenar modelo
    trainer = train_model(config)
    
    if trainer:
        print("Modelo entrenado exitosamente!")
    else:
        print("Error durante el entrenamiento")

if __name__ == "__main__":
    main()