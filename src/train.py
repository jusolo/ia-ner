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
    print("Preparando dataset...")
    
    # Cargar dataset
    try:
        dataset_path = os.path.join(config['data_paths']['raw_data'], config['dataset']['name'])
        if os.path.exists(dataset_path):
            dataset = load_from_disk(dataset_path)
        else:
            dataset = load_dataset(config['dataset']['name'])
    except Exception as e:
        print(f"Error cargando dataset: {e}")
        return None, None, None
    
    # Obtener nombres de labels
    label_names = get_label_names(dataset[config['dataset']['train_split']])
    num_labels = len(label_names)
    
    print(f"Labels encontrados: {label_names}")
    print(f"Número de labels: {num_labels}")
    
    # Cargar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Tokenizar dataset
    def tokenize_function(examples):
        return tokenize_and_align_labels(examples, tokenizer, config)
    
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=dataset[config['dataset']['train_split']].column_names
    )
    
    return tokenized_dataset, tokenizer, label_names

def train_model(config):
    """Entrenar modelo NER"""
    print("Iniciando entrenamiento...")
    
    # Preparar datos
    dataset, tokenizer, label_names = prepare_dataset(config)
    if dataset is None:
        return None
    
    # Cargar modelo preentrenado
    model = AutoModelForTokenClassification.from_pretrained(
        config['model']['name'],
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
    
    # Crear trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset[config['dataset']['train_split']],
        eval_dataset=dataset[config['dataset']['validation_split']],
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