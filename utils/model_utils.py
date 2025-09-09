"""
Utilidades para modelos
"""

import numpy as np
import evaluate
from seqeval.metrics import classification_report, f1_score

# Cargar métrica seqeval
seqeval = evaluate.load("seqeval")

def compute_metrics(eval_pred, label_names):
    """
    Calcular métricas para evaluación del modelo NER
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # Remover tokens ignorados (label -100)
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    true_labels = [
        [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    # Calcular métricas usando seqeval
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"], 
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def print_model_info(model):
    """
    Mostrar información del modelo
    """
    print("Información del modelo:")
    print(f"Nombre: {model.config.name_or_path}")
    print(f"Número de parámetros: {model.num_parameters():,}")
    print(f"Número de labels: {model.config.num_labels}")
    print(f"Labels: {list(model.config.id2label.values())}")

def save_model_summary(model, tokenizer, save_path):
    """
    Guardar resumen del modelo entrenado
    """
    import json
    
    summary = {
        "model_name": model.config.name_or_path,
        "num_parameters": model.num_parameters(),
        "num_labels": model.config.num_labels,
        "id2label": model.config.id2label,
        "label2id": model.config.label2id,
        "vocab_size": tokenizer.vocab_size,
        "max_position_embeddings": getattr(model.config, 'max_position_embeddings', 'N/A')
    }
    
    with open(f"{save_path}/model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Resumen del modelo guardado en: {save_path}/model_summary.json")

def load_trained_model(model_path):
    """
    Cargar modelo entrenado desde disco
    """
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        
        print(f"Modelo cargado desde: {model_path}")
        print_model_info(model)
        
        return model, tokenizer
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return None, None

def predict_entities(text, model, tokenizer, threshold=0.5):
    """
    Predecir entidades en texto usando modelo entrenado
    """
    from transformers import pipeline
    
    # Crear pipeline
    ner_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )
    
    # Hacer predicciones
    predictions = ner_pipeline(text)
    
    # Filtrar por threshold
    filtered_predictions = [
        pred for pred in predictions 
        if pred['score'] >= threshold
    ]
    
    return filtered_predictions