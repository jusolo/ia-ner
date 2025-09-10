#!/usr/bin/env python3
"""
Script simple para probar que todo funciona
"""

def test_imports():
    """Probar que todas las importaciones funcionan"""
    print("🧪 Probando importaciones...")
    
    try:
        import torch
        print("✅ PyTorch")
        
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        print("✅ Transformers")
        
        from datasets import Dataset
        print("✅ Datasets")
        
        import yaml
        print("✅ PyYAML")
        
        import os
        import sys
        sys.path.append('.')
        
        from utils.data_utils import get_label_names
        print("✅ Utils")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en importaciones: {e}")
        return False

def test_model_loading():
    """Probar carga de modelo"""
    print("\n🤖 Probando carga de modelo...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        
        model_name = "bert-base-multilingual-cased"
        print(f"Cargando: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Tokenizer cargado")
        
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, 
            num_labels=9
        )
        print("✅ Modelo cargado")
        
        return True, tokenizer, model
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return False, None, None

def test_dataset_creation():
    """Probar creación de dataset simple"""
    print("\n📊 Probando creación de dataset...")
    
    try:
        from datasets import Dataset, DatasetDict
        
        # Datos de ejemplo
        sample_data = {
            'tokens': [
                ['Juan', 'vive', 'en', 'Madrid'],
                ['Apple', 'es', 'una', 'empresa'],
                ['María', 'trabaja', 'en', 'Google']
            ],
            'ner_tags': [
                [1, 0, 0, 5],  # B-PER, O, O, B-LOC
                [3, 0, 0, 0],  # B-ORG, O, O, O
                [1, 0, 0, 3]   # B-PER, O, O, B-ORG
            ]
        }
        
        dataset = Dataset.from_dict(sample_data)
        print(f"✅ Dataset creado con {len(dataset)} ejemplos")
        
        return True, dataset
        
    except Exception as e:
        print(f"❌ Error creando dataset: {e}")
        return False, None

def test_tokenization():
    """Probar tokenización"""
    print("\n🔤 Probando tokenización...")
    
    try:
        success, tokenizer, model = test_model_loading()
        if not success:
            return False
        
        success, dataset = test_dataset_creation()
        if not success:
            return False
        
        # Probar tokenización simple
        sample_tokens = ['Juan', 'vive', 'en', 'Madrid']
        
        tokenized = tokenizer(
            sample_tokens,
            is_split_into_words=True,
            truncation=True,
            padding=True,
            max_length=128
        )
        
        print("✅ Tokenización exitosa")
        print(f"   Input tokens: {len(sample_tokens)}")
        print(f"   Tokenized: {len(tokenized['input_ids'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en tokenización: {e}")
        return False

def test_simple_training():
    """Probar un mini-entrenamiento"""
    print("\n🏋️ Probando mini-entrenamiento...")
    
    try:
        from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification
        import sys
        sys.path.append('.')
        from utils.data_utils import tokenize_and_align_labels
        
        success, tokenizer, model = test_model_loading()
        if not success:
            return False
            
        success, dataset = test_dataset_creation() 
        if not success:
            return False
        
        # Configuración simple para tokenización
        config = {
            'dataset': {
                'text_column': 'tokens',
                'label_column': 'ner_tags'
            },
            'model': {
                'max_length': 128
            }
        }
        
        # Tokenizar dataset
        def tokenize_function(examples):
            return tokenize_and_align_labels(examples, tokenizer, config)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        print("✅ Dataset tokenizado")
        
        # Configuración de entrenamiento mínima
        training_args = TrainingArguments(
            output_dir="./test_output",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            save_steps=1000,
            logging_steps=100,
            remove_unused_columns=False
        )
        
        data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            padding=True
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        
        print("✅ Trainer configurado")
        print("⚠️  Entrenamiento real omitido en prueba")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en configuración de entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    print("🚀 PRUEBAS SIMPLES DEL SISTEMA")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 4
    
    if test_imports():
        tests_passed += 1
    
    if test_model_loading()[0]:
        tests_passed += 1
    
    if test_dataset_creation()[0]:
        tests_passed += 1
    
    if test_simple_training():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 RESULTADOS: {tests_passed}/{total_tests} pruebas pasaron")
    
    if tests_passed == total_tests:
        print("🎉 ¡Todas las pruebas pasaron!")
        print("El sistema está listo para entrenar modelos NER")
    else:
        print("⚠️  Algunas pruebas fallaron")
        print("Revisa los errores arriba")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)