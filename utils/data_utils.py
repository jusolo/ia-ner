"""
Utilidades para manejo de datos
"""

def tokenize_and_align_labels(examples, tokenizer, config):
    """
    Tokenizar texto y alinear labels para NER
    """
    tokenized_inputs = tokenizer(
        examples[config['dataset']['text_column']], 
        truncation=True, 
        padding=True,
        max_length=config['model']['max_length'],
        is_split_into_words=True
    )
    
    labels = []
    for i, label in enumerate(examples[config['dataset']['label_column']]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                # Tokens especiales (CLS, SEP, PAD)
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # Primera subpalabra de una palabra
                label_ids.append(label[word_idx])
            else:
                # Subpalabras subsecuentes
                label_ids.append(-100)
            previous_word_idx = word_idx
            
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def get_label_names(dataset_split):
    """
    Extraer nombres de labels del dataset - versión mejorada
    """
    try:
        # Intentar obtener labels desde las features del dataset
        if hasattr(dataset_split, 'features'):
            # Buscar la columna de labels
            label_columns = ['ner_tags', 'labels', 'tags', 'tag', 'ner']
            for col in label_columns:
                if col in dataset_split.features:
                    feature = dataset_split.features[col]
                    if hasattr(feature, 'feature') and hasattr(feature.feature, 'names'):
                        print(f"Labels extraídos desde features.{col}")
                        return feature.feature.names
        
        # Si no funciona, obtener labels únicos del dataset
        print("Extrayendo labels únicos del dataset...")
        all_labels = set()
        label_columns = ['ner_tags', 'labels', 'tags', 'tag', 'ner']
        
        # Detectar la columna de labels
        available_columns = dataset_split.column_names
        label_column = None
        for col in label_columns:
            if col in available_columns:
                label_column = col
                break
        
        if not label_column:
            print(f"❌ No se encontró columna de labels en {available_columns}")
            # Usar labels por defecto
            return get_default_labels()
        
        print(f"Usando columna de labels: {label_column}")
        
        sample_size = min(1000, len(dataset_split))  # Muestra de 1000 ejemplos
        for i in range(sample_size):
            example = dataset_split[i]
            if label_column in example and example[label_column] is not None:
                labels = example[label_column]
                if isinstance(labels, list):
                    all_labels.update(labels)
                else:
                    all_labels.add(labels)
        
        # Convertir números a labels IOB si es necesario
        if all(isinstance(label, int) for label in all_labels):
            # Para datasets con números, mapear a IOB2
            max_label = max(all_labels)
            print(f"Labels numéricos encontrados: 0-{max_label}")
            
            # Crear mapeo IOB2 estándar sin duplicados
            if max_label <= 8:  # Esquema IOB2 estándar
                id2label = {
                    0: "O",
                    1: "B-PER", 2: "I-PER",
                    3: "B-ORG", 4: "I-ORG", 
                    5: "B-LOC", 6: "I-LOC",
                    7: "B-MISC", 8: "I-MISC"
                }
            else:  # Dataset con más labels
                id2label = {0: "O"}
                entity_types = ['PER', 'ORG', 'LOC', 'MISC']
                label_idx = 1
                
                # Crear pares B- e I- sin duplicados
                for entity_type in entity_types:
                    if label_idx <= max_label:
                        id2label[label_idx] = f"B-{entity_type}"
                        label_idx += 1
                    if label_idx <= max_label:
                        id2label[label_idx] = f"I-{entity_type}"
                        label_idx += 1
                
                # Completar con labels genéricos si quedan
                while label_idx <= max_label:
                    id2label[label_idx] = f"LABEL_{label_idx}"
                    label_idx += 1
            
            # Crear lista sin duplicados manteniendo orden
            label_names = []
            for i in range(max_label + 1):
                label = id2label.get(i, f"LABEL_{i}")
                if label not in label_names:  # Evitar duplicados
                    label_names.append(label)
            
            print(f"Labels mapeados desde números: {label_names}")
            return label_names
        else:
            # Labels ya son strings
            label_names = sorted(list(set(all_labels)))  # Remover duplicados y ordenar
            print(f"Labels como strings: {label_names}")
            return label_names
            
    except Exception as e:
        print(f"Error extrayendo labels: {e}")
        import traceback
        traceback.print_exc()
        
    # Labels por defecto
    return get_default_labels()

def get_default_labels():
    """Obtener labels por defecto para NER español (IOB2)"""
    print("Usando labels por defecto para NER español...")
    label_names = [
        "O",       # Outside
        "B-PER",   # Beginning Person (Persona)
        "I-PER",   # Inside Person  
        "B-ORG",   # Beginning Organization (Organización)
        "I-ORG",   # Inside Organization
        "B-LOC",   # Beginning Location (Localización)
        "I-LOC",   # Inside Location
        "B-MISC",  # Beginning Miscellaneous (Miscelánea)
        "I-MISC"   # Inside Miscellaneous
    ]
    return label_names

def prepare_data_splits(dataset, config):
    """
    Preparar splits de datos para entrenamiento
    """
    train_dataset = dataset[config['dataset']['train_split']]
    val_dataset = dataset[config['dataset']['validation_split']]
    test_dataset = dataset[config['dataset']['test_split']]
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    print(f"Test examples: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset

def preview_data(dataset_split, config, num_examples=3):
    """
    Mostrar vista previa de los datos
    """
    print("Vista previa de los datos:")
    print("-" * 40)
    
    for i in range(min(num_examples, len(dataset_split))):
        example = dataset_split[i]
        tokens = example[config['dataset']['text_column']]
        labels = example[config['dataset']['label_column']]
        
        print(f"\nEjemplo {i+1}:")
        print(f"Tokens: {tokens[:10]}...")  # Primeros 10 tokens
        print(f"Labels: {labels[:10]}...")  # Primeros 10 labels