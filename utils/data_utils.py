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
    Extraer nombres de labels del dataset
    """
    # Para CoNLL-2003, los labels son:
    label_names = [
        "O",       # Outside
        "B-PER",   # Beginning Person
        "I-PER",   # Inside Person  
        "B-ORG",   # Beginning Organization
        "I-ORG",   # Inside Organization
        "B-LOC",   # Beginning Location
        "I-LOC",   # Inside Location
        "B-MISC",  # Beginning Miscellaneous
        "I-MISC"   # Inside Miscellaneous
    ]
    
    # Si el dataset tiene información de features, usar esa
    if hasattr(dataset_split, 'features'):
        if 'ner_tags' in dataset_split.features:
            feature = dataset_split.features['ner_tags']
            if hasattr(feature, 'feature') and hasattr(feature.feature, 'names'):
                return feature.feature.names
    
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