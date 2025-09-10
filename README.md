# Proyecto de Entrenamiento NER

Este proyecto permite entrenar modelos de Reconocimiento de Entidades Nombradas (NER) usando modelos preentrenados de HuggingFace y datasets públicos.

## Estructura del Proyecto

```
ner-training/
├── requirements.txt      # Dependencias del proyecto
├── setup.py             # Configuración de instalación
├── Makefile             # Comandos útiles
├── config.yaml          # Configuración del proyecto
├── data/                # Datasets descargados
├── models/              # Modelos entrenados
├── utils/               # Funciones auxiliares
│   ├── data_utils.py    # Utilidades para datos
│   └── model_utils.py   # Utilidades para modelos
└── src/                 # Código principal
    ├── download_data.py # Descargar datasets
    ├── train.py         # Entrenar modelos
    └── evaluate.py      # Evaluar modelos
```

## Instalación

### 1. Clonar el repositorio
```bash
git clone <tu-repositorio>
cd ner-training
```

### 2. Crear entorno virtual (recomendado)
```bash
make setup-venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

### 3. Instalar dependencias
```bash
make install
# o
pip install -r requirements.txt
```

## Uso Rápido

### 1. Descargar dataset
```bash
make download-data
```

### 2. Entrenar modelo
```bash
make train
```

### 3. Evaluar modelo
```bash
make evaluate
```

## Configuración

Edita el archivo `config.yaml` para personalizar:

- **Modelo base**: Cambia `model.name` por cualquier modelo de HuggingFace
- **Dataset**: Cambia `dataset.name` por el dataset que quieras usar
- **Hiperparámetros**: Ajusta learning rate, batch size, epochs, etc.
- **Rutas**: Modifica las rutas de datos y modelos

### Ejemplos de modelos populares para español:
- `PlanTL-GOB-ES/roberta-base-bne` (recomendado)
- `dccuchile/bert-base-spanish-wwm-cased`
- `BSC-TeMU/roberta-base-bne`
- `bert-base-multilingual-cased`

### Ejemplos de datasets populares:
- `PlanTL-GOB-ES/CoNLL-NERC-es` (español - recomendado)
- `conll2003` (inglés)
- `conll2002` (español, holandés)  
- `wikiner_en`
- `wikiann` (multiidioma)

## Comandos Disponibles

```bash
make help              # Mostrar todos los comandos
make check             # Verificar instalación
make install           # Instalar dependencias
make download-data     # Descargar dataset
make train             # Entrenar modelo
make evaluate          # Evaluar modelo (test_model.py)
make clean             # Limpiar archivos temporales
make clean-all         # Limpiar todo (datos, modelos, temp)
```

## Scripts Adicionales

```bash
python test_simple.py     # Probar que todo funciona
python quick_start.py     # Pipeline completo automatizado
python src/test_model.py  # Evaluar modelo entrenado
```

## Personalización

### Cambiar el dataset
1. Modifica `dataset.name` en `config.yaml`
2. Ajusta `dataset.text_column` y `dataset.label_column` según el formato
3. Ejecuta `make download-data`

### Cambiar el modelo base
1. Modifica `model.name` en `config.yaml`
2. Ajusta `model.max_length` si es necesario
3. Ejecuta `make train`

### Ajustar hiperparámetros
Modifica la sección `training` en `config.yaml`:
- `num_train_epochs`: Número de épocas
- `learning_rate`: Tasa de aprendizaje
- `per_device_train_batch_size`: Tamaño del batch
- `warmup_steps`: Pasos de calentamiento

## Uso Avanzado

### Entrenar con Weights & Biases
1. Instala wandb: `pip install wandb`
2. Configura: `wandb login`
3. En `config.yaml` cambia `logging.use_wandb: true`

### Usar GPU
El proyecto detecta automáticamente si CUDA está disponible. Para usar GPU:
1. Instala PyTorch con soporte CUDA
2. El entrenamiento usará GPU automáticamente

### Evaluar con texto personalizado
Edita `src/evaluate.py` y modifica la lista `test_texts` con tus propios ejemplos.

## Solución de Problemas

### Error de memoria
- Reduce `per_device_train_batch_size` en `config.yaml`
- Reduce `model.max_length`

### Dataset no encontrado
- Verifica que el nombre del dataset es correcto en HuggingFace
- Ejecuta `make download-data` primero

### Modelo no carga
- Verifica que el modelo existe en HuggingFace
- Revisa que `model.name` en `config.yaml` es correcto

## Contribuir

1. Fork del proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## Licencia

[Especifica tu licencia aquí]