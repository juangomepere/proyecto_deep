# Proyecto Deep Learning — Detección de Huecos en Vías con Transfer Learning

## 1. Descripción general

Este proyecto implementa un sistema de clasificación de imágenes para detectar la presencia de huecos en vías a partir de fotografías de carreteras. El objetivo principal es construir un modelo de Deep Learning capaz de clasificar una imagen en una de dos clases:

```text
pothole
no pothole
```

El proyecto usa una arquitectura modular basada en PyTorch. La lógica está separada en archivos especializados para carga de datos, definición del modelo, entrenamiento, evaluación, visualización y explicabilidad. Esto permite que el notebook principal sea más limpio, mantenible y fácil de presentar.

El flujo general del proyecto es:

```text
Dataset Hugging Face
        ↓
EDA: análisis exploratorio de datos
        ↓
Preprocesamiento y data augmentation
        ↓
DataLoaders de PyTorch
        ↓
Modelo CNN con Transfer Learning
        ↓
Fase 1: Feature Extraction
        ↓
Fase 2: Fine-tuning
        ↓
Evaluación final
        ↓
Matriz de confusión, métricas, predicciones y Grad-CAM
```

El problema tiene una aplicación directa en mantenimiento vial, inspección urbana, priorización de reparaciones y ciudades inteligentes. Un sistema de este tipo podría integrarse con cámaras en vehículos, aplicaciones móviles o plataformas de monitoreo urbano.

---

## 2. Objetivos

### 2.1 Objetivo general

Desarrollar un modelo de visión por computador basado en redes neuronales convolucionales para clasificar imágenes de vías según la presencia o ausencia de huecos.

### 2.2 Objetivos específicos

- Cargar y analizar un dataset de imágenes de huecos en vías.
- Realizar análisis exploratorio de datos, incluyendo distribución de clases, visualización de imágenes, dimensiones e intensidad de píxeles.
- Preprocesar imágenes para que sean compatibles con una arquitectura CNN preentrenada.
- Implementar Transfer Learning usando InceptionV3.
- Entrenar el modelo en dos fases: feature extraction y fine-tuning.
- Evaluar el modelo usando accuracy, precision, recall, F1-score y matriz de confusión.
- Visualizar predicciones correctas e incorrectas.
- Aplicar Grad-CAM para interpretar qué regiones de la imagen influyen en las predicciones.
- Analizar implicaciones técnicas, prácticas y éticas de una posible implementación real.

---

## 3. Dataset

### 3.1 Fuente del dataset

El dataset utilizado es:

```text
taroii/pothole-detection
```

Este dataset está disponible en Hugging Face Datasets y contiene imágenes etiquetadas de vías con y sin huecos.

### 3.2 Tipo de problema

Aunque el nombre del dataset contiene la palabra `detection`, en este proyecto se formula como un problema de clasificación binaria de imágenes.

Esto significa que el modelo no predice la ubicación exacta del hueco con bounding boxes. En cambio, recibe una imagen completa y predice una clase:

```text
Imagen completa → pothole / no pothole
```

### 3.3 Justificación del dataset

El dataset es adecuado para el proyecto porque:

- Representa un problema del mundo real.
- Contiene imágenes etiquetadas.
- Permite entrenar una CNN para clasificación binaria.
- Permite realizar análisis de balanceo de clases.
- Es visualmente interpretable.
- Tiene una aplicación práctica clara en mantenimiento vial.
- Permite discutir sesgos, limitaciones, privacidad, escalabilidad y errores críticos.

---

## 4. Estructura del proyecto

La estructura del proyecto es:

```text
proyecto_deep_arreglado/
├── pothole_detection.ipynb
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── data_loaders.py
│   ├── model_training.py
│   ├── models.py
│   ├── utils.py
│   └── xai.py
├── models/
└── output/
```

La lógica principal se divide entre el notebook y los módulos de `src/`:

```text
pothole_detection.ipynb
        │
        ├── src/data_loaders.py      → carga y preprocesamiento de datos
        ├── src/models.py            → arquitectura CNN y transfer learning
        ├── src/model_training.py    → entrenamiento y evaluación
        ├── src/utils.py             → visualizaciones y utilidades
        └── src/xai.py               → Grad-CAM y explicabilidad
```

---

## 5. Descripción técnica de archivos

## 5.1 `pothole_detection.ipynb`

Este es el notebook principal del proyecto. Su función es coordinar todo el flujo experimental.

Contiene las siguientes etapas:

```text
1. Configuración del entorno
2. Importación de módulos
3. Carga del dataset
4. Análisis exploratorio de datos
5. Preprocesamiento
6. Creación de DataLoaders
7. Construcción del modelo
8. Entrenamiento fase 1
9. Entrenamiento fase 2
10. Evaluación final
11. Visualización de resultados
12. Grad-CAM
```

El notebook importa funciones desde `src/` para evitar duplicación de código y mejorar la organización.

Ejemplo:

```python
from src.data_loaders import create_dataloaders
from src.models import build_inception_v3, unfreeze_last_layers
from src.model_training import train_model, evaluate
from src.utils import plot_training_curves, plot_confusion_matrix, show_predictions
```

El notebook es el archivo que debe ejecutarse en Google Colab o Jupyter.

---

## 5.2 `requirements.txt`

Este archivo contiene las librerías necesarias para ejecutar el proyecto.

Ejemplo de dependencias:

```text
torch
torchvision
datasets
huggingface_hub
numpy
pandas
matplotlib
scikit-learn
pillow
opencv-python
tqdm
```

Para instalarlas:

```bash
pip install -r requirements.txt
```

### Dependencias principales

| Librería | Uso |
|---|---|
| `torch` | Entrenamiento del modelo con PyTorch |
| `torchvision` | Modelos preentrenados y transformaciones |
| `datasets` | Carga del dataset desde Hugging Face |
| `numpy` | Operaciones numéricas |
| `pandas` | Análisis tabular |
| `matplotlib` | Visualizaciones |
| `scikit-learn` | Métricas de clasificación |
| `Pillow` | Manejo de imágenes |
| `opencv-python` | Procesamiento de imágenes y Grad-CAM |
| `tqdm` | Barras de progreso |

---

## 5.3 `README.md`

Este archivo documenta el proyecto. Explica:

- Qué problema se resuelve.
- Qué dataset se usa.
- Cómo está organizada la carpeta.
- Cómo funciona cada módulo.
- Cómo ejecutar el proyecto.
- Cómo se entrena y evalúa el modelo.
- Qué implicaciones prácticas y éticas tiene.

El README es importante porque permite que otra persona entienda el proyecto sin necesidad de leer todo el notebook celda por celda.

---

## 5.4 `src/__init__.py`

Este archivo puede estar vacío. Su función es permitir que Python trate la carpeta `src/` como un paquete importable.

Gracias a este archivo se pueden usar imports como:

```python
from src.models import build_inception_v3
from src.model_training import train_model
```

Sin `__init__.py`, algunos entornos pueden fallar al reconocer `src` como un módulo válido.

---

## 5.5 `src/data_loaders.py`

Este archivo contiene la lógica de carga, transformación y preparación de datos.

### Responsabilidades principales

- Descargar el dataset desde Hugging Face.
- Detectar la columna de imagen.
- Detectar la columna de etiqueta.
- Definir transformaciones para entrenamiento, validación y prueba.
- Crear objetos `Dataset` compatibles con PyTorch.
- Crear `DataLoader`s para entrenamiento, validación y test.

### Flujo interno

```text
Hugging Face Dataset
        ↓
Detectar columna de imagen y etiqueta
        ↓
Aplicar transformaciones
        ↓
Crear Dataset personalizado
        ↓
Crear DataLoader
        ↓
train_loader, val_loader, test_loader
```

### Clase `HuggingFaceImageDataset`

Esta clase adapta el dataset de Hugging Face al formato esperado por PyTorch.

En PyTorch, un dataset debe implementar dos métodos principales:

```python
__len__()
__getitem__(idx)
```

- `__len__()` devuelve el número de ejemplos.
- `__getitem__(idx)` devuelve una imagen y su etiqueta.

La clase toma una imagen, la convierte a RGB, aplica transformaciones y devuelve:

```python
image_tensor, label
```

### Transformaciones

Para entrenamiento se aplican transformaciones con data augmentation:

```text
Resize
RandomHorizontalFlip
RandomRotation
ColorJitter
ToTensor
Normalize
```

Para validación y test se usan transformaciones determinísticas:

```text
Resize
ToTensor
Normalize
```

La diferencia es importante: en entrenamiento se modifica artificialmente la imagen para mejorar la generalización, mientras que en validación y test se evalúa el modelo de forma estable.

---

## 5.6 `src/models.py`

Este archivo define la arquitectura del modelo y funciones relacionadas con transfer learning.

### Funciones principales

```python
build_inception_v3()
build_resnet18()
unfreeze_last_layers()
count_trainable_parameters()
```

### `build_inception_v3()`

Construye un modelo InceptionV3 preentrenado.

InceptionV3 es una red neuronal convolucional profunda entrenada previamente en un dataset grande de imágenes. En este proyecto se reutiliza como extractor de características visuales.

El proceso es:

```text
InceptionV3 preentrenado
        ↓
Congelar pesos del backbone
        ↓
Reemplazar capa final
        ↓
Nueva salida de 2 clases
```

La capa final original de InceptionV3 se reemplaza por una nueva capa compatible con clasificación binaria:

```python
model.fc = nn.Sequential(
    nn.Dropout(dropout),
    nn.Linear(in_features, num_classes)
)
```

Esto permite adaptar el modelo a:

```text
pothole / no pothole
```

### Feature extraction

En la fase de feature extraction, la mayoría de los pesos del modelo se congelan:

```python
param.requires_grad = False
```

Esto significa que el modelo usa las características aprendidas previamente y solo entrena la cabeza clasificadora final.

### Fine-tuning

En la fase de fine-tuning, se descongelan algunas capas finales del modelo:

```python
unfreeze_last_layers(model, num_layers=40)
```

Esto permite que el modelo ajuste representaciones más específicas al problema de huecos en vías.

### `count_trainable_parameters()`

Cuenta los parámetros entrenables del modelo. Esto sirve para comparar la complejidad entre fases:

```text
Fase 1 → pocos parámetros entrenables
Fase 2 → más parámetros entrenables
```

---

## 5.7 `src/model_training.py`

Este archivo contiene la lógica de entrenamiento, validación y evaluación.

### Funciones principales

```python
train_one_epoch()
validate_one_epoch()
train_model()
evaluate()
```

### `train_one_epoch()`

Entrena el modelo durante una época.

Flujo:

```text
Batch de imágenes
        ↓
Enviar imágenes al device
        ↓
Forward pass
        ↓
Calcular loss
        ↓
Backward pass
        ↓
Actualizar pesos
        ↓
Calcular accuracy
```

En código, el proceso clave es:

```python
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

### `validate_one_epoch()`

Evalúa el modelo en el conjunto de validación.

A diferencia del entrenamiento, no actualiza pesos. Por eso usa:

```python
with torch.no_grad():
```

Esto reduce consumo de memoria y evita cálculo de gradientes innecesarios.

### `train_model()`

Es la función principal de entrenamiento.

Por cada época:

```text
Entrena en train_loader
        ↓
Valida en val_loader
        ↓
Guarda loss y accuracy
        ↓
Compara con mejor validación
        ↓
Guarda el mejor modelo
```

El mejor modelo se guarda en:

```text
models/{model_name}_best.pth
```

Esto evita quedarse con el modelo de la última época si hubo sobreajuste.

### `evaluate()`

Evalúa el modelo en el conjunto de test.

Calcula:

- Accuracy.
- Precision weighted.
- Recall weighted.
- F1-score weighted.
- Matriz de confusión.
- Classification report.
- Etiquetas reales.
- Predicciones.
- Probabilidades por clase.

Devuelve un diccionario con esta estructura:

```python
metrics = {
    "accuracy": accuracy,
    "precision_weighted": precision,
    "recall_weighted": recall,
    "f1_weighted": f1,
    "confusion_matrix": cm,
    "classification_report": report,
    "y_true": y_true,
    "y_pred": y_pred,
    "y_prob": y_prob
}
```

---

## 5.8 `src/utils.py`

Este archivo contiene funciones auxiliares de visualización.

### Funciones principales

```python
unnormalize()
plot_training_curves()
plot_confusion_matrix()
show_batch()
show_predictions()
```

### `unnormalize()`

Durante el preprocesamiento, las imágenes se normalizan usando medias y desviaciones estándar de ImageNet:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

Esto ayuda al modelo preentrenado porque InceptionV3 fue entrenado con esa normalización.

Sin embargo, una imagen normalizada no se ve correctamente al graficarla. `unnormalize()` revierte esa transformación para visualizar la imagen.

### `plot_training_curves()`

Grafica dos curvas:

```text
Train Loss vs Validation Loss
Train Accuracy vs Validation Accuracy
```

Estas curvas ayudan a detectar:

- subentrenamiento;
- sobreajuste;
- estabilidad del entrenamiento;
- diferencia entre entrenamiento y validación.

### `plot_confusion_matrix()`

Grafica la matriz de confusión.

Para este problema, la matriz tiene la forma:

```text
                   Pred: no pothole    Pred: pothole
Real: no pothole          TN                FP
Real: pothole             FN                TP
```

Donde:

- `TN`: imágenes sin hueco clasificadas correctamente.
- `FP`: imágenes sin hueco clasificadas como hueco.
- `FN`: imágenes con hueco clasificadas como no hueco.
- `TP`: imágenes con hueco clasificadas correctamente.

En este problema, los falsos negativos son especialmente importantes porque significan que el sistema no detectó un hueco real.

### `show_batch()`

Muestra imágenes de un batch del DataLoader junto con sus etiquetas reales.

Sirve para verificar que:

- el DataLoader funciona;
- las etiquetas corresponden a las imágenes;
- las transformaciones no dañan las imágenes.

### `show_predictions()`

Muestra imágenes de test junto con:

```text
Etiqueta real
Predicción del modelo
Confianza de la predicción
```

Esto ayuda a interpretar visualmente el comportamiento del modelo.

---

## 5.9 `src/xai.py`

Este archivo implementa funciones de explicabilidad usando Grad-CAM.

### ¿Qué es Grad-CAM?

Grad-CAM significa Gradient-weighted Class Activation Mapping.

Es una técnica que genera un mapa de calor para visualizar qué regiones de la imagen fueron más relevantes para la predicción del modelo.

Flujo conceptual:

```text
Imagen de entrada
        ↓
Forward pass
        ↓
Predicción del modelo
        ↓
Backward pass sobre la clase objetivo
        ↓
Gradientes de la capa convolucional
        ↓
Mapa de activación ponderado
        ↓
Heatmap sobre la imagen original
```

### ¿Por qué es útil?

Permite responder preguntas como:

```text
¿El modelo se fijó realmente en el hueco?
¿O se fijó en sombras, bordes, marcas o fondo?
```

Esto es relevante para:

- interpretabilidad;
- análisis de errores;
- confianza en el modelo;
- transparencia;
- discusión ética.

### Funciones principales

```python
GradCAM
resize_cam()
overlay_cam()
tensor_to_uint8()
```

### `GradCAM`

Clase que registra activaciones y gradientes de una capa convolucional objetivo.

Utiliza hooks de PyTorch:

```python
register_forward_hook
register_full_backward_hook
```

Los hooks permiten capturar información interna del modelo durante el forward y backward pass.

### `resize_cam()`

Redimensiona el mapa Grad-CAM al tamaño de la imagen original.

### `overlay_cam()`

Combina el mapa de calor con la imagen original para crear una visualización interpretable.

### `tensor_to_uint8()`

Convierte un tensor normalizado de PyTorch a una imagen en formato `uint8`, útil para visualización con OpenCV o matplotlib.

---

## 5.10 Carpeta `models/`

Esta carpeta almacena los pesos de los modelos entrenados.

Ejemplos:

```text
pothole_inception_phase1_best.pth
pothole_inception_phase2_best.pth
```

Los archivos `.pth` contienen los pesos aprendidos por el modelo.

Durante el entrenamiento, `train_model()` guarda automáticamente el mejor modelo según accuracy de validación.

---

## 5.11 Carpeta `output/`

Esta carpeta está destinada a guardar resultados generados durante el proyecto, como:

- gráficas de entrenamiento;
- matrices de confusión;
- predicciones visuales;
- imágenes Grad-CAM;
- reportes o resultados exportados.

Aunque muchas visualizaciones se muestran directamente en el notebook, esta carpeta permite conservar resultados para la presentación o informe.

---

## 6. Arquitectura del modelo

El modelo principal es InceptionV3 con Transfer Learning.

### 6.1 Transfer Learning

Transfer Learning consiste en reutilizar un modelo previamente entrenado en una tarea grande y adaptarlo a una tarea nueva.

En este proyecto:

```text
Modelo original: InceptionV3 entrenado en ImageNet
Nueva tarea: clasificar imágenes de vías como pothole o no pothole
```

### 6.2 Diagrama de arquitectura

```text
Imagen RGB
299 × 299 × 3
        ↓
Preprocesamiento
Resize + Normalize
        ↓
Backbone InceptionV3 preentrenado
Extracción de características visuales
        ↓
Global feature representation
        ↓
Dropout
        ↓
Linear Layer
        ↓
Logits de salida
        ↓
Softmax
        ↓
pothole / no pothole
```

### 6.3 Fase 1: Feature Extraction

En esta fase se congelan los pesos del backbone:

```text
Backbone InceptionV3 congelado
Clasificador final entrenable
```

Objetivo:

- entrenar rápidamente;
- evitar sobreajuste;
- aprovechar características generales aprendidas en ImageNet.

### 6.4 Fase 2: Fine-tuning

En esta fase se descongelan algunas capas finales:

```text
Primeras capas congeladas
Últimas capas entrenables
Clasificador final entrenable
```

Objetivo:

- adaptar características profundas al dominio de carreteras;
- mejorar la capacidad del modelo para reconocer huecos;
- refinar representaciones visuales específicas.

Se usa un learning rate menor para evitar destruir los pesos preentrenados.

---

## 7. Preprocesamiento

Las imágenes se procesan antes de entrar al modelo.

### 7.1 Resize

InceptionV3 requiere imágenes de tamaño fijo. En este proyecto se usa:

```text
299 × 299 píxeles
```

### 7.2 Conversión a tensor

Las imágenes PIL se convierten a tensores de PyTorch.

```text
PIL Image → Tensor [C, H, W]
```

### 7.3 Normalización

Se usa la normalización estándar de ImageNet:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

Esto es importante porque el modelo preentrenado espera imágenes con esa distribución.

### 7.4 Data augmentation

En entrenamiento se aplica aumento de datos:

```text
RandomHorizontalFlip
RandomRotation
ColorJitter
```

Esto ayuda a mejorar la generalización del modelo y reduce el riesgo de sobreajuste.

---

## 8. Entrenamiento

El entrenamiento se divide en dos fases.

### 8.1 Fase 1: Feature Extraction

```text
Entrenar solo la cabeza clasificadora
```

Configuración típica:

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
```

### 8.2 Fase 2: Fine-tuning

```text
Descongelar algunas capas finales del modelo
Entrenar con learning rate bajo
```

Configuración típica:

```python
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
```

### 8.3 Función de pérdida

Se usa:

```python
nn.CrossEntropyLoss()
```

Esta función es adecuada para clasificación multiclase o binaria con salida de logits.

### 8.4 Optimizador

Se usa Adam porque suele converger bien en problemas de transfer learning.

---

## 9. Evaluación

El modelo se evalúa usando múltiples métricas.

### 9.1 Accuracy

Proporción total de predicciones correctas.

```text
accuracy = predicciones correctas / total de predicciones
```

### 9.2 Precision

Indica qué tan confiables son las predicciones positivas.

```text
precision = TP / (TP + FP)
```

### 9.3 Recall

Indica cuántos casos reales positivos fueron detectados.

```text
recall = TP / (TP + FN)
```

En este proyecto, el recall de la clase `pothole` es especialmente importante porque un falso negativo implica no detectar un hueco real.

### 9.4 F1-score

Media armónica entre precision y recall.

```text
F1 = 2 × (precision × recall) / (precision + recall)
```

### 9.5 Matriz de confusión

Permite analizar los errores del modelo por clase.

```text
                   Pred: no pothole    Pred: pothole
Real: no pothole          TN                FP
Real: pothole             FN                TP
```

---

## 10. Explicabilidad con Grad-CAM

Grad-CAM se utiliza para generar mapas de calor que muestran las regiones relevantes para la predicción del modelo.

Ejemplo conceptual:

```text
Imagen original
        ↓
Modelo predice: pothole
        ↓
Grad-CAM genera heatmap
        ↓
Se verifica si el modelo miró el hueco real
```

Esto permite analizar:

- si el modelo se enfoca en el hueco;
- si confunde sombras con huecos;
- si depende demasiado del fondo;
- si hay sesgos visuales en el dataset.

---

## 11. Ejecución en Google Colab

### 11.1 Clonar repositorio

```python
%cd /content
!git clone https://github.com/USUARIO/NOMBRE_REPO.git
%cd /content/NOMBRE_REPO
```

### 11.2 Instalar dependencias

```python
!pip install -r requirements.txt
```

### 11.3 Verificar estructura

```python
import os
print(os.getcwd())
print(os.listdir())
print(os.listdir("src"))
```

### 11.4 Agregar raíz al path

```python
import sys
import os

project_root = os.getcwd()

if project_root not in sys.path:
    sys.path.insert(0, project_root)
```

### 11.5 Importar módulos

```python
from src.data_loaders import create_dataloaders
from src.models import build_inception_v3
from src.model_training import train_model, evaluate
```

---

## 12. Uso de GPU

En Google Colab:

```text
Runtime → Change runtime type → Hardware accelerator → T4 GPU
```

Luego verificar:

```python
import torch

print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

Selección del dispositivo:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## 13. Consideraciones prácticas

Un sistema real de detección de huecos podría implementarse como:

```text
Cámara o app móvil
        ↓
API de inferencia
        ↓
Modelo CNN
        ↓
Base de datos de reportes
        ↓
Dashboard municipal
        ↓
Priorización de mantenimiento
```

### Servicios necesarios

- Captura de imágenes.
- API backend para inferencia.
- Servidor cloud o edge device.
- Base de datos para almacenar reportes.
- Dashboard para visualización.
- Sistema de monitoreo del modelo.
- Proceso de reentrenamiento con nuevos datos.

---

## 14. Consideraciones éticas

### 14.1 Sesgos en los datos

El dataset puede no representar todas las condiciones reales:

- diferentes ciudades;
- tipos de pavimento;
- iluminación;
- lluvia;
- sombras;
- cámaras de distinta calidad.

Mitigación:

- recolectar datos más diversos;
- validar en imágenes locales;
- monitorear errores por contexto.

### 14.2 Privacidad

Si el sistema se implementa con cámaras urbanas o móviles, podrían capturarse:

- rostros;
- placas de vehículos;
- ubicaciones sensibles.

Mitigación:

- anonimización automática;
- difuminado de rostros y placas;
- almacenamiento mínimo necesario;
- políticas claras de uso de datos.

### 14.3 Impacto social

El sistema podría influir en qué zonas reciben mantenimiento primero.

Riesgo:

- priorizar zonas con más imágenes disponibles;
- ignorar barrios con menor cobertura tecnológica.

Mitigación:

- combinar el modelo con reportes ciudadanos;
- auditar cobertura geográfica;
- incluir criterios de equidad.

### 14.4 Seguridad

Un falso negativo puede hacer que un hueco peligroso no sea reportado.

Mitigación:

- mantener revisión humana;
- usar umbrales conservadores;
- reportar incertidumbre;
- reentrenar periódicamente.

### 14.5 Transparencia

Grad-CAM ayuda a explicar por qué el modelo toma una decisión.

Esto permite detectar si el modelo está aprendiendo señales incorrectas, como sombras o marcas de la carretera.

---

## 15. Limitaciones

- El modelo clasifica imágenes completas, no localiza huecos con bounding boxes.
- Puede fallar en imágenes con baja iluminación o lluvia.
- Puede confundir sombras o manchas con huecos.
- El desempeño depende de la calidad y diversidad del dataset.
- No debe usarse como única fuente de decisión para mantenimiento vial.

---

## 16. Posibles mejoras

- Implementar detección de objetos con YOLO o Faster R-CNN.
- Usar segmentación para marcar la zona exacta del hueco.
- Recolectar imágenes locales de diferentes ciudades.
- Comparar varias arquitecturas: ResNet, EfficientNet, MobileNet, InceptionV3.
- Implementar validación cruzada.
- Evaluar desempeño en imágenes reales tomadas por el equipo.
- Medir latencia de inferencia para uso en dispositivos móviles.
- Exportar el modelo a ONNX o TorchScript para despliegue.

---

## 17. Conclusión

Este proyecto implementa una solución completa de Deep Learning para clasificación de imágenes de vías según la presencia de huecos. La estructura modular permite organizar el código de forma profesional, separando carga de datos, modelo, entrenamiento, evaluación y explicabilidad.

El uso de Transfer Learning con InceptionV3 permite aprovechar representaciones visuales aprendidas previamente y adaptarlas al problema específico de detección de huecos. La evaluación con métricas múltiples y matriz de confusión permite analizar el rendimiento más allá del accuracy, mientras que Grad-CAM aporta interpretabilidad para entender qué regiones de la imagen influyen en las predicciones.

Aunque el sistema tiene potencial para apoyar procesos de mantenimiento vial, su implementación real requiere validación adicional, datos más diversos, monitoreo continuo, revisión humana y medidas éticas relacionadas con privacidad, sesgo, equidad y seguridad.
