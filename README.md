# Proyecto Deep Learning — Pothole Detection

Proyecto modular en PyTorch para clasificar imágenes de vías como `pothole` o `no pothole` usando transfer learning con InceptionV3.

## Estructura

```text
proyecto_deep/
├── pothole_detection.ipynb
├── requirements.txt
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

## Uso en Google Colab

1. Sube esta carpeta a GitHub o súbela como ZIP a Colab.
2. Si usas GitHub:

```python
%cd /content
!git clone https://github.com/TU_USUARIO/proyecto_deep.git
%cd /content/proyecto_deep
!pip install -r requirements.txt
```

3. Si subes el ZIP manualmente:

```python
%cd /content
!unzip proyecto_deep_arreglado.zip -d .
%cd /content/proyecto_deep_arreglado
!pip install -r requirements.txt
```

4. En Colab activa GPU:

```text
Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4
```

## Archivos principales

- `src/data_loaders.py`: carga dataset, EDA y `DataLoader`.
- `src/models.py`: modelos CNN con transfer learning.
- `src/model_training.py`: entrenamiento, validación y métricas.
- `src/utils.py`: visualizaciones.
- `src/xai.py`: Grad-CAM.
- `pothole_detection.ipynb`: notebook principal para ejecutar todo el flujo.

## Dataset

Dataset usado: `taroii/pothole-detection` desde Hugging Face.

## Flujo del notebook

1. Configuración del entorno.
2. Carga del dataset.
3. EDA: distribución de clases, muestras, dimensiones e intensidad de píxeles.
4. Preprocesamiento y DataLoaders.
5. Transfer learning con InceptionV3.
6. Fase 1: feature extraction.
7. Fase 2: fine-tuning.
8. Evaluación final con matriz de confusión y classification report.
9. Visualización de predicciones.
10. Grad-CAM para explicabilidad.
