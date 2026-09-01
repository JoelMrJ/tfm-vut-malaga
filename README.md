# Sistema inteligente de apoyo a la fijación de tarifas en alojamientos turísticos mediante aprendizaje automático geoespacial y RAG

Trabajo Fin de Máster desarrollado en el Máster en Big Data, Data Science e Inteligencia Artificial.

El proyecto propone un sistema de apoyo a la fijación de tarifas para viviendas de uso turístico en Málaga, combinando:

- aprendizaje automático para la estimación de precios,
- variables geoespaciales,
- técnicas de interpretabilidad mediante SHAP,
- y un sistema de asistencia documental basado en Retrieval-Augmented Generation (RAG).

## Arquitectura del proyecto

El sistema se divide en dos componentes principales:

### Motor predictivo

El motor predictivo estima el precio de un alojamiento a partir de variables relacionadas con sus características y localización.

Se compararon distintos modelos de aprendizaje automático y se seleccionó XGBoost como modelo final.

La evaluación final se realizó sobre un holdout espacial reservado, obteniendo aproximadamente:

- R²: 0.53
- RMSE: 57.94 €
- MAE: 33.20 €
- MAPE: 24.05 %

La interpretación de las predicciones se realiza mediante SHAP.

### Motor documental RAG

El segundo componente permite realizar consultas sobre normativa y documentación relacionada con viviendas de uso turístico en Málaga y Andalucía.

La arquitectura combina:

- procesamiento y fragmentación de documentos,
- embeddings,
- base de datos vectorial ChromaDB,
- recuperación documental,
- filtrado de fragmentos,
- y generación de respuestas mediante un modelo de lenguaje ejecutado localmente.

El modelo utilizado en la versión final es Mistral 7B mediante Ollama.

## Estructura del repositorio

```text
.
├── api.py
├── app.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── RAG/
│   └── vectorstore/
├── models/
├── notebooks/
├── reports/
│   ├── modeling/
│   └── rag_v2/
├── src/
├── requirements.txt
└── README.md
```

## Datos

Los datos de alojamientos turísticos utilizados en el proyecto proceden de Inside Airbnb.

Los datasets originales no se incluyen directamente en este repositorio. Los archivos necesarios deben situarse en:

```text
data/raw/
```

Archivos utilizados:

```text
listings.csv.gz
calendar.csv.gz
neighbourhoods.geojson
```

## Corpus documental del RAG

El corpus documental está compuesto por documentación oficial relacionada con viviendas de uso turístico, normativa urbanística y regulación autonómica.

Los documentos originales no se incluyen en el repositorio y deben situarse en:

```text
data/RAG/
```

## Instalación

Se recomienda crear un entorno virtual:

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
```

Instala las dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución

Primero se inicia la API:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

Después, en otra terminal, se inicia la interfaz:

```bash
streamlit run app.py
```

Para utilizar el motor RAG es necesario disponer de Ollama y del modelo Mistral configurado localmente.

## Tecnologías utilizadas

- Python
- Pandas
- GeoPandas
- XGBoost
- SHAP
- H3
- FastAPI
- Streamlit
- ChromaDB
- Ollama
- Mistral 7B

## Autor

Joel Martín Jurado