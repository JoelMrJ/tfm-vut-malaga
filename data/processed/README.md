# Processed data

Esta carpeta contiene los datos generados durante las etapas de limpieza, preprocesamiento e ingeniería de características.

Durante la ejecución del proyecto se generan, entre otros, los siguientes archivos:

```text
listings_cleaned.parquet
listings_features.parquet
listings_features_quality_report.csv
```

Los archivos .parquet no se incluyen en este repositorio, ya que pueden regenerarse a partir de los datos originales mediante los notebooks de preprocesamiento e ingeniería de características.

Sí se incluye:

listings_features_quality_report.csv

Este archivo contiene un resumen de controles de calidad realizados sobre el conjunto de datos procesado y permite mantener trazabilidad sobre el resultado del pipeline de preparación de datos.

La carpeta también puede contener archivos intermedios generados durante el procesamiento documental del sistema RAG. Estos artefactos no se incluyen en el repositorio, ya que pueden reconstruirse ejecutando nuevamente el pipeline correspondiente.