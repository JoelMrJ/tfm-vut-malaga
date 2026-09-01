# RAG document corpus

Esta carpeta contiene el corpus documental utilizado por el sistema de Retrieval-Augmented Generation (RAG).

El corpus está formado por documentación oficial relacionada con las viviendas de uso turístico, la normativa urbanística y la regulación aplicable en Málaga y Andalucía.

Los documentos utilizados en el proyecto son:

1. Memoria de modificación del PGOU relativa a viviendas de uso turístico.
2. Informe jurídico y propuesta sobre la Instrucción 1/2024.
3. Plano de viviendas turísticas por barrios de Málaga.
4. Resumen ejecutivo de la modificación del PGOU.
5. Informe sobre el impacto de las viviendas de uso turístico.
6. Acuerdo del Pleno de 29 de mayo de 2025.
7. Decreto 31/2024 de la Junta de Andalucía.
8. Decreto 28/2016, texto consolidado.
9. Instrucción 1/2024 sobre normativa urbanística aplicada a viviendas turísticas.
10. Resolución de aplicación de la Instrucción 1/2024.

Los archivos PDF originales no se incluyen en este repositorio.

Los documentos pueden obtenerse a partir de las fuentes oficiales del Ayuntamiento de Málaga y de la Junta de Andalucía.

Una vez descargados, deben colocarse directamente en esta carpeta antes de ejecutar el pipeline de indexación del sistema RAG.

A partir de estos documentos se generan los fragmentos de texto, embeddings y la base de datos vectorial utilizada durante la recuperación documental.