# Vector store

Esta carpeta contiene la base de datos vectorial persistente utilizada por el sistema RAG.

Durante la ejecución del proyecto, ChromaDB genera aquí el almacén vectorial con los embeddings de los fragmentos documentales.

La carpeta generada en el proyecto es:

```text
chroma_vut_malaga_v2/
```

El contenido del vector store no se incluye en este repositorio, ya que puede reconstruirse a partir del corpus documental ejecutando nuevamente el pipeline de indexación del sistema RAG.

Antes de utilizar el motor documental, deben haberse procesado e indexado los documentos situados en:

data/RAG/