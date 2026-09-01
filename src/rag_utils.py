from pathlib import Path
import re
import time
import unicodedata
import warnings

import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_RAG_DIR = (
    PROJECT_ROOT / "data" / "processed" / "rag_v2"
)

VECTOR_DB_DIR = (
    PROJECT_ROOT / "data" / "vectorstore" / "chroma_vut_malaga_v2"
)

CHUNKS_PATH = (
    PROCESSED_RAG_DIR / "rag_chunks_v2.parquet"
)

EMBEDDING_MODEL_NAME = (
    "paraphrase-multilingual-MiniLM-L12-v2"
)

COLLECTION_NAME = (
    "vut_malaga_normativa_v2"
)

FINAL_LLM = "mistral:7b"

CANDIDATE_POOL = 15
FINAL_TOP_K = 4
MIN_SIMILARITY = 0.28
NEAR_DUPLICATE_THRESHOLD = 0.93
MMR_LAMBDA = 0.72
MAX_PER_DOCUMENT = 2

if not CHUNKS_PATH.exists():
    raise FileNotFoundError(
        f"No existe el fichero de chunks: {CHUNKS_PATH}"
    )

if not VECTOR_DB_DIR.exists():
    raise FileNotFoundError(
        f"No existe el vectorstore: {VECTOR_DB_DIR}"
    )

chunks_df = pd.read_parquet(CHUNKS_PATH)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

chroma_client = chromadb.PersistentClient(
    path=str(VECTOR_DB_DIR)
)

collection = chroma_client.get_collection(
    COLLECTION_NAME
)



DOCUMENT_ALIASES = {
    "Decreto 31/2024": [
        "decreto 31/2024",
        "decreto 31 2024",
        "decreto 31",
    ],
    "Decreto 28/2016": [
        "decreto 28/2016",
        "decreto 28 2016",
        "decreto 28",
    ],
    "Instrucción 1/2024": [
        "instrucción 1/2024",
        "instruccion 1/2024",
        "instrucción 1 2024",
        "instruccion 1 2024",
    ],
    "Resolución sobre Instrucción 1/2024": [
        "resolución",
        "resolucion",
    ],
    "Informe jurídico sobre Instrucción 1/2024": [
        "informe jurídico",
        "informe juridico",
    ],
    "Memoria modificación PGOU VUT": [
        "memoria",
        "pgou",
    ],
    "Resumen Ejecutivo PGOU VUT": [
        "resumen ejecutivo",
    ],
    "Acuerdo Pleno aprobación definitiva": [
        "acuerdo pleno",
        "acuerdo del pleno",
        "aprobación definitiva",
        "aprobacion definitiva",
    ],
    "Informe de Impacto de la Vivienda Turística": [
        "informe de impacto",
        "impacto de la vivienda turística",
        "impacto de la vivienda turistica",
    ],
}


def normalize_text(text):
    text = unicodedata.normalize(
        "NFKD",
        str(text),
    )

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(
            char
        )
    )

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def detect_explicit_documents(query):
    query_normalized = normalize_text(
        query
    )

    detected = []

    for title, aliases in (
        DOCUMENT_ALIASES.items()
    ):
        normalized_aliases = [
            normalize_text(alias)
            for alias in aliases
        ]

        if any(
            alias in query_normalized
            for alias in normalized_aliases
        ):
            detected.append(title)

    return detected


def cosine_similarity_matrix(matrix):
    matrix = np.asarray(
        matrix,
        dtype=np.float32,
    )

    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    norms = np.clip(
        norms,
        1e-12,
        None,
    )

    normalized = matrix / norms

    return normalized @ normalized.T


CHANGE_TERMS = [
    "modifica",
    "modificación",
    "modificacion",
    "queda modificado",
    "queda modificada",
    "se incorpora",
    "se incorporan",
    "se añade",
    "se anade",
    "nuevos requisitos",
    "nuevo requisito",
    "se sustituye",
    "se refuerza",
]

RESIDENTIAL_IMPACT_TERMS = [
    "impacto",
    "vivienda residencial",
    "acceso a la vivienda",
    "alquiler",
    "precios de alquiler",
    "precio de la vivienda",
    "hogares",
    "población",
    "poblacion",
    "viviendas principales",
    "viviendas vacías",
    "viviendas vacias",
    "presión turística residencial",
    "presion turistica residencial",
    "tensionando",
    "pierden",
    "desplazamiento",
]


def is_change_query(query):
    query_norm = normalize_text(query)

    change_query_terms = [
        "que cambios",
        "que cambia",
        "que modifica",
        "que modificaciones",
        "que introduce",
        "que incorpora",
    ]

    return any(
        term in query_norm
        for term in change_query_terms
    )


def is_residential_impact_query(query):
    query_norm = normalize_text(query)

    return (
        (
            "impacto" in query_norm
            or "efecto" in query_norm
            or "consecuencia" in query_norm
        )
        and (
            "vivienda" in query_norm
            or "residencial" in query_norm
            or "alquiler" in query_norm
        )
    )


def lexical_change_bonus(text):
    text_norm = normalize_text(text)

    hits = sum(
        1
        for term in CHANGE_TERMS
        if normalize_text(term) in text_norm
    )

    return min(
        0.12,
        0.03 * hits,
    )


def lexical_residential_bonus(text):
    text_norm = normalize_text(text)

    hits = sum(
        1
        for term in RESIDENTIAL_IMPACT_TERMS
        if normalize_text(term) in text_norm
    )

    # El bonus es deliberadamente moderado:
    # ayuda a distinguir impacto residencial de regulación,
    # sin sustituir la similitud semántica.
    return min(
        0.18,
        0.025 * hits,
    )


def expand_query_for_retrieval(query):
    """
    Expansión muy controlada solo para intenciones donde la consulta
    original es demasiado general para recuperar evidencia concreta.
    """
    if is_residential_impact_query(query):
        return (
            f"{query} "
            "impacto sobre vivienda residencial acceso a la vivienda "
            "precios de alquiler hogares población viviendas principales "
            "viviendas vacías presión turística residencial"
        )

    return query


def retrieve_candidates(
    query,
    candidate_pool=CANDIDATE_POOL,
    document_bonus=0.05,
):
    retrieval_query = expand_query_for_retrieval(
        query
    )

    # En preguntas de impacto residencial ampliamos el pool
    # porque los mejores fragmentos pueden no estar entre los
    # primeros resultados de una consulta muy genérica.
    if is_residential_impact_query(query):
        candidate_pool = max(
            candidate_pool,
            30,
        )

    query_embedding = embedding_model.encode(
        [retrieval_query],
        normalize_embeddings=True,
    )

    n_candidates = min(
        candidate_pool,
        collection.count(),
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_candidates,
        include=[
            "documents",
            "metadatas",
            "distances",
            "embeddings",
        ],
    )

    explicit_documents = detect_explicit_documents(
        query
    )

    rows = []

    for i in range(
        len(results["documents"][0])
    ):
        metadata = results["metadatas"][0][i]
        text = results["documents"][0][i]

        distance = float(
            results["distances"][0][i]
        )

        semantic_similarity = 1.0 - distance

        explicit_match = (
            metadata["short_title"]
            in explicit_documents
        )

        residential_bonus = (
            lexical_residential_bonus(text)
            if is_residential_impact_query(query)
            else 0.0
        )

        effective_relevance = (
            semantic_similarity
            + (
                document_bonus
                if explicit_match
                else 0.0
            )
            + residential_bonus
        )

        rows.append(
            {
                "candidate_rank": i + 1,
                "documento": metadata[
                    "short_title"
                ],
                "tipo": metadata[
                    "document_type"
                ],
                "pagina": int(
                    metadata["page"]
                ),
                "chunk_index": int(
                    metadata.get(
                        "chunk_index",
                        0,
                    )
                ),
                "texto": text,
                "distance": distance,
                "semantic_similarity": (
                    semantic_similarity
                ),
                "lexical_bonus": residential_bonus,
                "explicit_document_match": (
                    explicit_match
                ),
                "effective_relevance": (
                    effective_relevance
                ),
                "embedding": np.asarray(
                    results["embeddings"][0][i],
                    dtype=np.float32,
                ),
            }
        )

    return pd.DataFrame(rows)


def retrieve_explicit_document(
    query,
    document_title,
    n_results=12,
):
    """
    Recuperación directa dentro del documento citado.
    Para preguntas de cambios/modificaciones añade un bonus léxico.
    """
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
    )[0]

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=n_results,
        where={
            "short_title": document_title
        },
        include=[
            "documents",
            "metadatas",
            "distances",
            "embeddings",
        ],
    )

    rows = []
    change_query = is_change_query(query)

    for i in range(
        len(results["documents"][0])
    ):
        metadata = results["metadatas"][0][i]
        text = results["documents"][0][i]

        distance = float(
            results["distances"][0][i]
        )

        semantic_similarity = 1.0 - distance

        lexical_bonus = (
            lexical_change_bonus(text)
            if change_query
            else 0.0
        )

        rows.append(
            {
                "candidate_rank": i + 1,
                "documento": metadata[
                    "short_title"
                ],
                "tipo": metadata[
                    "document_type"
                ],
                "pagina": int(
                    metadata["page"]
                ),
                "chunk_index": int(
                    metadata.get(
                        "chunk_index",
                        0,
                    )
                ),
                "texto": text,
                "distance": distance,
                "semantic_similarity": (
                    semantic_similarity
                ),
                "lexical_bonus": lexical_bonus,
                "explicit_document_match": True,
                "effective_relevance": (
                    semantic_similarity
                    + 0.15
                    + lexical_bonus
                ),
                "embedding": np.asarray(
                    results["embeddings"][0][i],
                    dtype=np.float32,
                ),
            }
        )

    return pd.DataFrame(rows)


def get_neighbor_chunk(
    document_title,
    page,
    chunk_index,
    direction=1,
):
    document_chunks = chunks_df.loc[
        chunks_df["short_title"]
        == document_title
    ].copy()

    document_chunks = (
        document_chunks
        .sort_values(
            [
                "page",
                "chunk_index",
            ]
        )
        .reset_index(drop=True)
    )

    current_matches = document_chunks.index[
        (
            document_chunks["page"]
            == page
        )
        & (
            document_chunks["chunk_index"]
            == chunk_index
        )
    ].tolist()

    if not current_matches:
        return None

    neighbor_position = (
        current_matches[0]
        + direction
    )

    if (
        neighbor_position < 0
        or neighbor_position
        >= len(document_chunks)
    ):
        return None

    return document_chunks.iloc[
        neighbor_position
    ]


def remove_near_duplicates(
    candidates,
    threshold=NEAR_DUPLICATE_THRESHOLD,
):
    if candidates.empty:
        return candidates.copy()

    candidates = (
        candidates
        .sort_values(
            [
                "effective_relevance",
                "semantic_similarity",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    kept_rows = []
    kept_embeddings = []

    for _, row in candidates.iterrows():
        current_embedding = row[
            "embedding"
        ]

        if not kept_embeddings:
            kept_rows.append(
                row.to_dict()
            )
            kept_embeddings.append(
                current_embedding
            )
            continue

        similarities = [
            float(
                np.dot(
                    current_embedding,
                    previous_embedding,
                )
            )
            for previous_embedding
            in kept_embeddings
        ]

        if max(similarities) < threshold:
            kept_rows.append(
                row.to_dict()
            )
            kept_embeddings.append(
                current_embedding
            )

    return pd.DataFrame(
        kept_rows
    )


def mmr_select(
    candidates,
    top_k=FINAL_TOP_K,
    lambda_mult=MMR_LAMBDA,
    max_per_document=MAX_PER_DOCUMENT,
):
    if candidates.empty:
        return candidates.copy()

    candidates = (
        candidates
        .sort_values(
            "effective_relevance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    selected_indices = []
    document_counts = {}

    while (
        len(selected_indices) < top_k
        and len(selected_indices)
        < len(candidates)
    ):
        best_index = None
        best_score = -np.inf

        for idx, row in candidates.iterrows():
            if idx in selected_indices:
                continue

            document = row["documento"]

            if (
                document_counts.get(
                    document,
                    0,
                )
                >= max_per_document
            ):
                continue

            relevance = float(
                row["effective_relevance"]
            )

            if not selected_indices:
                diversity_penalty = 0.0
            else:
                current_embedding = row[
                    "embedding"
                ]

                diversity_penalty = max(
                    float(
                        np.dot(
                            current_embedding,
                            candidates.loc[
                                selected_idx,
                                "embedding",
                            ],
                        )
                    )
                    for selected_idx
                    in selected_indices
                )

            mmr_score = (
                lambda_mult
                * relevance
                - (
                    1.0 - lambda_mult
                )
                * diversity_penalty
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_index = idx

        if best_index is None:
            break

        selected_indices.append(
            best_index
        )

        selected_document = candidates.loc[
            best_index,
            "documento",
        ]

        document_counts[
            selected_document
        ] = (
            document_counts.get(
                selected_document,
                0,
            )
            + 1
        )

        candidates.loc[
            best_index,
            "mmr_score",
        ] = best_score

    selected = (
        candidates
        .loc[selected_indices]
        .copy()
        .reset_index(drop=True)
    )

    selected.insert(
        0,
        "rank",
        np.arange(
            1,
            len(selected) + 1,
        ),
    )

    return selected


def merge_open_change_continuations(
    selected,
    query,
    document_title,
):
    """
    Si un fragmento seleccionado anuncia que una norma 'queda modificada
    en los siguientes términos', concatena el chunk inmediatamente posterior
    dentro del mismo resultado.

    Así no se desperdicia uno de los 4 puestos con un mero encabezado y,
    al mismo tiempo, se conserva la continuidad jurídica entre páginas/chunks.
    """
    if (
        selected.empty
        or not is_change_query(query)
    ):
        return selected.copy()

    open_ending_terms = [
        "queda modificado",
        "queda modificada",
        "en los siguientes terminos",
        "se modifica",
    ]

    merged = selected.copy()

    for idx, row in merged.iterrows():
        text_norm = normalize_text(
            row["texto"]
        )

        if not any(
            normalize_text(term)
            in text_norm
            for term in open_ending_terms
        ):
            continue

        neighbor = get_neighbor_chunk(
            document_title=document_title,
            page=int(row["pagina"]),
            chunk_index=int(
                row["chunk_index"]
            ),
            direction=1,
        )

        if neighbor is None:
            continue

        neighbor_text = str(
            neighbor["text"]
        ).strip()

        if not neighbor_text:
            continue

        neighbor_page = int(
            neighbor["page"]
        )

        merged.at[
            idx,
            "texto",
        ] = (
            str(row["texto"]).rstrip()
            + "\n\n"
            + (
                f"[Continuación inmediata, "
                f"página {neighbor_page}]\n"
            )
            + neighbor_text
        )

        merged.at[
            idx,
            "continuation_page",
        ] = neighbor_page

    return merged


def retrieve_documents(
    query,
    n_results=FINAL_TOP_K,
    min_similarity=MIN_SIMILARITY,
):
    """
    Retrieval híbrido v2.2.

    DOCUMENTO EXPLÍCITO:
    - consulta filtrada al documento citado;
    - bonus léxico para preguntas de cambios;
    - continuidad jurídica: un encabezado de modificación se fusiona
      con su chunk inmediatamente posterior.

    PREGUNTA GENERAL:
    - búsqueda semántica;
    - expansión controlada para impacto residencial;
    - candidate pool ampliado en esa intención;
    - bonus léxico temático;
    - deduplicación + MMR + diversidad documental.
    """
    explicit_documents = detect_explicit_documents(
        query
    )

    # =====================================================
    # A. Documento citado explícitamente
    # =====================================================
    if explicit_documents:
        primary_document = explicit_documents[0]

        candidates = retrieve_explicit_document(
            query=query,
            document_title=primary_document,
            n_results=12,
        )

        if not candidates.empty:
            candidates = candidates.loc[
                candidates[
                    "semantic_similarity"
                ]
                >= min_similarity
            ].copy()

        if candidates.empty:
            # Fallback general si el filtro documental no recupera
            # evidencia suficiente.
            candidates = retrieve_candidates(
                query
            )

            if candidates.empty:
                return candidates

            candidates = candidates.loc[
                candidates[
                    "semantic_similarity"
                ]
                >= min_similarity
            ].copy()

            if candidates.empty:
                return candidates

            selected = mmr_select(
                remove_near_duplicates(
                    candidates
                ),
                top_k=n_results,
            )

        else:
            deduplicated = remove_near_duplicates(
                candidates
            )

            selected = mmr_select(
                deduplicated,
                top_k=n_results,
                max_per_document=n_results,
            )

            selected = (
                merge_open_change_continuations(
                    selected=selected,
                    query=query,
                    document_title=primary_document,
                )
            )

    # =====================================================
    # B. Pregunta general
    # =====================================================
    else:
        candidates = retrieve_candidates(
            query
        )

        if candidates.empty:
            return candidates

        candidates = candidates.loc[
            candidates[
                "semantic_similarity"
            ]
            >= min_similarity
        ].copy()

        if candidates.empty:
            return candidates

        deduplicated = remove_near_duplicates(
            candidates
        )

        selected = mmr_select(
            deduplicated,
            top_k=n_results,
            max_per_document=(
                MAX_PER_DOCUMENT
            ),
        )

    if selected.empty:
        return selected

    selected = (
        selected
        .drop_duplicates(
            subset=[
                "documento",
                "pagina",
                "chunk_index",
            ],
            keep="first",
        )
        .reset_index(drop=True)
        .drop(
            columns=[
                "embedding",
            ],
            errors="ignore",
        )
    )

    selected["rank"] = np.arange(
        1,
        len(selected) + 1,
    )

    columns = [
        "rank"
    ] + [
        column
        for column in selected.columns
        if column != "rank"
    ]

    return selected[columns]

def format_rag_context(
    retrieved_df
):
    context_parts = []
    source_map = {}

    for i, row in (
        retrieved_df.iterrows()
    ):
        source_id = f"S{i + 1}"

        source_map[
            source_id
        ] = {
            "documento": row[
                "documento"
            ],
            "pagina": int(
                row["pagina"]
            ),
        }

        similarity = row.get(
            "semantic_similarity",
            np.nan,
        )

        similarity_text = (
            f"{float(similarity):.3f}"
            if pd.notna(similarity)
            else "contexto_vecino"
        )

        context_parts.append(
            (
                f"[{source_id}]\n"
                f"FUENTE: {row['documento']}\n"
                f"TIPO: {row.get('tipo', '')}\n"
                f"PÁGINA: {int(row['pagina'])}\n"
                f"SIMILITUD: {similarity_text}\n\n"
                f"{row['texto']}"
            )
        )

    return (
        "\n\n---\n\n".join(
            context_parts
        ),
        source_map,
    )


def replace_source_ids(
    answer,
    source_map,
):
    def replacement(match):
        source_id = match.group(1)

        if source_id not in source_map:
            return match.group(0)

        source = source_map[
            source_id
        ]

        return (
            f"[{source['documento']}, "
            f"p. {source['pagina']}]"
        )

    return re.sub(
        r"\[(S\d+)\]",
        replacement,
        answer,
    )

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Eres un asistente documental especializado en viviendas de uso turístico
(VUT) en Málaga y Andalucía.

Debes responder ÚNICAMENTE con la información incluida en el CONTEXTO.

Responde SIEMPRE en español, aunque el contenido recuperado o el modelo estén en otro idioma.

PROCEDIMIENTO OBLIGATORIO:

Antes de redactar la respuesta:
- revisa TODOS los fragmentos recuperados;
- identifica qué aporta cada fragmento a la pregunta;
- descarta únicamente los fragmentos que no sean relevantes;
- no te limites al primer fragmento si otros aportan información
  complementaria necesaria.

REGLAS:

1. Responde exactamente a la pregunta formulada y hazlo SIEMPRE en español.

2. Si la pregunta menciona expresamente un documento
   (por ejemplo, "¿Qué establece la Instrucción 1/2024?"),
   sintetiza las principales ideas relevantes recuperadas DE ESE
   DOCUMENTO. No atribuyas al documento principal afirmaciones que
   procedan de otra fuente.

3. Cuando varios fragmentos del documento solicitado aporten
   aspectos diferentes y relevantes, intégralos en la respuesta.

4. No añadas información que no aparezca en el contexto.

5. No utilices conocimiento general ni conocimiento previo del modelo.

6. Distingue, cuando sea necesario, entre:
   - normativa autonómica;
   - normativa o instrumentos municipales;
   - planeamiento urbanístico;
   - informes técnicos o jurídicos.

7. No confundas una norma citada dentro de un documento con
   el propio contenido o finalidad del documento consultado.

8. Si varias fuentes repiten la misma idea, exprésala una sola vez.

9. Si la evidencia recuperada no permite responder con seguridad,
   responde exactamente:
   "La documentación recuperada no aporta evidencia suficiente para
   responder con precisión a esta pregunta."

10. Cada afirmación importante debe indicar la fuente que la respalda
    mediante [S1], [S2], [S3] o [S4].

11. Solo puedes utilizar identificadores presentes en el CONTEXTO.

12. No inventes fuentes, artículos, cifras ni consecuencias.

13. Para preguntas amplias, organiza la respuesta en varios puntos
    breves cuando existan varias ideas diferentes.

14. Evita repetir una conclusión en un párrafo final si ya ha quedado
    explicada en la respuesta.

15. La respuesta debe ser concisa, pero suficientemente completa
    para responder a toda la pregunta.

16. La respuesta es informativa y no sustituye asesoramiento jurídico.
            """,
        ),
        (
            "human",
            """
PREGUNTA:
{question}

CONTEXTO:
{context}

Analiza todos los fragmentos relevantes y responde únicamente
a la pregunta formulada.
            """,
        ),
    ]
)

print("Prompt v3 preparado")

llm_cache = {}


def get_llm(model_name):
    if model_name not in llm_cache:
        llm_cache[
            model_name
        ] = ChatOllama(
            model=model_name,
            temperature=0,
            num_ctx=4096,
        )

    return llm_cache[
        model_name
    ]

def ask_rag(
    question,
    model_name=FINAL_LLM,
    n_results=FINAL_TOP_K,
):
    retrieved = retrieve_documents(
        question,
        n_results=n_results,
    )

    if retrieved.empty:
        return {
            "question": question,
            "model": model_name,
            "answer": (
                "La documentación recuperada no aporta "
                "evidencia suficiente para responder "
                "con precisión a esta pregunta."
            ),
            "raw_answer": None,
            "sources": retrieved,
            "generation_seconds": 0.0,
            "abstained_before_llm": True,
        }

    context, source_map = (
        format_rag_context(
            retrieved
        )
    )

    llm = get_llm(
        model_name
    )

    chain = (
        rag_prompt
        | llm
    )

    start = time.perf_counter()

    response = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    raw_answer = str(
        response.content
    ).strip()

    final_answer = (
        replace_source_ids(
            raw_answer,
            source_map,
        )
    )

    return {
        "question": question,
        "model": model_name,
        "answer": final_answer,
        "raw_answer": raw_answer,
        "sources": retrieved,
        "generation_seconds": elapsed,
        "abstained_before_llm": False,
    }
