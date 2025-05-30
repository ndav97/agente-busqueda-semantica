# buscador/search_engine.py

import os
import sys
import math
import pickle

import numpy as np
import fasttext

from extractor.preprocess import preprocess_text
from indexador.tfidf_index import load_tfidf_index, vectorize_query
from indexador.bm25f_index import load_bm25f_index, score_bm25f
from expansion.semantic_expand import expand_query
from indexador.fasttext_index import build_fasttext_index
from config import FASTTEXT_MODEL_PATH, EMBEDDINGS_PATH, SEMANTIC_WEIGHT

# 1) Carga índices TF-IDF
tfidf_index, idf, doc_ids = load_tfidf_index()

# 2) Carga índice invertido BM25F
inverted_index, bm25f_stats = load_bm25f_index()

# 3) Asegurarse de que existen los embeddings; si no, generarlos
if not os.path.exists(EMBEDDINGS_PATH):
    print("⚙️  doc_embeddings.pkl no encontrado, generando embeddings con fastText…")
    build_fasttext_index()

# 4) Carga modelo fastText y embeddings de documentos
FT_MODEL = fasttext.load_model(FASTTEXT_MODEL_PATH)
with open(EMBEDDINGS_PATH, 'rb') as f:
    doc_embeddings = pickle.load(f)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula la similitud de coseno entre dos vectores.
    """
    num = np.dot(a, b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(num / den) if den else 0.0


def search(query: str, top_n: int = 10, tfidf_weight: float = 0.5) -> list:
    """
    Ejecuta el pipeline de búsqueda:
    1. Preprocesa la consulta
    2. Expande la consulta semánticamente
    3. Vectoriza con TF-IDF
    4. Calcula similitud coseno TF-IDF y score BM25F
    5. Calcula score semántico con fastText
    6. Combina scores léxico y semántico y retorna top_n resultados
    """
    # 1) Preprocesado
    tokens = preprocess_text(query)
    print(f"Tokens preprocesados: {tokens}")

    # 2) Expansión semántica
    expanded = expand_query(tokens)
    print(f"Tokens expandidos: {expanded}")

    # 3) Vectorizar consulta (TF-IDF)
    q_vec = vectorize_query(expanded)
    print(f"Vector de consulta (TF-IDF): {q_vec}")

    # 4) Similitud coseno (TF-IDF)
    cos_scores = {
        doc: sum(q_vec.get(term, 0.0) * vec.get(term, 0.0) for term in q_vec)
        for doc, vec in tfidf_index.items()
    }

    # 5) Score BM25F
    bm25_scores = score_bm25f(expanded, inverted_index, bm25f_stats)

    # 6) Componente léxico: TF-IDF + BM25F
    lex_scores = {
        doc: tfidf_weight * cos_scores.get(doc, 0.0)
        + (1 - tfidf_weight) * bm25_scores.get(doc, 0.0)
        for doc in doc_ids
    }

    # 7) Score semántico con fastText
    emb_list = [FT_MODEL.get_word_vector(t) for t in expanded if t]
    if emb_list:
        q_emb = np.mean(emb_list, axis=0)
    else:
        q_emb = np.zeros(FT_MODEL.get_dimension())
    sem_scores = {
        doc: cosine_sim(q_emb, emb)
        for doc, emb in doc_embeddings.items()
    }

    print(f"Scores semánticos: {sem_scores}")

    # 8) Score final: mezcla de léxico y semántico
    final_scores = {
        doc: (1 - SEMANTIC_WEIGHT) * lex_scores.get(doc, 0.0)
        + SEMANTIC_WEIGHT * sem_scores.get(doc, 0.0)
        for doc in doc_ids
    }

    print(f"Scores finales combinados: {final_scores}")

    # 9) Filtrar y ordenar resultados con score > 0
    ranked = sorted(
        ((doc, sc) for doc, sc in final_scores.items() if sc > 0.0),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"Resultados ordenados: {ranked}")

    return ranked[:top_n]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python search_engine.py 'consulta de prueba'")
        sys.exit(1)

    query_str = sys.argv[1]
    results = search(query_str)

    if not results:
        print("No se encontraron documentos relevantes con score > 0.")
    else:
        for rank, (doc_id, score) in enumerate(results, start=1):
            print(f"{rank}. {doc_id:<40} Score: {score:.4f}")
