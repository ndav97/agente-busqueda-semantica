import math

from extractor.preprocess import preprocess_text
from indexador.tfidf_index import load_tfidf_index, vectorize_query
from indexador.bm25f_index import load_bm25f_index, score_bm25f
from expansion.semantic_expand import expand_query


def search(query: str, top_n: int = 10, tfidf_weight: float = 0.5) -> list:
    """
    Ejecuta el pipeline de búsqueda:
    1. Preprocesa la consulta
    2. Expande la consulta semánticamente
    3. Vectoriza con TF-IDF
    4. Calcula similitud coseno TF-IDF y score BM25F
    5. Combina scores y retorna top_n resultados
    """
    # 1. Preprocesado
    tokens = preprocess_text(query)
    print(f"Tokens preprocesados: {tokens}")
    # 2. Expansión semántica
    expanded = expand_query(tokens)
    print(f"Tokens expandidos: {expanded}")

    # 3. Vectorizar consulta
    q_vec = vectorize_query(expanded)
    print(f"Vector de consulta: {q_vec}")

    # 4. Cargar índices
    tfidf_index, _, doc_ids = load_tfidf_index()
    inverted_index, bm25f_stats = load_bm25f_index()

    # 5. Calcular similitud coseno TF-IDF
    cos_scores = {}
    for doc_id, doc_vec in tfidf_index.items():
        # producto punto
        score = sum(q_vec.get(term, 0.0) * doc_vec.get(term, 0.0)
                    for term in q_vec)
        cos_scores[doc_id] = score

    # 6. Calcular score BM25F
    bm25_scores = score_bm25f(expanded, inverted_index, bm25f_stats)

    # 7. Combinar scores (ponderación configurable)
    combined = {}
    for doc_id in doc_ids:
        c1 = cos_scores.get(doc_id, 0.0)
        c2 = bm25_scores.get(doc_id, 0.0)
        combined[doc_id] = tfidf_weight * c1 + (1.0 - tfidf_weight) * c2

    # 8. Seleccionar top_n
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Uso: python search_engine.py 'consulta de prueba'")
        sys.exit(1)
    query_str = sys.argv[1]
    results = search(query_str)
    for rank, (doc_id, score) in enumerate(results, start=1):
        print(f"{rank}. {doc_id:<40} Score: {score:.4f}")
