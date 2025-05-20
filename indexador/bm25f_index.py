import os
import math
import pickle
from collections import defaultdict

from config import EXTRACTED_TEXT_DIR, INDEX_DIR, BM25F_K1, BM25F_B, BM25F_FIELD_WEIGHTS
from extractor.preprocess import preprocess_text


def build_bm25f_index():
    """
    Construye el índice invertido y estadísticas necesarias para BM25F:
    - Lee todos los .txt en EXTRACTED_TEXT_DIR
    - Preprocesa y tokeniza
    - Calcula df (document frequency)
    - Almacena freq por documento
    - Calcula longitudes y avgdl
    - Guarda estructuras en INDEX_DIR
    """
    # Estructuras intermedias
    inverted_index = defaultdict(dict)  # term -> {doc_id: freq}
    df = defaultdict(int)               # term -> doc frequency
    doc_lengths = {}                    # doc_id -> {field: length}

    # Procesar cada documento
    for root, _, files in os.walk(EXTRACTED_TEXT_DIR):
        for filename in files:
            if not filename.lower().endswith('.txt'):
                continue
            path = os.path.join(root, filename)
            rel = os.path.relpath(path, EXTRACTED_TEXT_DIR)
            doc_id = os.path.splitext(rel)[0]

            text = open(path, 'r', encoding='utf-8').read()
            tokens = preprocess_text(text)
            # Solo campo 'cuerpo' disponible
            length = len(tokens)
            doc_lengths[doc_id] = {'cuerpo': length}

            # Frecuencia de término en cuerpo
            freqs = defaultdict(int)
            for t in tokens:
                freqs[t] += 1
            # Actualizar df e inverted index
            for term, cnt in freqs.items():
                df[term] += 1
                inverted_index[term][doc_id] = cnt

    N = len(doc_lengths)
    if N == 0:
        print(f"No hay documentos para BM25F en {EXTRACTED_TEXT_DIR}")
        return

    # Calcular promedio de longitud por campo
    avgdl = {}
    for field in BM25F_FIELD_WEIGHTS:
        total = sum(lengths.get(field, 0) for lengths in doc_lengths.values())
        avgdl[field] = total / float(N)

    # Preparar estadísticas
    stats = {
        'N': N,
        'df': dict(df),
        'doc_lengths': doc_lengths,
        'avgdl': avgdl
    }

    # Guardar en disco
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(os.path.join(INDEX_DIR, 'bm25f_index.pkl'), 'wb') as f:
        pickle.dump(dict(inverted_index), f)
    with open(os.path.join(INDEX_DIR, 'bm25f_stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)

    print(
        f"Índice BM25F construido y guardado en '{INDEX_DIR}' (N={N}, términos={len(df)})")


def load_bm25f_index():
    """
    Carga el índice BM25F y estadísticas desde disco.
    Retorna: (inverted_index, stats)
    """
    with open(os.path.join(INDEX_DIR, 'bm25f_index.pkl'), 'rb') as f:
        inverted_index = pickle.load(f)
    with open(os.path.join(INDEX_DIR, 'bm25f_stats.pkl'), 'rb') as f:
        stats = pickle.load(f)
    return inverted_index, stats


def score_bm25f(query_terms, inverted_index, stats):
    """
    Calcula scores BM25F para todos los documentos dados los términos de consulta.
    Retorna: dict doc_id -> score
    """
    N = stats['N']
    df = stats['df']
    doc_lengths = stats['doc_lengths']
    avgdl = stats['avgdl']
    k1 = BM25F_K1
    b = BM25F_B
    field_weights = BM25F_FIELD_WEIGHTS

    scores = defaultdict(float)
    # Para cada término en la consulta (sin ponderar tf de consulta)
    for term in set(query_terms):
        if term not in inverted_index:
            continue
        # idf BM25F suavizado
        df_t = df.get(term, 0)
        idf = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)
        postings = inverted_index[term]
        # Para cada doc que contiene el término
        for doc_id, f_td in postings.items():
            # Solo campo cuerpo
            dl = doc_lengths[doc_id].get('cuerpo', 0)
            avg = avgdl.get('cuerpo', 0)
            # Factor de normalización
            denom = f_td + k1 * (1 - b + b * (dl / avg))
            score = idf * ((f_td * (k1 + 1)) / denom)
            # Aplicar peso de campo (cuerpo)
            score *= field_weights.get('cuerpo', 1.0)
            scores[doc_id] += score
    return dict(scores)


if __name__ == '__main__':
    build_bm25f_index()
