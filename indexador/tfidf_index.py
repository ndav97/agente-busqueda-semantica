import os
import math
import pickle
from collections import defaultdict

from config import EXTRACTED_TEXT_DIR, INDEX_DIR, TFIDF_USE_IDF, TFIDF_SMOOTH_IDF, TFIDF_NORMALIZE
from extractor.preprocess import preprocess_text


def build_tfidf_index():
    """
    Construye el índice TF-IDF manualmente:
    - Lee todos los .txt en EXTRACTED_TEXT_DIR
    - Preprocesa y tokeniza
    - Calcula TF, DF y, opcionalmente, IDF
    - Genera vectores TF-IDF y los normaliza
    - Guarda estructuras en INDEX_DIR
    """
    # Recopilar documentos y tokens
    docs_tokens = {}
    for root, _, files in os.walk(EXTRACTED_TEXT_DIR):
        for filename in files:
            if filename.lower().endswith('.txt'):
                path = os.path.join(root, filename)
                rel = os.path.relpath(path, EXTRACTED_TEXT_DIR)
                doc_id = os.path.splitext(rel)[0]
                text = open(path, 'r', encoding='utf-8').read()
                tokens = preprocess_text(text)
                docs_tokens[doc_id] = tokens

    N = len(docs_tokens)
    if N == 0:
        print("No hay documentos para indexar en", EXTRACTED_TEXT_DIR)
        return

    # Calcular DF (document frequency) y TF (term frequency)
    df = defaultdict(int)
    tf = {}  # doc_id -> dict term->freq
    for doc_id, tokens in docs_tokens.items():
        freqs = defaultdict(int)
        for term in tokens:
            freqs[term] += 1
        tf[doc_id] = freqs
        for term in freqs:
            df[term] += 1

    # Calcular IDF
    idf = {}
    for term, doc_freq in df.items():
        if TFIDF_SMOOTH_IDF:
            # idf suavizada: log(1 + N/df)
            idf_val = math.log(1.0 + (N / float(doc_freq)))
        else:
            idf_val = math.log(N / float(doc_freq)) if doc_freq > 0 else 0.0
        idf[term] = idf_val if TFIDF_USE_IDF else 1.0

    # Construir vectores TF-IDF y normalizar
    tfidf_index = {}
    for doc_id, freqs in tf.items():
        # calcular tfidf
        vec = {}
        for term, freq in freqs.items():
            vec[term] = freq * idf.get(term, 0.0)
        if TFIDF_NORMALIZE:
            # L2 norm
            norm = math.sqrt(sum(val * val for val in vec.values()))
            if norm > 0:
                for term in vec:
                    vec[term] /= norm
        tfidf_index[doc_id] = vec

    # Guardar en disco
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(os.path.join(INDEX_DIR, 'tfidf_index.pkl'), 'wb') as f:
        pickle.dump(tfidf_index, f)
    with open(os.path.join(INDEX_DIR, 'idf.pkl'), 'wb') as f:
        pickle.dump(idf, f)
    with open(os.path.join(INDEX_DIR, 'doc_ids.pkl'), 'wb') as f:
        pickle.dump(list(docs_tokens.keys()), f)

    print(f"Índice TF-IDF construido y guardado en '{INDEX_DIR}' con {N} documentos y {len(idf)} términos.")


def load_tfidf_index():
    """
    Carga estructuras TF-IDF desde disco.
    Retorna: (tfidf_index, idf, doc_ids)
    """
    with open(os.path.join(INDEX_DIR, 'tfidf_index.pkl'), 'rb') as f:
        tfidf_index = pickle.load(f)
    with open(os.path.join(INDEX_DIR, 'idf.pkl'), 'rb') as f:
        idf = pickle.load(f)
    with open(os.path.join(INDEX_DIR, 'doc_ids.pkl'), 'rb') as f:
        doc_ids = pickle.load(f)
    return tfidf_index, idf, doc_ids


def vectorize_query(tokens: list) -> dict:
    """
    Dado un listado de tokens de consulta, retorna un vector TF-IDF normalizado.
    """
    # TF de la consulta
    freqs = defaultdict(int)
    for t in tokens:
        freqs[t] += 1
    # Cargar IDF
    _, idf, _ = load_tfidf_index()
    # Construir vector
    vec = {}
    for term, freq in freqs.items():
        idf_val = idf.get(term, 0.0)
        weight = freq * idf_val if TFIDF_USE_IDF else freq
        vec[term] = weight
    # Normalizar
    if TFIDF_NORMALIZE:
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            for term in vec:
                vec[term] /= norm
    return vec


if __name__ == '__main__':
    build_tfidf_index()
