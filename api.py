# api.py
import os
import math
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from config import PDF_DIR, TEXT_DIR, INDEX_DIR
from extractor.preprocess import preprocess_text
from expansion.semantic_expand import expand_query
from indexador.tfidf_index import load_tfidf_index, vectorize_query
from indexador.bm25f_index import load_bm25f_index, score_bm25f
from pathlib import Path
from urllib.parse import quote
from fastapi.middleware.cors import CORSMiddleware
import base64

# --- Modelo de respuesta ---


class SearchResult(BaseModel):
    title: str
    content: str
    url: str
    file_base64: str
    score: float


app = FastAPI(title="Buscador Semántico")
origins = [
    "http://localhost:3000",
    # puedes agregar más dominios:
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # o ["*"] para permitir todos
    allow_credentials=True,
    allow_methods=["*"],         # GET, POST, PUT…
    allow_headers=["*"],         # Content-Type, Authorization…
)


@app.on_event("startup")
def startup_event():
    global TFIDF_INDEX, IDF, DOC_IDS
    global INV_INDEX, BM25_STATS

    # Carga índices TF-IDF
    TFIDF_INDEX, IDF, DOC_IDS = load_tfidf_index()
    # Carga índice invertido BM25F
    INV_INDEX, BM25_STATS = load_bm25f_index()

# --- Función para extraer fragmento de texto ---


def get_snippet(query: str, txt_path: str) -> str:
    """
    Busca el primer término de la consulta en el texto y devuelve
    100 caracteres antes, el término y 100 después.
    Si no aparece, devuelve primeros 200 caracteres.
    """
    if not os.path.exists(txt_path):
        return ""
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    lower = text.lower()
    # intentamos cada término
    for term in query.lower().split():
        idx = lower.find(term)
        if idx != -1:
            start = max(0, idx - 100)
            end = min(len(text), idx + len(term) + 100)
            return text[start:end].strip()
    # sin coincidencias
    return text[:200].strip()

# --- Endpoint de búsqueda ---


@app.get("/search", response_model=List[SearchResult])
async def search_endpoint(
    q: str,
    top: int = 10,
    weight: float = 0.5
):
    """
    q: términos de búsqueda
    top: número de resultados a devolver (default 10)
    weight: peso TF-IDF vs BM25F (entre 0 y 1)
    """
    # 1) Preprocesar y expandir
    tokens = preprocess_text(q)
    tokens = expand_query(tokens)
    # 2) Vectorizar
    query_vec = vectorize_query(tokens)
    # 3) Coseno
    cos_scores = {}
    for doc_id, doc_vec in TFIDF_INDEX.items():
        # calcular coseno manualmente
        num = sum(doc_vec.get(t, 0)*query_vec.get(t, 0) for t in tokens)
        denom = math.sqrt(sum(v*v for v in doc_vec.values())) \
            * math.sqrt(sum(v*v for v in query_vec.values()))
        cos_scores[doc_id] = num/denom if denom > 0 else 0.0
    # 4) BM25F
    bm25_scores = score_bm25f(tokens, INV_INDEX, BM25_STATS)
    # 5) Combinar scores y ordenar
    final_scores = {
        doc: weight * cos_scores.get(doc, 0.0) +
        (1 - weight) * bm25_scores.get(doc, 0.0)
        for doc in DOC_IDS
    }
    # 5.1) FILTRAR solo con score > 0
    filtered_scores = {doc: sc for doc, sc in final_scores.items() if sc > 0.0}
    # 6) Ordenar y tomar top N
    ranked = sorted(
        filtered_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top]

    # 6) Construir respuesta
    results = []
    for doc_path, score in ranked:
        filename = os.path.basename(doc_path)
        name_without_ext, _ = os.path.splitext(filename)
        download_url = os.path.abspath(os.path.join(PDF_DIR, filename))+'.pdf'
        # Ruta absoluta al PDF
        absolute_pdf_path = os.path.abspath(os.path.join(PDF_DIR, filename))+'.pdf'
        # Lectura y codificación Base64
        with open(absolute_pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        file_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

        filename = os.path.basename(doc_path)
        title = os.path.splitext(filename)[0]
        txt_path = os.path.abspath(os.path.join(
            TEXT_DIR, name_without_ext + '.txt'))
        snippet = get_snippet(q, txt_path)
        results.append(SearchResult(
            title=title,
            content=snippet,
            url=download_url,
            file_base64=file_base64,
            score=round(score, 6)
        ))

    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
