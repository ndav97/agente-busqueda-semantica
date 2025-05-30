import os
import base64
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import RAW_PDF_DIR, EXTRACTED_TEXT_DIR
from buscador.search_engine import search


class SearchResult(BaseModel):
    title: str
    content: str
    url: str
    file_base64: str
    score: float


app = FastAPI(title="Buscador Semántico")

# CORS para permitir llamadas desde React en localhost:3000
origins = [
    "http://localhost:3000",
    # añade aquí otros orígenes si los necesitas
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # o ["*"] para permitir todos
    allow_credentials=True,
    allow_methods=["*"],         # GET, POST, PUT…
    allow_headers=["*"],         # Content-Type, Authorization…
)


def get_snippet(query: str, txt_path: str) -> str:
    """
    Extrae 100 caracteres antes y después del primer término encontrado
    en txt_path; si no hay coincidencias, devuelve los primeros 200 caracteres.
    """
    if not os.path.exists(txt_path):
        return ""
    text = open(txt_path, 'r', encoding='utf-8', errors='ignore').read()
    lower = text.lower()
    for term in query.lower().split():
        idx = lower.find(term)
        if idx != -1:
            start = max(0, idx - 100)
            end = min(len(text), idx + len(term) + 100)
            return text[start:end].strip()
    return text[:200].strip()


@app.get("/search", response_model=List[SearchResult])
async def search_endpoint(
    q: str,
    top: int = 10,
    weight: float = 0.5
):
    """
    q: términos de búsqueda
    top: número de resultados a devolver (default 10)
    weight: peso TF-IDF vs BM25F [0..1]
    """
    # Llamamos a la función híbrida (TF-IDF, BM25F y fastText)
    results = search(q, top_n=top, tfidf_weight=weight)

    response = []
    for doc_path, score in results:
        # Título (nombre de archivo sin extensión)
        filename = os.path.basename(doc_path)
        title, _ = os.path.splitext(filename)

        # Snippet
        txt_path = os.path.join(EXTRACTED_TEXT_DIR, title + '.txt')
        snippet = get_snippet(q, txt_path)

        # Ruta absoluta al PDF
        absolute_pdf_path = os.path.abspath(os.path.join(RAW_PDF_DIR, filename))+'.pdf'

        # Lectura y codificación Base64
        with open(absolute_pdf_path, 'rb') as f:
            file_bytes = f.read()
        file_base64 = base64.b64encode(file_bytes).decode('utf-8')

        response.append(
            SearchResult(
                title=title,
                content=snippet,
                url=absolute_pdf_path,
                file_base64=file_base64,
                score=round(score, 6)
            )
        )

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
