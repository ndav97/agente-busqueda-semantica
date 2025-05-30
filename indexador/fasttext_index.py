# indexador/fasttext_index.py

import os
import pickle

import numpy as np
import fasttext

from extractor.preprocess import preprocess_text
from config import PDF_DIR, TEXT_DIR, EMBEDDINGS_PATH, FASTTEXT_MODEL_PATH


def build_fasttext_index():
    """
    Genera embeddings de documento con fastText:
      1) Carga el modelo preentrenado.
      2) Por cada PDF en PDF_DIR, lee el .txt preprocesado en TEXT_DIR,
         calcula el embedding promedio y lo guarda en EMBEDDINGS_PATH.
    """
    # 1) Cargar el modelo fastText
    model = fasttext.load_model(FASTTEXT_MODEL_PATH)

    embeddings = {}

    # 2) Recorre todos los archivos PDF
    for fname in os.listdir(PDF_DIR):
        print(f"Procesando {fname}...")
        if not fname.lower().endswith('.pdf'):
            continue
        base, _ = os.path.splitext(fname)
        txt_path = os.path.join(TEXT_DIR, base + '.txt')
        print(f"  - Buscando texto preprocesado en: {txt_path}")
        if not os.path.exists(txt_path):
            # Si no hay .txt para este PDF, lo saltamos
            continue

        # 3) Leer y preprocesar el texto
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        tokens = preprocess_text(text)

        # 4) Obtener vectores fastText para cada token
        vecs = [model.get_word_vector(t) for t in tokens if t]
        if not vecs:
            # Si no hay tokens válidos, saltamos
            continue

        # 5) Calcular embedding de documento (media de vectores)
        doc_emb = np.mean(vecs, axis=0)

        # 6) Guardar embedding en el diccionario, clave = ruta absoluta al PDF
        pdf_path = os.path.join(PDF_DIR, fname)
        embeddings[pdf_path] = doc_emb

    # 7) Asegurar que el directorio de salida existe
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)

    # 8) Persistir todos los embeddings a disco
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(embeddings, f)

    print(
        f"✅ fastText: embeddings generados para {len(embeddings)} documentos.")
