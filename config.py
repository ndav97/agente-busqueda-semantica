import os

# config.py
import os


# Directorios base del proyecto\ nBASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PDF_DIR = os.path.join(BASE_DIR, 'data', 'libros_raw')  # PDFs originales
EXTRACTED_TEXT_DIR = os.path.join(
    BASE_DIR, 'data', 'libros_extraidos')  # Textos limpios
# Metadatos de libros (títulos, autores...)
METADATA_DIR = os.path.join(BASE_DIR, 'data', 'metadatos')

# Parámetros de TF-IDF
TFIDF_USE_IDF = True  # Calcular IDF
TFIDF_SMOOTH_IDF = True  # Suavizar IDF
TFIDF_NORMALIZE = True  # Normalizar vectores TF-IDF

# Parámetros de BM25F
BM25F_K1 = 1.5
BM25F_B = 0.75
# Pesos por campo para BM25F (p. ej. título, cuerpo)
BM25F_FIELD_WEIGHTS = {
    'titulo': 2.0,
    'cuerpo': 1.0
}

# Rendimiento y extensiones
NUM_WORKERS = 4  # Número de procesos para extracción y preprocesado
PDF_EXTENSIONS = ['.pdf']  # Extensiones válidas


PDF_DIR = os.path.join(BASE_DIR, 'data/libros_raw')
TEXT_DIR = os.path.join(BASE_DIR, 'data/libros_extraidos')
INDEX_DIR = os.path.join(BASE_DIR, 'data/indices')
STOPWORDS_PATH = os.path.join(BASE_DIR, 'data/stopwords.txt')
SYNONYMS_PATH = os.path.join(BASE_DIR, 'expansion/dictionary.json')
