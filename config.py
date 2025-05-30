import os

# Directorios base del proyecto\ nBASE_DIR = os.path.dirname(os.path.abspath(__file__))
# config.py

import os

# ─── RUTAS BASE ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Directorio donde guardas los PDFs "raw"
PDF_DIR = os.path.join(DATA_DIR, 'libros_raw')

# Directorio donde guardas los TXT extraídos de los PDFs
TEXT_DIR = os.path.join(DATA_DIR, 'libros_extraidos')

# Directorio donde persistes índices y archivos auxiliares
INDEX_DIR = os.path.join(DATA_DIR, 'indices')


# ─── RECURSOS LINGÜÍSTICOS ─────────────────────────────────────────────────────
# Stopwords en español (una por línea)
STOPWORDS_PATH = os.path.join(DATA_DIR, 'stopwords.txt')

# Diccionario de sinónimos para expansión semántica
SYNONYMS_PATH = os.path.join(BASE_DIR, 'expansion', 'dictionary.json')


# ─── PARÁMETROS TF-IDF ─────────────────────────────────────────────────────────
# Tipo de normalización para vectores TF-IDF ('l1', 'l2', o None)
TFIDF_NORM = 'l2'


# ─── PARÁMETROS BM25F ──────────────────────────────────────────────────────────
# k1 controla la saturación de la frecuencia de término
BM25_K1 = 1.5
# b controla cuánto penaliza la longitud del documento
BM25_B = 0.75


# ─── PARÁMETROS DE PARALELIZACIÓN ──────────────────────────────────────────────
# Número de procesos para extracción y preprocesado de PDFs
NUM_WORKERS = 4


# ─── fastText (EMBEDDINGS SEMÁNTICOS) ──────────────────────────────────────────
# Directorio donde cargas tu modelo preentrenado de fastText
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FASTTEXT_MODEL_PATH = os.path.join(MODELS_DIR, 'cc.es.300.bin')

# Archivo donde guardas los embeddings de documento generados
EMBEDDINGS_PATH = os.path.join(INDEX_DIR, 'doc_embeddings.pkl')

# Peso del componente semántico al combinar scores (0.0–1.0)
SEMANTIC_WEIGHT = 0.3


# ─── AJUSTES DE BÚSQUEDA POR DEFECTO ───────────────────────────────────────────
# Número de resultados por defecto
TOP_N_DEFAULT = 10
# Peso TF-IDF vs BM25F en el componente léxico (0.0–1.0)
LEX_WEIGHT_DEFAULT = 0.5
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PDF_DIR = os.path.join(BASE_DIR, 'data', 'libros_raw')  # PDFs originales
EXTRACTED_TEXT_DIR = os.path.join(
    BASE_DIR, 'data', 'libros_extraidos')  # Textos limpios
INDEX_DIR = os.path.join(BASE_DIR, 'data', 'indices')  # Índices TF-IDF y BM25F
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

# Stopwords
STOPWORDS_PATH = os.path.join(BASE_DIR, 'data', 'stopwords.txt')

# Diccionario de sinónimos para expansión semántica
SYNONYMS_PATH = os.path.join(BASE_DIR, 'expansion', 'dictionary.json')

# Rendimiento y extensiones
NUM_WORKERS = 4  # Número de procesos para extracción y preprocesado
PDF_EXTENSIONS = ['.pdf']  # Extensiones válidas
