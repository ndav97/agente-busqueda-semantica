import re
import os
import unicodedata
from typing import List

from config import STOPWORDS_PATH


# Cargar stopwords una sola vez
def load_stopwords(path: str = STOPWORDS_PATH) -> set:
    stopwords = set()
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                token = line.strip().lower()
                if token:
                    stopwords.add(token)
    return stopwords


STOPWORDS = load_stopwords()


def strip_accents(text: str) -> str:
    """Elimina acentos y diacríticos"""
    normalized = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in normalized if not unicodedata.combining(c))


def clean_text(text: str) -> str:
    """
    Limpia el texto:
    - Normaliza acentos
    - Convierte a minúsculas
    - Elimina caracteres no alfabéticos
    """
    text = strip_accents(text)
    text = text.lower()
    # Reemplazar todo lo que no sea letra por espacio
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Colapsar espacios múltiple
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """
    Tokeniza el texto limpio por espacios
    """
    if not text:
        return []
    return text.split(' ')


def preprocess_text(text: str) -> List[str]:
    """
    Pipeline completo de preprocesado:
    - Limpieza
    - Tokenización
    - Eliminación de stopwords
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    # Filtrar stopwords y tokens muy cortos
    return [t for t in tokens if t and t not in STOPWORDS]