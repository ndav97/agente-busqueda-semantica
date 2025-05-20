import json
import os

from config import SYNONYMS_PATH

# Cargar diccionario de sinónimos una sola vez


def load_synonyms(path: str = SYNONYMS_PATH) -> dict:
    """Carga el JSON de sinónimos. Formato esperado:
    {
      "termino1": ["sinonimo1", "sinonimo2"],
      ...
    }
    """
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


SYNONYMS = load_synonyms()


def expand_query(tokens: list) -> list:
    """
    Dada una lista de tokens, agrega sinónimos basados en el diccionario.
    Devuelve la lista original + sinónimos (únicos, sin orden garantizado).
    """
    expanded = list(tokens)
    for term in tokens:
        if term in SYNONYMS:
            for syn in SYNONYMS[term]:
                # Evitar duplicados y tokens vacíos
                if syn and syn not in expanded:
                    expanded.append(syn)
    return expanded