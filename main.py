import sys

from extractor.pdf_extractor import extract_all_texts
from indexador.tfidf_index import build_tfidf_index, load_tfidf_index
from indexador.bm25f_index import build_bm25f_index, load_bm25f_index
from buscador.search_engine import search
from expansion.semantic_expand import expand_query
from indexador.fasttext_index import build_fasttext_index

def mostrar_menu():
    print("\n=== Buscador Semántico ===")
    print("1. Indexar documentos")
    print("2. Buscar consulta")
    print("3. Salir")


def opcion_indexar():
    print("[1/3] Extrayendo texto de PDFs...")
    processed = extract_all_texts()
    print(f"   Documentos procesados: {len(processed)}")

    print("[2/3] Construyendo índice TF-IDF...")
    build_tfidf_index()

    print("[3/3] Construyendo índice BM25F...")
    build_bm25f_index()

    build_fasttext_index()
    

    print("Indexación completada.")


def opcion_buscar():
    try:
        # Cargar índices si no están en memoria
        load_tfidf_index()
        load_bm25f_index()
    except Exception:
        pass

    query = input("Ingresa tu consulta: ").strip()
    if not query:
        print("La consulta no puede estar vacía.")
        return

    try:
        top_n = int(input("Cantidad de resultados [enter=10]: ") or 10)
    except ValueError:
        print("Valor inválido. Usando top_n=10.")
        top_n = 10

    try:
        peso = float(input("Peso TF-IDF [0.0-1.0, enter=0.5]: ") or 0.5)
    except ValueError:
        print("Valor inválido. Usando peso=0.5.")
        peso = 0.5

    print(f"Buscando '{query}' (top {top_n}, peso TF-IDF {peso})")
    results = search(query, top_n=top_n, tfidf_weight=peso)

    if not results:
        print("No se encontraron documentos relevantes.")
        return

    print("\nResultados:")
    for idx, (doc_id, score) in enumerate(results, start=1):
        print(f"{idx}. {doc_id} — Score combinado: {score:.4f}")


def main():
    while True:
        mostrar_menu()
        choice = input("Selecciona una opción: ").strip()
        if choice == '1':
            opcion_indexar()
        elif choice == '2':
            opcion_buscar()
        elif choice == '3':
            print("Saliendo...")
            sys.exit(0)
        else:
            print("Opción no válida. Intenta de nuevo.")


if __name__ == "__main__":
    main()

