import os
import logging
from concurrent.futures import ProcessPoolExecutor
from PyPDF2 import PdfReader

from config import RAW_PDF_DIR, EXTRACTED_TEXT_DIR, PDF_EXTENSIONS, NUM_WORKERS

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extrae todo el texto de un archivo PDF.
    """
    try:
        reader = PdfReader(pdf_path)
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
        return "\n".join(pages_text)
    except Exception as e:
        logging.error(f"Error extrayendo texto de '{pdf_path}': {e}")
        return ""


def save_extracted_text(pdf_path: str, text: str) -> None:
    """
    Guarda el texto extraído en un archivo .txt manteniendo la misma estructura de carpetas.
    """
    rel_path = os.path.relpath(pdf_path, RAW_PDF_DIR)
    txt_path = os.path.join(
        EXTRACTED_TEXT_DIR, os.path.splitext(rel_path)[0] + '.txt')
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
    logging.info(f"Texto guardado en '{txt_path}'")


def process_pdf(pdf_path: str) -> str:
    """
    Extrae y guarda el texto de un único PDF. Retorna la ruta del PDF procesado.
    """
    text = extract_text_from_pdf(pdf_path)
    if text:
        save_extracted_text(pdf_path, text)
    return pdf_path


def extract_all_texts() -> list:
    """
    Recorre todos los PDFs en RAW_PDF_DIR y extrae su texto en paralelo.
    Devuelve la lista de rutas procesadas.
    """
    pdf_paths = []
    for root, _, files in os.walk(RAW_PDF_DIR):
        for file in files:
            if os.path.splitext(file)[1].lower() in PDF_EXTENSIONS:
                pdf_paths.append(os.path.join(root, file))

    if not pdf_paths:
        logging.warning(f"No se encontraron PDFs en {RAW_PDF_DIR}")
        return []

    os.makedirs(EXTRACTED_TEXT_DIR, exist_ok=True)
    logging.info(
        f"Iniciando extracción de {len(pdf_paths)} PDFs con {NUM_WORKERS} workers...")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for pdf in executor.map(process_pdf, pdf_paths):
            logging.info(f"Procesado: {pdf}")

    return pdf_paths


if __name__ == '__main__':
    processed = extract_all_texts()
    logging.info(f"Extracción completada. PDFs procesados: {len(processed)}")
