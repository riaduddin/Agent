# backend/app/services/doc_processing_helpers/pdf_utils.py
import io
import logging
from typing import List, Tuple
from PyPDF2 import PdfReader, PdfWriter

logger = logging.getLogger(__name__)

def split_pdf(pdf_content: bytes, chunk_size: int = 2) -> List[Tuple[bytes, int, int]]:
    """
    Splits PDF content into chunks of specified page size.
    Args:
        pdf_content: Bytes of the PDF.
        chunk_size: Number of pages per chunk.
    Returns:
        A list of tuples, where each tuple contains (chunk_bytes, start_page, end_page).
    """
    pdf_chunks = []
    try:
        reader = PdfReader(io.BytesIO(pdf_content))
        total_pages = len(reader.pages)
        if total_pages == 0:
            logger.warning("PDF has no pages.")
            return []

        for i in range(0, total_pages, chunk_size):
            writer = PdfWriter()
            start_page = i + 1
            end_page = min(i + chunk_size, total_pages)
            for page_num in range(i, end_page):
                writer.add_page(reader.pages[page_num])

            chunk_buffer = io.BytesIO()
            writer.write(chunk_buffer)
            pdf_chunks.append((chunk_buffer.getvalue(), start_page, end_page))
            chunk_buffer.close()

        logger.info(f"Split PDF into {len(pdf_chunks)} chunks (chunk_size={chunk_size}).")
        return pdf_chunks

    except Exception as e:
        logger.error(f"Failed to split PDF: {e}", exc_info=True)
        return []
