from pathlib import Path

import pdfplumber


def extract_pages(pdf_path: str | Path) -> list[dict]:
    """
    Extract text from a PDF while preserving page boundaries.

    Returns:
        A list of dictionaries containing page number and raw text.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages