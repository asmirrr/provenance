from src.ingestion.pdf_parser import extract_pages


def test_extract_pages_requires_existing_file():
    try:
        extract_pages("does_not_exist.pdf")
    except FileNotFoundError:
        return

    raise AssertionError("Expected FileNotFoundError")