from src.ingestion.pdf_parser import extract_pages
from unittest.mock import MagicMock, patch


def test_extract_pages_preserves_empty_pages_and_order(tmp_path):
    path = tmp_path / "fixture.pdf"
    path.touch()
    with patch("src.ingestion.pdf_parser.pdfplumber.open") as open_pdf:
        pages = [MagicMock(), MagicMock(), MagicMock()]
        for page, value in zip(pages, ["First", None, "Third"]):
            page.extract_text.return_value = value
        open_pdf.return_value.__enter__.return_value.pages = pages
        assert extract_pages(path) == [
            {"page": 1, "text": "First"}, {"page": 2, "text": ""},
            {"page": 3, "text": "Third"},
        ]


def test_extract_pages_requires_existing_file():
    try:
        extract_pages("does_not_exist.pdf")
    except FileNotFoundError:
        return

    raise AssertionError("Expected FileNotFoundError")
