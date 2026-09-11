from pydantic import BaseModel


class Chunk(BaseModel):
    """
    A single piece of source evidence in the Provenance corpus.
    """

    chunk_id: str
    text: str

    # Company / document identity
    company: str
    ticker: str
    document_type: str

    # Time
    fiscal_year: int
    filing_date: str

    # Location within the source
    section: str | None = None
    page: int
    source_path: str
    source_sha256: str = ""
    source_url: str = ""
    printed_page: int | None = None
    # Half-open character offsets in the persisted raw PDF page extraction.
    raw_start: int = 0
    raw_end: int = 0
