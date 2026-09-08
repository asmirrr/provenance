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
    section: str
    page: int
    source_path: str