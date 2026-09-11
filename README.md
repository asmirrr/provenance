# Provenance

Traceable financial intelligence, built from source evidence upward. The current
milestone ingests one real SEC filing PDF into page-preserving text and citation-ready
chunks. Retrieval, LLMs, and claim verification are intentionally not implemented.

## Run locally (PowerShell)

Python 3.12 and uv are used by this repository. Install the locked environment:

```powershell
uv sync --locked
```

The example is Apple Inc.'s FY2024 Form 10-K, filed November 1, 2024,
[SEC accession 0000320193-24-000123](https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/0000320193-24-000123-index.htm).
The PDF is the issuer-hosted as-filed version, including exhibits; SEC's primary
filing is HTML. Document metadata and both source URLs are checked in separately.

```powershell
New-Item -ItemType Directory -Force data/raw | Out-Null
Invoke-WebRequest -Uri 'https://s2.q4cdn.com/470004039/files/doc_earnings/2024/q4/filing/10-Q4-2024-As-Filed.pdf' -OutFile data/raw/aapl-2024-10k.pdf
uv run python -m src.ingestion.pipeline data/raw/aapl-2024-10k.pdf data/raw/aapl-2024-10k.metadata.json data/processed/aapl-2024-10k.json
uv run pytest -q
```

PDFs and derived output are local, Git-ignored artifacts. Tests require no network.
The real-filing integration test runs when the PDF is present and explicitly skips
otherwise. Synthetic unit-test strings are fixtures, not financial research data.

## Representation and citation contract

One JSON file contains metadata, source SHA-256, parser version, chunk size,
warnings, every raw and cleaned page, and validated `Chunk` objects. Reopen it with
`ProcessedFiling.model_validate_json(path.read_text(encoding="utf-8"))`.

- `page` is the **one-based PDF page**, including cover pages; `printed_page` is
  the report footer's page number when recognized.
- Each chunk belongs to one page and at most one detected SEC Item section.
  Sections before the first heading and after signatures are nullable. The table
  of contents does not assign section labels. This is a conservative heading
  heuristic, not a general filing structure parser.
- `raw_start:raw_end` are half-open Python character offsets into that page's
  `raw_text`. The slice equals chunk text exactly. They are not PDF byte offsets
  or visual bounding boxes. The source digest identifies the exact PDF bytes.
- Chunk IDs are SHA-256 of source digest, PDF page, and raw offsets. Reprocessing
  the same source and settings produces the same IDs. Metadata includes company,
  ticker, fiscal year, filing date, document type, source URL, and local path.
- Chunks default to at most 1,800 characters, prefer line boundaries in the latter
  half of the window, then whitespace, and hard-split only oversized tokens. There
  is no overlap. All non-whitespace cleaned text is covered in source order.

## Observed extraction and preprocessing

The downloaded filing contains **121 PDF pages and 409,559 extracted characters**.
The signed main report ends on PDF page 60 (printed page 57); exhibits follow.
No page was empty. PDF page 3 is the table of contents, page 24 begins Item 7,
and page 32 is the consolidated statements of operations (printed page 29).

Text is generally readable, but paragraph breaks are often single newlines and
line-wrapped words can retain hyphens. Financial rows retain labels and numbers:
`Total net sales 391,035 383,285 394,328` appears on PDF page 32. Visual inspection
of that page confirms the three year columns, but extracted text does not encode
their column geometry. Units and year headings remain in text; no structured-table
accuracy is claimed. Chunks can separate a table header from its rows.

Cleaning only strips outer whitespace and the anchored, company/year-specific
report footer. Raw extraction is always preserved. Internal line breaks, currency,
parenthesized negatives, bullets, standalone numbers, and hyphenation remain intact.
No OCR, inferred financial values, dehyphenation, or broad header/footer deletion
is performed. Empty pages produce warnings; all-empty documents fail explicitly.

The next milestone should evaluate citation and chunk quality on a small manually
checked set of passages and financial tables from this filing, especially whether
table headers need to stay with rows, before adding retrieval.
