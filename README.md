# Provenance

Traceable financial intelligence, built from source evidence upward. The current
milestone ingests one real SEC filing PDF into page-preserving text and citation-ready
chunks, with an offline evidence-context diagnostic set. Retrieval, LLMs, and claim
verification are intentionally not implemented.

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

One JSON file contains metadata, source SHA-256, parser version, chunking strategy, chunk size,
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
- Chunks default to at most 1,800 characters. `sentence-v2` prefers sentence endings
  in the latter half of the window, then line boundaries and whitespace, and
  hard-splits only oversized tokens. Obvious abbreviations such as `U.S.` and
  `Inc.` are excluded; this is a heuristic, not a complete sentence tokenizer.
  There is no overlap. All non-whitespace cleaned text is covered in source order.
  `--chunking line-v1` reproduces the initial line-based baseline. Older JSON files
  without a strategy field are interpreted as `line-v1`.

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

## Evaluate evidence context

Eight source-backed cases in `data/evaluation/aapl-2024-10k.json` select verbatim
fragments independently of chunk boundaries: financial values plus their years,
units and headings, and two prose passages with their headings. The source PDF
SHA-256 is pinned. These cases were selected and checked by the coding agent against
the raw extraction; they are **not human-annotated ground truth or a held-out benchmark**.
They include known failures deliberately and should not be interpreted as a sample
of overall filing accuracy. No LLM calls or additional dependencies are required.

```powershell
uv run python -m src.evaluation data/processed/aapl-2024-10k.json data/evaluation/aapl-2024-10k.json --output data/processed/chunk-quality-after.json
```

To reproduce the original comparison, without replacing the current ingestion:

```powershell
uv run python -m src.ingestion.pipeline data/raw/aapl-2024-10k.pdf data/raw/aapl-2024-10k.metadata.json data/processed/aapl-2024-10k-line-v1.json --chunking line-v1
uv run python -m src.evaluation data/processed/aapl-2024-10k-line-v1.json data/evaluation/aapl-2024-10k.json --output data/processed/chunk-quality-before.json
```

Measured with pdfplumber 0.11.10 and a 1,800-character limit:

| Diagnostic | line-v1 | sentence-v2 |
| --- | ---: | ---: |
| Filing chunks | 302 | 314 |
| Cases with all evidence in one chunk | 3 / 8 | 7 / 8 |
| Cases with all evidence on the cleaned page | 8 / 8 | 8 / 8 |
| Cases with expected SEC section labels | 8 / 8 | 8 / 8 |

The evaluator returns exact page spans and chunk IDs for each case. Missing or
ambiguous quote anchors, source-digest mismatches and invalid chunk-to-raw-text
citations fail instead of receiving a score. It validates against the stored raw
text; it does not independently authenticate the PDF or re-extract it. Page context
completeness is an upper bound on available context, not retrieval performance.

The new strategy keeps the SG&A heading with its sentence on PDF page 27 and the
repatriation-tax passage together on page 28. It also improves the selected table
cases on pages 39 and 48 by placing boundaries before their introductions. This does
not amount to table detection: some long tables still require multiple chunks, and
financial note headings remain under their containing SEC Item label.

The remaining failure is PDF page 36: units, years and the supplemental cash tax
payment row span 2,085 characters, exceeding the chunk limit. The next milestone
should add explicitly cited context for table continuations and review a broader
set of passages before introducing a retrieval baseline. Merely increasing chunk
size until these eight cases pass would not establish general quality.
