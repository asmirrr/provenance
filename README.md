# Provenance

Traceable financial intelligence, built from source evidence upward. The current
milestone ingests one real SEC filing PDF into page-preserving text and citation-ready
chunks, with an offline evidence-context diagnostic set and a first dense retrieval
baseline. LLM generation and claim verification are not implemented.

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

The remaining single-chunk failure is PDF page 36: units, years and the supplemental
cash tax payment row span 2,085 characters, exceeding the chunk limit. This is now
handled by an explicit page-context option, described below; the original chunk
diagnostic remains 7/8. Merely increasing chunk size until these eight cases pass
would not establish general quality.

## Cited context for long tables

`src.evidence` resolves selected chunk IDs without searching or generating answers.
The default returns only those chunks. `--context page` returns each selected
chunk's entire cleaned page as a separately cited span, deduplicating repeated
pages. Source metadata, PDF/printed page numbers and exact raw offsets accompany
the text. Page context can contain multiple SEC Items, so it does not inherit a
single selected chunk's section label. The processed corpus and chunk IDs are unchanged.

For example, resolve the cash tax payment row's chunk with its page context:

```powershell
uv run python -m src.evidence data/processed/aapl-2024-10k.json d1548dced249295422ae1b9817c9a680099395017a43a83b7be256719df6f9d1 --context page --output data/processed/cash-tax-evidence.json
```

This returns 2,195 characters from PDF page 36, preserving the year columns, units,
parenthesized negatives and the final cash-tax row. On the eight existing diagnostics,
all required fragments are present for **8/8 after page expansion**, compared with
**7/8 in a single chunk**. This is an oracle-seeded context check: the evaluator
already knows the target row's chunk. It is not retrieval or financial accuracy.

Context has a separate 8,000-character default budget (`--max-chars`). An over-budget
request fails explicitly instead of silently removing text; callers must make an
explicit budget or selection decision. Counts measure characters, not model tokens.
Unknown IDs, duplicate corpus IDs, altered citations and mismatched metadata fail.
Checks establish consistency against persisted raw extraction, not PDF authenticity.

Both `src.evidence` and `src.benchmark` accept `--source-pdf data/raw/aapl-2024-10k.pdf`.
This hashes the supplied file and requires it to match the digest recorded during
ingestion **before writing outputs**. Missing or replaced files fail explicitly;
a renamed byte-identical copy passes. Exported JSON records `source_pdf_verification`
as `matched` with the checked path/digest, or `not_checked` when the option is omitted.
This verifies PDF byte identity, not whether stored extracted text is faithful to
those bytes; re-extraction is still needed to establish that independently.

This is **not table extraction or automatic multi-page continuation detection**.
PDF page 39 contains the 2024 investment table; page 40 begins a separate 2023 table
and has footnotes applying to both years. Selecting page 40 never automatically
adds page 39 or labels its rows with 2024. Multi-page evidence requires explicit
chunk selections from each relevant page. Headers, units and footnotes on other
pages may still be needed; page expansion does not establish evidence sufficiency.

## Draft benchmark and human review

`data/benchmark/aapl-2024-10k.v1.json` contains a **version 0.3.0 draft of 28 questions**:
23 answerable items and 5 unsupported/abstention items, spanning factual, numerical,
comparative, table, temporal, section-specific and combined-fact questions, plus
false premises with counterevidence. Answers, atomic expected claims, notes and
verbatim evidence anchors are included. Derived answers record their arithmetic
in notes; a calculation engine is not implemented.

All items are **pending human review**. The eight original chunk diagnostics remain
a separate development set. Neither set is held out, and their overlap prevents
treating results as an unbiased generalization measure. Many draft items use the
same financial statements; breadth and alternative supporting passages need review.

Generate a resolved JSON artifact and a readable review packet:

```powershell
uv run python -m src.benchmark data/processed/aapl-2024-10k.json data/benchmark/aapl-2024-10k.v1.json --output data/processed/benchmark-bound.json --review data/processed/benchmark-review.md
```

The binder verifies the source digest, finds unique quote anchors, checks quote
coverage by chunks, and records PDF/printed pages, raw offsets, section labels and
supporting chunk IDs. Rebinding after rechunking updates IDs instead of keeping
stale labels. Canonical model hashes for the benchmark and corpus identify the
exact inputs independently of JSON formatting. The packet includes the expected
answer, claims, excerpts, IDs and review notes for every question.

These chunk IDs are **mechanical mappings of selected evidence, not exhaustive
relevance judgments**. Other passages may answer the same question. Version 0.2.0
records alternative groups for eight questions. It also corrects the two original
multi-hop labels: the operating-cash-flow comparison is `comparative`, and the
Services share calculation is `numerical`, both medium difficulty. Both operands
can be found on one page; combining numbers does not itself require multi-hop retrieval.
Unsupported questions have no supporting IDs; lack of quote anchors does not prove
the question is unanswerable.

In the source JSON, `evidence` is the primary group and `alternative_evidence` is a
list of additional groups. All excerpts **within** a group are jointly required;
any **complete** group is an acceptable candidate support set. Units and year labels
stay with the relevant values. The binder validates and budgets each group separately
and exports them as `evidence_groups`; it never turns all alternative chunks into
one mandatory union. Top-level `supporting_chunk_ids` and `evidence` still describe
only the primary group for compatibility. New consumers should use `evidence_groups`.

`complete_evidence_groups(item, selected_chunk_ids)` checks raw-chunk coverage of a
bound item's groups. It never mixes incomplete groups, and unsupported questions
return no completed groups. This helper is a diagnostic, not a retrieval/answer score:
it does not infer correctness or credit page expansion as if more chunks were retrieved.
The review packet displays primary and alternative groups separately. All alternatives
were checked by the coding agent against the extraction and still require human review;
they are not an exhaustive list of relevant passages.

Human reviewers should check the original PDF, values, units, fiscal years, claims,
alternative evidence and abstention labels. Record `review_status: "human_reviewed"`,
`reviewer`, and `reviewed_on` (YYYY-MM-DD) in the source question item. Only mark the
overall status `human_reviewed` after all items are reviewed; update the review-method
description and version as appropriate, then regenerate artifacts. The tool requires
attribution and dates but cannot authenticate who performed the review. It never
marks generated questions reviewed automatically. Current output explicitly reports
`ready_for_scored_evaluation: false`.

The generated packet starts with counts by category (reviewed, pending and
unsupported) and a pending-item queue. Unsupported questions appear first because
their absence/scope labels need explicit review; source order is preserved within
each group. Regenerating the packet removes reviewed questions from the queue but
does not automatically finalize the benchmark. The same summary is stored under
`review_progress` in the bound JSON.

### Cross-page table cases

Version 0.3.0 adds three source-checked draft items on PDF pages 39–40 (printed
36–37), using the original table layout as well as extracted text:

- `aapl24-026`: corporate-debt fair value in 2024 versus 2023. The year tables
  are on separate pages; fair value must not be confused with adjusted cost.
- `aapl24-027`: the 2024 Note 4 cash total and its restricted portion. Footnote
  (2) is printed below the 2023 table but explicitly refers to September 28, 2024.
  The total is in millions; the restricted portion is stated in billions.
- `aapl24-028`: 2023 corporate-debt unrealized losses. Page 40 supplies the year,
  column header and parenthesized value; page 39 supplies the dates and units.

Each selected evidence group requires both pages. Tests re-extract these pages
from the original PDF and bind all three items under both chunking strategies.
They check incomplete single-page selections, explicit two-page expansion, exact
quote preservation and rejection when the context budget is too small. These are
source-backed regression checks, not measured retrieval accuracy or exhaustive
relevance judgments. All three items remain pending human review.

No automatic table stitching is needed for these cases: explicit page selection
preserves the evidence without assigning one table's year to the next. Extraction
still lacks column geometry, and the resolver does not establish that a selected
context is sufficient to answer an arbitrary question.

Next: human-review the source answers and alternate-evidence labels, then evaluate
the dense retrieval baseline with Recall@k and MRR. Keep corpus/chunk retrieval
scores separate from expanded-context coverage. Do not describe draft-label
experiments as results against human-reviewed ground truth.

## Dense retrieval baseline

`src.dense` searches the ingested filing and returns ranked chunks with cosine
scores and their complete citation metadata. It generates no answer and applies
no abstention threshold: even an unsupported question receives candidate chunks.
Similarity is not evidence sufficiency or a calibrated confidence score.

The baseline uses the existing Sentence Transformers dependency and
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
pinned to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
It runs on CPU with one PyTorch thread and deterministic algorithms, using exact
cosine ranking with chunk ID as the tie-breaker. Identical environments should
reproduce scores; bitwise equality across hardware/library versions is not promised.
The first run downloads public model weights; no API key or hosted inference is
required. Queries and filing text are embedded locally.

```powershell
uv run python -m src.dense data/processed/aapl-2024-10k.json "What were Apple total net sales in fiscal 2024?" --top-k 5 --allow-truncation --source-pdf data/raw/aapl-2024-10k.pdf --output data/processed/dense-sales.json
```

After the model is cached, add `--local-files-only` to require offline loading.
The corpus is embedded anew each run; there is no vector database or persisted
embedding cache. This keeps corpus-to-vector identity simple for this small filing.
Tests use explicitly synthetic vectors and never download a model.

**Token-limit limitation:** this model accepts 256 tokens, including special
tokens. In the current 314-chunk corpus, 206 chunks exceed that limit. By default
the command rejects oversized chunks or queries before encoding. The explicit
`--allow-truncation` flag runs a **prefix-only baseline**: suffixes of long chunks
do not contribute to the embedding. Returned citation text remains complete and
unaltered; this does not mean the encoder saw all of it. No chunking settings or
benchmark labels are changed to make the model fit.

Each JSON artifact records the question, source and canonical corpus hashes,
model revision, package versions, token limit, affected chunk IDs, per-result
truncation flags, ranks, scores, full chunks, and separate model-loading,
tokenization, corpus-encoding, query-encoding and ranking times. Top-k is capped
at the available chunk count. Optional source-PDF verification runs before loading
the model; output paths cannot overwrite the provided inputs.

The initial net-sales smoke query ranked PDF pages 26 and 32 first and second.
This demonstrates an end-to-end search, not benchmark accuracy. The next retrieval
experiment should compare this explicit prefix baseline with encoding that covers
complete chunks, then add batch evaluation with clearly defined alternative-group
metrics. All 28 benchmark items still require human review before results can be
presented as evaluation against human-reviewed ground truth.
