# Provenance

Traceable financial intelligence, built from source evidence upward. The current
milestone ingests one real SEC filing PDF into page-preserving text and citation-ready
chunks, with an offline evidence-context diagnostic set and a first dense retrieval
baseline, BM25 lexical retrieval, a fixed hybrid comparison, and a query command
that exports bounded cited evidence. LLM generation
and claim verification are not implemented.

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
experiment now compares this explicit prefix baseline with encoding that covers
complete chunks, using the batch diagnostics below. All 28 benchmark items still require human review before results can be
presented as evaluation against human-reviewed ground truth.

### Full-chunk encoding and batch diagnostics

Use `--encoding window-mean` instead of `--allow-truncation` to encode complete
chunks. Text is recursively divided near its midpoint, preferring nearby whitespace,
until each window fits the tokenizer's limit including special tokens. Oversized
tokens can be split at character boundaries. Windows cover every character in
order, without overlap, rewriting, or changing persisted chunks and citation IDs.
The same policy applies to oversized questions.

Each window vector is normalized, then averaged with weights proportional to
window character lengths. Ranking uses cosine similarity of those pooled vectors.
The artifact records window character offsets relative to each chunk (add the
chunk's `raw_start` for page offsets), token counts and pooling policy. Full text
coverage does not preserve all relationships across window boundaries and does
not guarantee better retrieval. The prefix policy remains available for comparison.

Run both policies against the current draft labels:

```powershell
uv run python -m src.retrieval_evaluation data/processed/aapl-2024-10k.json data/benchmark/aapl-2024-10k.v1.json --encoding window-mean --allow-draft --local-files-only --source-pdf data/raw/aapl-2024-10k.pdf --output data/processed/dense-window-diagnostic.json
uv run python -m src.retrieval_evaluation data/processed/aapl-2024-10k.json data/benchmark/aapl-2024-10k.v1.json --encoding prefix --allow-draft --local-files-only --source-pdf data/raw/aapl-2024-10k.pdf --output data/processed/dense-prefix-diagnostic.json
```

These commands require the model downloaded by the initial search command.
Each batch embeds the corpus once and all questions together. Repeated timing
fields in per-question artifacts describe the shared batch, not independent runs;
do not sum them. Ranking time is per question. Outputs retain ranked chunks and
scores, corpus/benchmark hashes, model revision and label-review status. Draft
benchmarks are rejected unless `--allow-draft` is explicit; this never updates review labels.

Metrics use only the 23 answerable questions:

- `best_group_recall_at_k`: largest fraction of required chunk IDs recovered in
  any one evidence group; alternatives are never combined into a mandatory union.
- `complete_group_at_k`: whether at least one whole evidence group is retrieved.
- `reciprocal_rank_at_10`: reciprocal of the first rank matching any annotated
  chunk, or zero when none occurs in the top 10. Its mean is truncated MRR;
  finding one chunk does not establish complete support.

Five unsupported questions retain retrieved candidates but have null metrics and
are excluded from aggregates. No abstention accuracy, answer accuracy, nDCG,
exhaustive relevance, or page-expanded coverage is claimed.

Observed development comparison on the same 314 chunks and 28 draft questions:

| Diagnostic (23 answerable questions) | Prefix | Window mean |
| --- | ---: | ---: |
| Complete group @1 | 11/23 | 9/23 |
| Complete group @3 | 14/23 | 14/23 |
| Complete group @5 | 17/23 | 15/23 |
| Complete group @10 | 21/23 | 20/23 |
| Best-group recall @10 | 0.935 | 0.913 |
| MRR @10, first annotated chunk | 0.655 | 0.611 |

Window encoding covered all 314 chunks using 530 windows with no overflow.
It **did not improve these draft-label metrics**. Its top-10 incomplete cases were
`aapl24-007` (cash balance), `aapl24-009` (cash taxes), and `aapl24-027` (cash total
plus restricted-cash footnote). This is evidence to inspect pooling and financial
context failures, not a reason to assume truncation is generally superior.

These failures motivated the BM25 comparison below, using the same fixed corpus
and diagnostic protocol. Human benchmark review remains outstanding. Avoid tuning
solely to these small, overlapping development cases.

## BM25 lexical baseline

`src.bm25` searches full chunk text locally without a model download or additional
dependencies. It builds one in-memory index per batch. Scores use term-frequency
saturation and document-length normalization from the
[BM25 formulation](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html),
with fixed `k1=1.2`, `b=0.75` and positive smoothed IDF
`log(1 + (N - df + 0.5)/(df + 0.5))`. Parameters were not tuned to these questions.

The versioned `words-numeric-v1` tokenizer casefolds words, retains years and decimal
numbers, and removes grouping commas (`391,035` matches `391035`). It uses no stop
list, stemming, synonyms, or units conversion. Signs, parentheses and percent symbols
are not indexed: this is term matching, not financial-number interpretation. Original
citation text remains unchanged. Repeated query terms count once. Equal scores are
ordered by chunk ID, and zero-overlap chunks are excluded rather than padding top-k.
An empty result is not proof that a question is unanswerable.

```powershell
uv run python -m src.bm25 data/processed/aapl-2024-10k.json "How much cash did Apple pay for income taxes, net, in fiscal 2024?" --top-k 10 --source-pdf data/raw/aapl-2024-10k.pdf --output data/processed/bm25-cash-tax.json
uv run python -m src.retrieval_evaluation data/processed/aapl-2024-10k.json data/benchmark/aapl-2024-10k.v1.json --retriever bm25 --allow-draft --source-pdf data/raw/aapl-2024-10k.pdf --output data/processed/bm25-diagnostic.json
```

The common evaluator uses the same questions, evidence groups and denominators as
the dense runs. BM25 does not load the dense model. Artifacts record tokenizer and
scoring settings, query tokens, full ranked chunks, scores, hashes and timings.
BM25 scores are not cosine similarities or calibrated probabilities; they cannot
be added to dense scores without an explicit combination policy.

All three development runs have matching corpus and benchmark model hashes:

| Diagnostic (23 answerable draft questions) | Dense prefix | Dense window mean | BM25 |
| --- | ---: | ---: | ---: |
| Complete group @1 | 11/23 | 9/23 | 9/23 |
| Complete group @3 | 14/23 | 14/23 | 15/23 |
| Complete group @5 | 17/23 | 15/23 | 19/23 |
| Complete group @10 | 21/23 | 20/23 | 23/23 |
| MRR @10, first annotated chunk | 0.655 | 0.611 | 0.643 |

Inspecting the earlier window-encoding failures:

- `aapl24-007`: the selected balance-sheet cash evidence on PDF page 34 is absent
  from dense-window top 10; BM25 finds it at rank 10. Its higher-ranked Note 4
  cash tables must not be assumed interchangeable with the balance-sheet figure.
- `aapl24-009`: the tax-payment row is separate from its table header. Dense-window
  retrieves only one required chunk; BM25 retrieves both page-36 chunks at ranks
  2 and 7. One relevant chunk alone still does not complete this evidence group.
- `aapl24-027`: dense-window finds the restriction footnote on page 40 but misses
  the page-39 table. BM25 retrieves the selected two chunks at ranks 2 and 5.

The observed BM25 index build took about 29 ms and the first query about 1 ms on
this machine; these are single-run timings, not performance benchmarks. The 23/23
result measures coverage of selected draft evidence on a small development set,
not answer accuracy, exhaustive relevance, or generalization. All 28 questions
remain pending human review, and the five unsupported questions are excluded from
these aggregates.

The fixed hybrid comparison below inspects early-ranking gains as well as top-10
coverage losses. Human source-label review remains outstanding.

## Fixed hybrid rank fusion

`src.hybrid` combines dense and BM25 candidates using
[reciprocal rank fusion (Cormack et al., 2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf):
each retrieved chunk receives `1 / (60 + rank)` from each component that retrieved
it. Missing candidates contribute zero. Both components have equal weights; the
constant 60 and candidate depth 10 are fixed, not optimized against these labels.
Ties use chunk ID. Raw BM25 and cosine scores remain separate and are never added.

```powershell
uv run python -m src.retrieval_evaluation data/processed/aapl-2024-10k.json data/benchmark/aapl-2024-10k.v1.json --retriever hybrid --encoding prefix --allow-draft --local-files-only --source-pdf data/raw/aapl-2024-10k.pdf --output data/processed/hybrid-prefix-diagnostic.json
```

This reruns both components against the same corpus and questions. It fuses the
union of their top-10 candidates and returns at most 10. Each result records which
retrievers contributed, their original ranks and scores, and each RRF contribution.
The artifact retains the full component runs, including dense truncation metadata,
and rejects inconsistent source, corpus, question or citation identities. Batch
component time is shared across questions; fusion time is measured per question.

The fused shortlist can lose evidence present in the larger candidate union.
Scores are ranking utilities, not confidence or sufficiency judgments. This
comparison uses dense-prefix explicitly; `--encoding window-mean` is available as
a separate experiment and is not silently substituted. No reranker is involved.

Observed comparison on the same 23 answerable draft questions:

| Diagnostic | Dense prefix | BM25 | Hybrid RRF |
| --- | ---: | ---: | ---: |
| Complete group @1 | 11/23 | 9/23 | 14/23 |
| Complete group @3 | 14/23 | 15/23 | 19/23 |
| Complete group @5 | 17/23 | 19/23 | 21/23 |
| Complete group @10 | 21/23 | 23/23 | 22/23 |
| MRR @10, first annotated chunk | 0.655 | 0.643 | 0.841 |

The component reruns produced identical ranks, scores and citations to the saved
baselines, with matching corpus and benchmark hashes. Replaying fusion from the
saved component lists reproduced all 28 fused rankings. No parameters or evidence
labels were changed after observing results.

Hybrid improves early coverage here, but loses BM25's complete top-10 evidence for
`aapl24-009` (cash taxes). Both methods rank the first required page-36 chunk second;
fusion promotes it to first. The other required chunk is BM25-only at rank 7 and
falls outside the fused top 10 as agreement and competing single-method candidates
take those slots. Finding the right page does not imply finding the complete
annotated chunk group. No dense-prefix complete-group rank-1 successes were lost.

This is a small draft-label development result, not proof of general improvement
or answer accuracy. Unsupported items remain excluded and human review is pending.
The bounded context diagnostic below examines this cash-tax failure separately
from raw-chunk retrieval. Human label review is still required before treating the
benchmark as ground truth.

## Page-context diagnostics from retrieved chunks

The evaluator now reports `page_context` at top 1, 3, 5 and 10 separately from
unchanged chunk-retrieval metrics. It expands only pages actually reached by the
retriever, deduplicates them, and tests every exact quote span in each evidence
group against the returned context. One complete alternative group suffices;
fragments of different groups are never mixed. Benchmark anchors are used only
to check coverage, never to select pages. No adjacent pages are inferred.

The fixed policy is **all selected pages or an explicit budget failure**, with an
8,000-character default (`--max-context-chars`). Over-budget requests return no
bundle, zero delivered characters and incomplete delivered coverage. Required
character counts and selected pages remain recorded. The denominator includes
failed deliveries; it is not restricted to successful context requests. Unsupported
questions get context diagnostics but null evidence-completeness labels and remain
excluded from aggregates. Complete context still does not establish answer accuracy.

Existing rankings can be replayed without model loading or retrieval:

```powershell
uv run python -m src.retrieval_evaluation data/processed/aapl-2024-10k.json data/benchmark/aapl-2024-10k.v1.json --retrieval-run data/processed/hybrid-prefix-diagnostic.json --allow-draft --source-pdf data/raw/aapl-2024-10k.pdf --output data/processed/hybrid-page-context.json
```

Replay requires matching corpus and benchmark hashes, item IDs/order, questions,
retrieval depth, ranks and full chunk citations. It ignores saved metrics and
recomputes them from the supplied source models and ranked IDs. The output records
the input artifact's path and byte hash; that artifact retains the original model,
settings and runtime metadata. These are consistency checks, not proof that the
saved scores were honestly generated. Output cannot overwrite supplied inputs.
For the other baselines substitute `bm25-diagnostic.json` or
`dense-prefix-diagnostic.json` and use distinct output filenames.

Observed delivered complete-group coverage under the 8,000-character policy:

| Diagnostic (23 answerable draft questions) | Dense prefix | BM25 | Hybrid |
| --- | ---: | ---: | ---: |
| Top-1 page context complete | 12/23 | 10/23 | 16/23 |
| Top-3 page context complete | 13/23 | 8/23 | 17/23 |
| Top-3 over-budget requests | 7/23 | 13/23 | 5/23 |
| Top-5 over-budget requests | 23/23 | 23/23 | 23/23 |
| Top-10 over-budget requests | 23/23 | 23/23 | 23/23 |

For `aapl24-009`, hybrid's top result expands PDF page 36 to 2,195 characters and
recovers both required fragments, despite the tax-row chunk ranking 11th in fusion.
Top-3 page expansion also fits (5,731 characters). Top-5 needs 10,425 characters
and is rejected. Raw retrieval scores remain unchanged: expanded text does not
retroactively count as retrieved chunks. The top-5/10 context failures reflect
this strict budget policy, not absence of the underlying evidence.

The ranked-fit policy below addresses these budget failures without changing the
retrieval rankings or human review status.

### Budget-aware selection of whole pages

`--page-policy ranked-fit` considers distinct pages in first-retrieved order. It
keeps a whole cleaned page if it fits the remaining character budget, otherwise
records the omission and continues to later candidates. Repeated hits on a page
do not cost extra characters or change its priority. No page is trimmed, no
adjacent page is inferred, and benchmark labels never enter the selector.
The reusable implementation is `src.evidence.select_page_context`.

```powershell
uv run python -m src.retrieval_evaluation data/processed/aapl-2024-10k.json data/benchmark/aapl-2024-10k.v1.json --retrieval-run data/processed/hybrid-prefix-diagnostic.json --page-policy ranked-fit --allow-draft --source-pdf data/raw/aapl-2024-10k.pdf --output data/processed/hybrid-ranked-context.json
```

The default remains `--page-policy strict` for reproducibility. Both policies use
the same default 8,000-character budget. Artifacts now distinguish `candidate_pages`
from actually delivered `selected_pages`. Each omitted page records its first
retrieval rank, full size, remaining budget, and reason. A partly delivered request
has status `partial`; `over_budget` means no candidate page could be delivered.
`required_chars` still counts all candidate pages, while `delivered_chars` counts
only returned spans. The page-context aggregate includes `requests_with_omissions`
so partial selection is not mistaken for delivery of every candidate.

Observed complete evidence-group coverage under ranked-fit (23 answerable draft items):

| Candidate cutoff | Dense prefix | BM25 | Hybrid |
| --- | ---: | ---: | ---: |
| Top 1 | 12/23 | 10/23 | 16/23 |
| Top 3 | 17/23 | 16/23 | 20/23 |
| Top 5 | 17/23 | 19/23 | 20/23 |
| Top 10 | 17/23 | 20/23 | 20/23 |

Every answerable request delivered some context within budget. At top 5 and top 10,
all 23 requests omitted pages. For hybrid, top-3 completeness improves from 17/23
under strict delivery to 20/23 under ranked-fit. The tax case remains complete at
top 10: pages 36, 32, 37 and 33 use 6,830 characters. Raw chunk-retrieval scores
remain unchanged, and unsupported questions still have no completeness score.

Hybrid's incomplete top-10 context cases are `aapl24-015`, `aapl24-027` and
`aapl24-028`. This greedy policy prefers early-ranked pages; it does not optimize
evidence sufficiency, diversity or combinations of pages. Adding candidates can
fill remaining space, but cannot displace an earlier accepted page. These draft
results do not establish answer accuracy or eliminate the need for human review.

Inspection found the required pages in the hybrid candidate set for all three
failures, but earlier accepted pages consumed the available budget. This motivated
the smaller-context experiment below, without editing labels or rankings.

### Whole-chunk context comparison

Evaluation and replay also report `chunk_context` and `chunk_context_aggregate`.
`src.evidence.select_chunk_context` considers whole retrieved chunks in rank order,
accepts those that fit, and records omitted IDs, ranks, sizes and reasons. It never
expands a page or changes text. The same character budget and candidate cutoffs
apply to both policies; raw retrieval metrics remain separate. Complete support
requires every bound supporting chunk in at least one evidence group to survive
selection, so dropping a header or row cannot receive page-expansion credit.

Replaying the command above regenerates both comparisons in one artifact. At top
10, the 8,000-character budget yields:

| Complete context evidence (23 answerable draft questions) | Dense prefix | BM25 | Hybrid |
| --- | ---: | ---: | ---: |
| Ranked whole pages | 17/23 | 20/23 | 20/23 |
| Ranked whole chunks | 18/23 | 17/23 | 22/23 |

For hybrid, whole chunks preserve evidence for `aapl24-015`, `aapl24-027` and
`aapl24-028`, which whole-page selection omitted. However, chunk context still
misses `aapl24-009`: its tax-row chunk was not in the fused top 10. Page expansion
recovers that row from a retrieved chunk on the same page. Both approaches omit
some candidates at top 10; these results describe delivered evidence, not an
unbounded union. No per-question oracle chooses whichever policy matches the labels.

### Query a filing without benchmark labels

From the repository directory, retrieve evidence and save the complete run:

```powershell
uv run python -m src.query data/processed/aapl-2024-10k.json "How much cash did Apple pay for income taxes in 2024?" --source-pdf data/raw/aapl-2024-10k.pdf --output data/processed/query-bm25.json
```

The default is BM25 with whole-chunk context: it requires no model download.
The command retrieves up to 10 candidates and delivers whole spans within an
8,000-character text budget. It skips candidates that cannot fit and records
every omission. This budget excludes JSON metadata and is not a token limit.
Use `--top-k` (1–10), `--max-chars`, or `--context page` to change these choices.
Page context uses ranked-fit selection, with no automatic adjacent-page expansion.

To use the existing hybrid model and whole-page context:

```powershell
uv run python -m src.query data/processed/aapl-2024-10k.json "How much cash did Apple pay for income taxes in 2024?" --retriever hybrid --encoding prefix --local-files-only --context page --source-pdf data/raw/aapl-2024-10k.pdf --output data/processed/query-hybrid.json
```

Dense and hybrid modes require an explicit `--encoding`: `prefix` permits embedding
truncation, `window-mean` covers all text, and `reject` fails on oversized inputs.
Exported citations always retain original text. `--local-files-only` requires the
pinned model to be cached; omit it to allow the first download. Models are encoded
again on each run; there is no persistent index yet.

The JSON separates `retrieval` (all candidates, scores and configuration) from
`selection.bundle` (only delivered source spans). Read `selection.status` and
omissions alongside the evidence. `ok` and `partial` describe context delivery,
not verified support; `no_results` and `over_budget` do not establish that the
question is unanswerable. No generated answer or benchmark labels are used.
PDF verification is recorded as `matched` or `not_checked`; a mismatch fails
before retrieval. Matching bytes does not independently authenticate extracted text.

### Structured answer and citation validation

`src.answer.Answer` defines a provider-independent response contract. An `answered`
response has one or more claims, each with nonempty exact citations. An `abstained`
response has no claims and a nonblank `abstention_reason`. Unknown fields and
coerced citation coordinates are rejected. This module does not generate answers.

The response fields are `schema_version: 1`, `query_sha256`, `status`, `claims`,
and optional `abstention_reason`. Each claim contains `text` and `citations`;
each citation contains `page`, `raw_start`, `raw_end`, and `quote`. Coordinates
are one-based PDF pages and half-open character offsets in raw extracted page
text, exactly as in the evidence bundle. A quote must fit within one delivered
span; use multiple citations for separate headers, rows, or chunks.

Compute `query_sha256` with `src.answer.query_digest(run)` on the entire parsed
query JSON. Whitespace/key ordering in the JSON file do not affect this digest;
rerunning retrieval produces a new artifact and requires a new binding.
`Answer.model_json_schema()` exposes the contract for later provider integration.
After preparing an answer JSON, validate it with:

```powershell
uv run python -m src.answer data/processed/aapl-2024-10k.json data/processed/query-hybrid.json data/processed/manual-answer-smoke.json --source-pdf data/raw/aapl-2024-10k.pdf --output data/processed/answer-validation-smoke.json
```

The example answer file is a local, manually prepared smoke artifact, not created
by the query command. The smoke check used PDF page 36's year/unit header and
income-tax payment row for the $26,102 million fiscal-2024 value; both citations
passed. This is an integrity check, not a benchmark accuracy measurement.

Validation checks corpus identity, retrieved citations, reconstructed context
selection, and every exact quotation. It rejects citations to omitted evidence
even when that text exists elsewhere in the filing. Output records
`citation_integrity: passed`, while `claim_support`, `answer_completeness`, and
`abstention_correctness` remain `not_checked`. A false claim with a genuine quote
can pass these mechanical checks. Hashes bind supplied artifacts; they are not
signatures and do not authenticate extraction or prove that a ranking was produced
by the recorded model. Optional PDF verification checks byte identity separately.

### Optional Claude generation

`src.generation` connects a saved query run to Anthropic's
[Messages API](https://platform.claude.com/docs/en/api/messages/create).
It uses the Python standard library, so offline ingestion and retrieval need no
additional dependencies or API credentials. Specify a model ID available to your
Anthropic account; there is intentionally no implicit model choice.

First inspect a request without sending anything:

```powershell
$model = Read-Host 'Anthropic model ID'
uv run python -m src.generation data/processed/aapl-2024-10k.json data/processed/query-hybrid.json --model $model --dry-run --source-pdf data/raw/aapl-2024-10k.pdf --output data/processed/generation-preview.json
```

For a live request, configure `ANTHROPIC_API_KEY` in the process environment.
In PowerShell 7, this prompts without echoing the key or putting it in history:

```powershell
$env:ANTHROPIC_API_KEY = Read-Host 'Anthropic API key' -MaskInput
uv run python -m src.generation data/processed/aapl-2024-10k.json data/processed/query-hybrid.json --model $model --source-pdf data/raw/aapl-2024-10k.pdf --output data/processed/generation-live-01.json
Remove-Item Env:ANTHROPIC_API_KEY
```

The live command sends the question, filing metadata, delivered spans and output
schema to Anthropic and may incur API charges. It does not send omitted candidate
text, benchmark labels, or the complete filing. Credentials are neither written
to artifacts nor loaded automatically from `.env`; `.env` files are Git-ignored.
The default output limit is 2,048 tokens (`--max-tokens`, range 1–8,192), separate
from the retrieval character budget. Calls have a 60-second timeout and no
automatic retries. A timeout may still have incurred provider usage.

The schema is included in the prompt and enforced locally, not guaranteed by
provider-side constrained decoding. The artifact preserves the request, raw
response (including model/usage when returned), and validation result. Only a
normal `end_turn` text response is parsed. Truncation, refusal, unexpected content,
invalid JSON, stale bindings or bad citations produce `invalid_response`, preserve
the response, and exit nonzero. Transport/API failures produce `provider_error`;
error bodies and credentials are not saved. Nothing is silently repaired or retried.
Every run requires a new output filename, protecting prior paid runs.

`citation_validated` means mechanical checks passed; semantic support remains
`not_checked`. No delivered evidence produces `local_abstention` without an API
call. Dry runs produce `dry_run` with no generated answer. Tests use explicitly
synthetic provider responses. A real-filing dry run has passed; live model behavior
has not yet been tested because no API key was configured during implementation.

Next: run one live question, inspect its claims/citations and failure modes, then
add a small generation evaluation over the existing benchmark. Human review is
still required before reporting validated accuracy. Semantic verification,
calculation checks, and a frontend remain later milestones.
