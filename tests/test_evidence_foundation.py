"""Synthetic unit fixtures; real financial evidence is checked separately below."""

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.benchmark import Benchmark, bind_benchmark, review_markdown
from src.evidence import resolve_evidence, validate_document
from src.ingestion import pipeline
from src.ingestion.pipeline import FilingMetadata, ProcessedFiling, ingest


@pytest.fixture
def filing(tmp_path, monkeypatch):
    pdf = tmp_path / "synthetic.pdf"
    pdf.write_bytes(b"Unit test only: mocked PDF extraction")
    raw = [
        {"page": 1, "text": "Header 2024 (millions)\n" + "Filler words.\n" * 22 + "Net cash (42)"},
        {"page": 2, "text": "Separate table 2023\nTotal sales 99"},
    ]
    monkeypatch.setattr(pipeline, "extract_pages", lambda _: raw)
    metadata = FilingMetadata(company="Example", ticker="TEST", fiscal_year=2024,
                              filing_date="2024-11-01", source_url="https://example.org/fixture.pdf",
                              sec_url="https://example.org/fixture")
    return ingest(pdf, metadata, max_chars=100)


@pytest.fixture
def benchmark(filing):
    return Benchmark.model_validate({
        "version": "0.1.0", "status": "draft_pending_human_review",
        "source_sha256": filing.source_sha256, "review_method": "synthetic unit fixture",
        "items": [{
            "id": "test-1", "question": "What is net cash?", "category": "table",
            "answerable": True, "expected_answer": "Negative 42 million.",
            "expected_claims": ["Net cash is negative 42 million."], "difficulty": "easy",
            "notes": "Synthetic fixture, not research evidence.",
            "evidence": [{"page": 1, "quote": "Header 2024 (millions)"},
                         {"page": 1, "quote": "Net cash (42)"}],
        }, {
            "id": "test-2", "question": "Future sales?", "category": "unsupported",
            "answerable": False, "expected_answer": None, "expected_claims": [],
            "difficulty": "easy", "notes": "Outside fixture scope.", "evidence": [],
        }],
    })


def test_page_expansion_preserves_exact_text_and_does_not_change_chunks(filing):
    before = filing.model_dump_json()
    target = next(c for c in filing.chunks if "Net cash (42)" in c.text)
    chunk_only = resolve_evidence(filing, [target.chunk_id])
    assert "Header 2024" not in chunk_only["spans"][0]["text"]
    result = resolve_evidence(filing, [target.chunk_id], context="page")
    span = result["spans"][0]
    assert "Header 2024 (millions)" in span["text"] and "Net cash (42)" in span["text"]
    assert span["text"] == filing.pages[0].raw_text[span["raw_start"]:span["raw_end"]]
    assert span["section"] is None
    assert result["metadata"]["fiscal_year"] == 2024
    assert result["context_chars"] == len(filing.pages[0].text)
    assert filing.model_dump_json() == before


def test_page_context_deduplicates_and_never_guesses_next_page(filing):
    ids = [c.chunk_id for c in filing.chunks if c.page == 1]
    bundle = resolve_evidence(filing, ids + ids, context="page")
    assert bundle["selected_chunk_ids"] == ids
    assert [s["page"] for s in bundle["spans"]] == [1]
    explicit = resolve_evidence(filing, ids + [filing.chunks[-1].chunk_id], context="page")
    assert [s["page"] for s in explicit["spans"]] == [1, 2]
    assert explicit["context_chars"] == sum(len(p.text) for p in filing.pages)


def test_context_budget_is_exact_and_does_not_truncate(filing):
    cid = filing.chunks[0].chunk_id
    size = len(filing.pages[0].text)
    assert resolve_evidence(filing, [cid], context="page", max_chars=size)["context_chars"] == size
    with pytest.raises(ValueError, match="Nothing was truncated"):
        resolve_evidence(filing, [cid], context="page", max_chars=size - 1)


@pytest.mark.parametrize("ids,options", [([], {}), (["unknown"], {}),
                                         (None, {"context": "automatic"}), (None, {"max_chars": 0})])
def test_invalid_context_requests_fail(filing, ids, options):
    with pytest.raises(ValueError):
        resolve_evidence(filing, ids if ids is not None else [filing.chunks[0].chunk_id], **options)


@pytest.mark.parametrize("corruption", ["duplicate_id", "id", "metadata", "printed_page", "duplicate_page"])
def test_invalid_document_identity_is_rejected(filing, corruption):
    if corruption == "duplicate_id":
        filing.chunks.append(filing.chunks[0])
    elif corruption == "id":
        filing.chunks[0].chunk_id = "wrong"
    elif corruption == "metadata":
        filing.chunks[0].ticker = "WRONG"
    elif corruption == "printed_page":
        filing.chunks[0].printed_page = 999
    else:
        filing.pages.append(filing.pages[0])
    with pytest.raises(ValueError):
        validate_document(filing)


def test_benchmark_binds_quotes_to_multiple_chunks_without_claiming_review(filing, benchmark):
    result = bind_benchmark(filing, benchmark)
    assert result == bind_benchmark(filing, benchmark)
    assert result["pending_review_count"] == 2
    assert result["ready_for_scored_evaluation"] is False
    assert result["items"][1]["supporting_chunk_ids"] == []
    evidence = result["items"][0]
    assert len(evidence["supporting_chunk_ids"]) == 2
    assert evidence["page_context_chars"] == len(filing.pages[0].text)
    assert evidence["page_context_error"] is None
    for anchor in evidence["evidence"]:
        assert filing.pages[anchor["page"] - 1].raw_text[anchor["raw_start"]:anchor["raw_end"]] == anchor["quote"]
    review = review_markdown(result)
    assert "pending" in review and "Net cash (42)" in review
    assert evidence["supporting_chunk_ids"][0] in review


def test_rechunking_rebinds_ids_and_changes_manifest(filing, benchmark, tmp_path, monkeypatch):
    before = bind_benchmark(filing, benchmark)
    monkeypatch.setattr(pipeline, "extract_pages", lambda _: [
        {"page": p.page, "text": p.raw_text} for p in filing.pages
    ])
    rechunked = ingest(filing.source_path, filing.metadata, max_chars=1000)
    after = bind_benchmark(rechunked, benchmark)
    assert before["benchmark_model_sha256"] == after["benchmark_model_sha256"]
    assert before["document_model_sha256"] != after["document_model_sha256"]
    assert len(after["items"][0]["supporting_chunk_ids"]) == 1
    assert before["items"][0]["supporting_chunk_ids"] != after["items"][0]["supporting_chunk_ids"]


@pytest.mark.parametrize("corruption", ["missing_quote", "ambiguous_quote", "missing_page", "missing_chunk", "source", "order"])
def test_invalid_benchmark_evidence_never_gets_bound(filing, benchmark, corruption):
    if corruption == "missing_quote":
        benchmark.items[0].evidence[0].quote = "fabricated"
    elif corruption == "ambiguous_quote":
        benchmark.items[0].evidence[0].quote = "Filler words."
    elif corruption == "missing_page":
        benchmark.items[0].evidence[0].page = 99
    elif corruption == "missing_chunk":
        filing.chunks.pop(0)
    elif corruption == "source":
        benchmark.source_sha256 = "0" * 64
    else:
        benchmark.items[0].evidence.reverse()
    with pytest.raises(ValueError):
        bind_benchmark(filing, benchmark)


def test_review_requires_explicit_attribution_and_consistent_labels(benchmark):
    payload = benchmark.model_dump(mode="json")
    payload["status"] = "human_reviewed"
    with pytest.raises(ValidationError):
        Benchmark.model_validate(payload)
    for item in payload["items"]:
        item["review_status"] = "human_reviewed"
    with pytest.raises(ValidationError):
        Benchmark.model_validate(payload)
    for item in payload["items"]:
        item["reviewer"] = "Synthetic reviewer for unit test only"
        item["reviewed_on"] = "2024-11-02"
    assert Benchmark.model_validate(payload).status == "human_reviewed"


def test_benchmark_context_budget_failure_is_not_reported_as_zero_cost(filing, benchmark, monkeypatch):
    text = filing.pages[0].raw_text.replace("Filler words.", "Filler words." * 40)
    monkeypatch.setattr(pipeline, "extract_pages", lambda _: [{"page": 1, "text": text}])
    expanded = ingest(filing.source_path, filing.metadata, max_chars=100)
    item = bind_benchmark(expanded, benchmark)["items"][0]
    assert item["page_context_chars"] is None
    assert "budget is 8000" in item["page_context_error"]


@pytest.mark.parametrize("corruption", ["no_answer", "no_evidence", "unsupported_answer", "duplicate_id"])
def test_benchmark_label_validation(benchmark, corruption):
    payload = benchmark.model_dump(mode="json")
    if corruption == "no_answer":
        payload["items"][0]["expected_answer"] = None
    elif corruption == "no_evidence":
        payload["items"][0]["evidence"] = []
    elif corruption == "unsupported_answer":
        payload["items"][1]["expected_answer"] = "Invented answer"
    else:
        payload["items"][1]["id"] = payload["items"][0]["id"]
    with pytest.raises(ValidationError):
        Benchmark.model_validate(payload)


def test_real_benchmark_and_cash_flow_context():
    root = Path(__file__).resolve().parents[1]
    path = root / "data/processed/aapl-2024-10k.json"
    if not path.exists():
        pytest.skip("Run the documented ingestion command to enable artifact integration checks")
    document = ProcessedFiling.model_validate_json(path.read_text(encoding="utf-8"))
    benchmark = Benchmark.model_validate_json((root / "data/benchmark/aapl-2024-10k.v1.json").read_text(encoding="utf-8"))
    bound = bind_benchmark(document, benchmark)
    assert bound["item_count"] == 25 and bound["answerable_count"] == 20
    assert bound["pending_review_count"] == 25 and not bound["ready_for_scored_evaluation"]
    assert all(not item["page_context_error"] for item in bound["items"])
    tax = next(c for c in document.chunks if c.page == 36 and "Cash paid for income taxes, net" in c.text)
    bundle = resolve_evidence(document, [tax.chunk_id], context="page")
    assert bundle["context_chars"] == 2195
    assert "2024 2023 2022" in bundle["spans"][0]["text"]
    assert "Cash paid for income taxes, net $ 26,102 $ 18,679 $ 19,573" in bundle["spans"][0]["text"]
    # Adjacent real investment tables have different years. Never carry a 2024
    # header onto the 2023 table by automatically including the preceding page.
    investment = next(c for c in document.chunks if c.page == 40)
    separate = resolve_evidence(document, [investment.chunk_id], context="page")
    assert [s["page"] for s in separate["spans"]] == [40]
    assert separate["spans"][0]["text"].startswith("2023\n")
    pdf = root / "data/raw/aapl-2024-10k.pdf"
    if pdf.exists():
        assert hashlib.sha256(pdf.read_bytes()).hexdigest() == benchmark.source_sha256
