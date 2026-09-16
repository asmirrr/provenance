"""Synthetic evidence coverage and saved-run validation checks."""

from copy import deepcopy

import pytest

from src.benchmark import Benchmark
from src.ingestion import pipeline
from src.retrieval_evaluation import evaluate, page_coverage, replay


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    pdf = tmp_path / "fixture.pdf"
    pdf.write_bytes(b"synthetic fixture")
    raw = [{"page": 1, "text": "Header units\n" + "Filler. " * 30 + "\nTarget row"},
           {"page": 2, "text": "Separate year footnote"}]
    monkeypatch.setattr(pipeline, "extract_pages", lambda _: raw)
    doc = pipeline.ingest(pdf, pipeline.FilingMetadata(
        company="Fixture", ticker="TEST", fiscal_year=2024, filing_date="2024-11-01",
        source_url="https://example.org/pdf", sec_url="https://example.org/filing"), max_chars=100)
    benchmark = Benchmark.model_validate({"version": "0.1.0", "status": "draft_pending_human_review",
        "source_sha256": doc.source_sha256, "review_method": "Synthetic test",
        "items": [{"id": "one", "question": "Target row?", "category": "table", "answerable": True,
                   "expected_answer": "Fixture", "expected_claims": ["Fixture"], "difficulty": "easy",
                   "notes": "Synthetic test", "evidence": [{"page": 1, "quote": "Header units"},
                                                             {"page": 1, "quote": "Target row"}]}]})
    from src.benchmark import bind_benchmark
    item = bind_benchmark(doc, benchmark)["items"][0]
    return doc, benchmark, item


def test_page_expansion_recovers_unretrieved_row_with_exact_citation(inputs):
    doc, _, item = inputs
    cid = doc.chunks[0].chunk_id
    assert "Target row" not in doc.chunks[0].text
    result = page_coverage(doc, item, [cid])
    assert result["complete_group_ids"] == ["primary"]
    assert result["selected_pages"] == [1]
    span = result["bundle"]["spans"][0]
    assert span["text"] == doc.pages[0].raw_text[span["raw_start"]:span["raw_end"]]
    assert result["bundle"]["selected_chunk_ids"] == [cid]


def test_budget_boundary_deduplication_and_no_truncation(inputs):
    doc, _, item = inputs
    ids = [c.chunk_id for c in doc.chunks if c.page == 1]
    size = len(doc.pages[0].text)
    assert page_coverage(doc, item, ids, size)["delivered_chars"] == size
    failure = page_coverage(doc, item, ids, size - 1)
    assert failure["status"] == "over_budget" and failure["bundle"] is None
    assert failure["required_chars"] == size and failure["delivered_chars"] == 0
    assert failure["complete_group"] is False


def test_alternative_groups_never_mix_and_adjacent_pages_are_not_added(inputs):
    doc, _, item = inputs
    second = {"page": 2, "raw_start": 0, "raw_end": len(doc.pages[1].text), "quote": doc.pages[1].text}
    item["evidence_groups"][0]["evidence"].append(second)
    result = page_coverage(doc, item, [doc.chunks[0].chunk_id])
    assert result["selected_pages"] == [1] and not result["complete_group"]
    both = page_coverage(doc, item, [doc.chunks[0].chunk_id, doc.chunks[-1].chunk_id])
    assert both["complete_group"]
    item["evidence_groups"].append({"group_id": "alternative", "evidence": [second]})
    assert page_coverage(doc, item, [doc.chunks[-1].chunk_id])["complete_group_ids"] == ["alternative"]


def test_empty_unsupported_and_invalid_selections(inputs):
    doc, _, item = inputs
    assert page_coverage(doc, item, [])["status"] == "no_results"
    item["answerable"] = False
    assert page_coverage(doc, item, [doc.chunks[0].chunk_id])["complete_group"] is None
    for ids, budget in [(["unknown"], 8000), ([doc.chunks[0].chunk_id] * 2, 8000), ([], 0)]:
        with pytest.raises(ValueError):
            page_coverage(doc, item, ids, budget)


def test_replay_preserves_raw_metrics_and_recomputes_context(inputs):
    doc, benchmark, _ = inputs
    artifact = evaluate(doc, benchmark, retriever="bm25", allow_draft=True)
    original = deepcopy(artifact)
    result = replay(doc, benchmark, artifact, allow_draft=True, max_chars=1)
    assert result["aggregate"] == artifact["aggregate"]
    assert result["page_context_aggregate"]["1"]["over_budget_count"] == 1
    assert result["page_context_aggregate"]["1"]["complete_group_count"] == 0
    assert artifact == original
    with pytest.raises(ValueError, match="allow-draft"):
        replay(doc, benchmark, artifact)


@pytest.mark.parametrize("corruption", ["hash", "id", "question", "citation", "rank", "duplicate"])
def test_replay_rejects_stale_or_corrupt_runs(inputs, corruption):
    doc, benchmark, _ = inputs
    artifact = evaluate(doc, benchmark, retriever="bm25", allow_draft=True)
    run = artifact["items"][0]["retrieval"]
    if corruption == "hash":
        artifact["benchmark_model_sha256"] = "stale"
    elif corruption == "id":
        artifact["items"][0]["item_id"] = "different"
    elif corruption == "question":
        run["question"] = "different"
    elif corruption == "citation":
        run["results"][0]["chunk"]["text"] = "altered"
    elif corruption == "rank":
        run["results"][0]["rank"] = 2
    else:
        duplicate = deepcopy(run["results"][0])
        duplicate["rank"] = 2
        run["results"].append(duplicate)
    with pytest.raises(ValueError):
        replay(doc, benchmark, artifact, allow_draft=True)


def test_replay_cli_does_not_load_model_and_protects_input(inputs, tmp_path, monkeypatch):
    import json
    from src import retrieval_evaluation as module
    doc, benchmark, _ = inputs
    document_path, benchmark_path, run_path = [tmp_path / name for name in ("doc.json", "benchmark.json", "run.json")]
    document_path.write_text(doc.model_dump_json(), encoding="utf-8")
    benchmark_path.write_text(benchmark.model_dump_json(), encoding="utf-8")
    run_path.write_text(json.dumps(evaluate(doc, benchmark, retriever="bm25", allow_draft=True)), encoding="utf-8")
    monkeypatch.setattr(module, "load_model", lambda *a: pytest.fail("Replay must not load a model"))
    args = ["evaluation", str(document_path), str(benchmark_path), "--retrieval-run", str(run_path), "--allow-draft"]
    monkeypatch.setattr("sys.argv", args + ["--output", str(run_path)])
    before = run_path.read_bytes()
    with pytest.raises(SystemExit):
        module.main()
    assert run_path.read_bytes() == before
    output = tmp_path / "context.json"
    monkeypatch.setattr("sys.argv", args + ["--output", str(output)])
    module.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["retriever"] == "bm25"
    assert result["replayed_from"]["sha256"]
    assert result["page_context_policy"]["max_chars"] == 8000
