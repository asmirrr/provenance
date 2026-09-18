"""Fresh-ingestion audit tests use synthetic PDF extraction fixtures."""
import json
import sys

import pytest

from src.evidence import validate_document
from src.ingestion import audit as module
from test_evidence_foundation import filing


def test_matching_corpus_ignores_local_paths_and_chunk_list_order(filing):
    pdf = filing.source_path
    filing.source_path = 'relocated/original.pdf'
    for chunk in filing.chunks:
        chunk.source_path = filing.source_path
    filing.chunks.reverse()
    report = module.audit(filing, pdf)
    assert report['status'] == 'matched'
    assert not any(report['differences'].values())


def test_missing_entire_page_is_detected_despite_valid_remaining_citations(filing):
    filing.pages = filing.pages[:1]
    filing.chunks = [c for c in filing.chunks if c.page == 1]
    validate_document(filing)
    report = module.audit(filing, filing.source_path)
    assert report['status'] == 'mismatch'
    assert report['differences']['missing_pages'] == [2]
    assert report['differences']['missing_chunk_ids']


@pytest.mark.parametrize('change', ['section', 'page_order', 'extractor', 'warnings', 'schema'])
def test_other_reproducibility_differences(filing, change):
    if change == 'section':
        filing.chunks[0].section = 'Invented heading'
    elif change == 'page_order':
        filing.pages.reverse()
    elif change == 'extractor':
        filing.extractor = 'different version'
    elif change == 'warnings':
        filing.warnings = ['invented']
    else:
        filing.schema_version = 99
    validate_document(filing)
    assert module.audit(filing, filing.source_path)['status'] == 'mismatch'


def test_wrong_pdf_fails_before_extraction(filing, tmp_path, monkeypatch):
    wrong = tmp_path / 'wrong.pdf'
    wrong.write_bytes(b'wrong')
    monkeypatch.setattr(module, 'ingest', lambda *a, **kw: pytest.fail('extracted wrong PDF'))
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        module.audit(filing, wrong)


def test_consistently_edited_raw_and_chunk_text_is_detected(filing):
    filing.pages[0].raw_text = filing.pages[0].raw_text.replace('Header', 'Hxader')
    filing.pages[0].text = filing.pages[0].text.replace('Header', 'Hxader')
    for chunk in filing.chunks:
        if chunk.page == 1:
            chunk.text = chunk.text.replace('Header', 'Hxader')
    validate_document(filing)
    report = module.audit(filing, filing.source_path)
    assert report['status'] == 'mismatch'
    assert report['differences']['changed_pages'] == [1]
    assert report['differences']['changed_chunk_ids']


def test_cli_persists_mismatch_and_protects_files(filing, tmp_path, monkeypatch):
    pdf = filing.source_path
    filing.pages = filing.pages[:1]
    filing.chunks = [c for c in filing.chunks if c.page == 1]
    document = tmp_path / 'doc.json'
    document.write_text(filing.model_dump_json(), encoding='utf-8')
    output = tmp_path / 'audit.json'
    def invoke(path):
        monkeypatch.setattr(sys, 'argv', ['audit', str(document), pdf, '--output', str(path)])
        module.main()
    with pytest.raises(SystemExit) as error:
        invoke(output)
    assert error.value.code == 1
    assert json.loads(output.read_text(encoding='utf-8'))['status'] == 'mismatch'
    before = document.read_bytes()
    with pytest.raises(SystemExit):
        invoke(document)
    assert document.read_bytes() == before
