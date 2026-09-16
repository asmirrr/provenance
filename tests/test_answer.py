"""Synthetic contract tests; citation integrity is not financial claim verification."""
from copy import deepcopy
import json
import sys

import pytest

from src import answer, query
from src.ingestion import pipeline


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    pdf = tmp_path / 'synthetic.pdf'
    pdf.write_bytes(b'synthetic fixture')
    monkeypatch.setattr(pipeline, 'extract_pages', lambda _: [
        {'page': 1, 'text': 'Cash balance is 10.\n' + 'Filler. ' * 20 + '\nDebt balance is 20.'},
        {'page': 2, 'text': 'Unretrieved information.'}])
    doc = pipeline.ingest(pdf, pipeline.FilingMetadata(
        company='Fixture', ticker='TEST', fiscal_year=2024, filing_date='2024-11-01',
        source_url='https://example.org/pdf', sec_url='https://example.org/filing'), max_chars=100)
    run = query.query(doc, 'Cash', top_k=1)
    span = run['selection']['bundle']['spans'][0]
    payload = {'schema_version': 1, 'query_sha256': answer.query_digest(run), 'status': 'answered',
               'claims': [{'text': 'Cash balance is 10.', 'citations': [
                   {k: span[k] for k in ('page', 'raw_start', 'raw_end')} | {'quote': span['text']}]}]}
    return doc, run, payload


def test_valid_and_semantic_boundary(inputs):
    doc, run, payload = inputs
    payload['claims'][0]['text'] = 'Intentionally false claim for integrity-only test.'
    report = answer.validate_answer(doc, run, answer.Answer.model_validate(payload))
    assert report['citation_integrity'] == 'passed'
    assert report['checked_citations'] == 1
    assert report['claim_support'] == 'not_checked'
    assert answer.query_digest(json.loads(json.dumps(run, indent=4))) == answer.query_digest(run)


@pytest.mark.parametrize('change', ['quote', 'page', 'offset', 'undelivered', 'duplicate'])
def test_invalid_citations(inputs, change):
    doc, run, payload = inputs
    citation = payload['claims'][0]['citations'][0]
    if change == 'quote':
        citation['quote'] += ' invented'
    elif change == 'page':
        citation['page'] = 2
    elif change == 'offset':
        citation['raw_end'] = citation['raw_start']
    elif change == 'undelivered':
        chunk = doc.chunks[-1]
        citation.update(page=chunk.page, raw_start=chunk.raw_start, raw_end=chunk.raw_end, quote=chunk.text)
    else:
        payload['claims'][0]['citations'].append(deepcopy(citation))
    with pytest.raises(ValueError):
        answer.validate_answer(doc, run, answer.Answer.model_validate(payload))


@pytest.mark.parametrize('change', ['hash', 'corpus', 'question', 'span', 'omissions', 'rank', 'chunk'])
def test_query_tampering(inputs, change):
    doc, run, payload = inputs
    if change == 'hash':
        payload['query_sha256'] = '0' * 64
    elif change == 'corpus':
        run['retrieval']['document_model_sha256'] = '0' * 64
    elif change == 'question':
        run['question'] = 'Different question'
    elif change == 'span':
        run['selection']['bundle']['spans'][0]['text'] = 'Altered'
    elif change == 'omissions':
        run['selection']['omitted_chunks'].append({'chunk_id': 'invented'})
    elif change == 'rank':
        run['retrieval']['results'][0]['rank'] = 2
    else:
        run['retrieval']['results'][0]['chunk']['text'] = 'Altered'
    if change != 'hash':
        payload['query_sha256'] = answer.query_digest(run)
    with pytest.raises(ValueError):
        answer.validate_answer(doc, run, answer.Answer.model_validate(payload))


@pytest.mark.parametrize('change', ['empty', 'blank', 'uncited', 'extra', 'string_page', 'reason'])
def test_schema_rejects_invalid_answers(inputs, change):
    _, _, payload = inputs
    if change == 'empty':
        payload['claims'] = []
    elif change == 'blank':
        payload['claims'][0]['text'] = ' \n '
    elif change == 'uncited':
        payload['claims'][0]['citations'] = []
    elif change == 'extra':
        payload['claims'][0]['verified'] = True
    elif change == 'string_page':
        payload['claims'][0]['citations'][0]['page'] = '1'
    else:
        payload['abstention_reason'] = 'Cannot answer'
    with pytest.raises(ValueError):
        answer.Answer.model_validate(payload)


def test_abstention_and_empty_context(inputs):
    doc, _, _ = inputs
    run = query.query(doc, 'absentword')
    payload = {'schema_version': 1, 'query_sha256': answer.query_digest(run), 'status': 'abstained',
               'claims': [], 'abstention_reason': 'No delivered evidence.'}
    result = answer.validate_answer(doc, run, answer.Answer.model_validate(payload))
    assert result['checked_citations'] == 0 and result['abstention_correctness'] == 'not_checked'
    payload['abstention_reason'] = None
    with pytest.raises(ValueError):
        answer.Answer.model_validate(payload)


def test_page_context_can_cite_unretrieved_text_on_delivered_page(inputs):
    doc, _, _ = inputs
    run = query.query(doc, 'Cash', top_k=1, context='page')
    page = doc.pages[0]
    quote = 'Debt balance is 20.'
    start = page.raw_text.index(quote)
    payload = {'schema_version': 1, 'query_sha256': answer.query_digest(run), 'status': 'answered',
               'claims': [{'text': quote, 'citations': [{'page': 1, 'raw_start': start,
                          'raw_end': start + len(quote), 'quote': quote}]}]}
    assert answer.validate_answer(doc, run, answer.Answer.model_validate(payload))['checked_citations'] == 1


def test_cli_and_input_protection(inputs, tmp_path, monkeypatch):
    doc, run, payload = inputs
    paths = [tmp_path / name for name in ('doc.json', 'query.json', 'answer.json')]
    for path, data in zip(paths, [doc.model_dump(mode='json'), run, payload]):
        path.write_text(json.dumps(data), encoding='utf-8')
    output = tmp_path / 'report.json'
    def invoke(out):
        monkeypatch.setattr(sys, 'argv', ['answer', *map(str, paths), '--output', str(out),
                                        '--source-pdf', doc.source_path])
        answer.main()
    invoke(output)
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['source_pdf_verification']['status'] == 'matched'
    for path in paths:
        before = path.read_bytes()
        with pytest.raises(SystemExit):
            invoke(path)
        assert path.read_bytes() == before
    payload['claims'][0]['citations'][0]['quote'] = 'invented'
    paths[-1].write_text(json.dumps(payload), encoding='utf-8')
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        invoke(output)
    assert output.read_bytes() == before
