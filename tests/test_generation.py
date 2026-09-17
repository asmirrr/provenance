"""Offline provider fixtures, not real model responses or accuracy measurements."""
import json
import sys
from copy import deepcopy
from urllib import error

import pytest

from src import generation
from test_answer import inputs


def response(payload, stop='end_turn'):
    return {'id': 'synthetic-message', 'model': 'synthetic-model', 'stop_reason': stop,
            'usage': {'input_tokens': 1, 'output_tokens': 1},
            'content': [{'type': 'text', 'text': json.dumps(payload)}]}


def test_request_only_delivered_evidence_and_dry_run(inputs, monkeypatch):
    doc, run, _ = inputs
    monkeypatch.setattr(generation, 'send_request', lambda *_: pytest.fail('Network called'))
    result = generation.generate(doc, run, model='synthetic-model', dry_run=True)
    content = json.loads(result['request']['messages'][0]['content'])
    assert content['evidence'] == run['selection']['bundle']['spans']
    assert doc.pages[-1].text not in json.dumps(content)
    assert 'retrieval' not in content and 'source_path' not in content
    assert result['status'] == 'dry_run' and result['raw_response'] is None


def test_success_preserves_raw_and_checks_citations(inputs, monkeypatch):
    doc, run, payload = inputs
    raw = response(payload)
    calls = []
    def send(body, key):
        calls.append(body)
        assert key == 'synthetic-key'
        return raw
    monkeypatch.setattr(generation, 'send_request', send)
    result = generation.generate(doc, run, model='synthetic-model', api_key='synthetic-key')
    assert len(calls) == 1 and result['raw_response'] == raw
    assert result['status'] == 'citation_validated'
    assert result['validation']['claim_support'] == 'not_checked'
    assert 'synthetic-key' not in json.dumps(result)


@pytest.mark.parametrize('case', ['truncated', 'refusal', 'tool', 'empty', 'json', 'citation', 'binding'])
def test_invalid_outputs_preserved_without_retry(inputs, monkeypatch, case):
    doc, run, payload = inputs
    raw = response(payload)
    if case == 'truncated':
        raw['stop_reason'] = 'max_tokens'
    elif case == 'refusal':
        raw['stop_reason'] = 'refusal'
    elif case == 'tool':
        raw['content'] = [{'type': 'tool_use'}]
    elif case == 'empty':
        raw['content'] = []
    elif case == 'json':
        raw['content'][0]['text'] = '```json\n{}\n```'
    else:
        if case == 'citation':
            payload['claims'][0]['citations'][0]['quote'] = 'invented'
        else:
            payload['query_sha256'] = '0' * 64
        raw = response(payload)
    calls = []
    monkeypatch.setattr(generation, 'send_request', lambda *args: calls.append(args) or raw)
    result = generation.generate(doc, run, model='synthetic-model', api_key='synthetic-key')
    assert len(calls) == 1
    assert result['status'] == 'invalid_response' and result['raw_response'] == raw
    assert result['validation'] is None


def test_preflight_missing_key_and_empty_evidence(inputs, monkeypatch):
    from src.query import query
    doc, run, _ = inputs
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.setattr(generation, 'send_request', lambda *_: pytest.fail('Network called'))
    with pytest.raises(ValueError, match='ANTHROPIC_API_KEY'):
        generation.generate(doc, run, model='synthetic-model')
    empty = query(doc, 'absentword')
    result = generation.generate(doc, empty, model='synthetic-model')
    assert result['status'] == 'local_abstention'
    assert result['validation']['answer']['status'] == 'abstained'
    bad = deepcopy(run)
    bad['selection']['bundle']['spans'][0]['text'] = 'tampered'
    with pytest.raises(ValueError):
        generation.generate(doc, bad, model='synthetic-model', dry_run=True)


def test_http_request_and_sanitized_errors(monkeypatch):
    class Opener:
        def open(self, req, timeout):
            assert req.full_url == generation.ENDPOINT and timeout == 60
            assert req.get_header('X-api-key') == 'synthetic-key'
            assert req.get_header('Anthropic-version') == '2023-06-01'
            assert json.loads(req.data) == {'model': 'synthetic-model'}
            raise error.HTTPError(req.full_url, 401, 'secret error body', {}, None)
    monkeypatch.setattr(generation.request, 'build_opener', lambda *_: Opener())
    with pytest.raises(ValueError, match='HTTP 401') as exc:
        generation.send_request({'model': 'synthetic-model'}, 'synthetic-key')
    assert 'secret' not in str(exc.value) and 'synthetic-key' not in str(exc.value)


def test_cli_failure_artifact_and_overwrite_guard(inputs, tmp_path, monkeypatch):
    doc, run, _ = inputs
    document = tmp_path / 'doc.json'
    query = tmp_path / 'query.json'
    document.write_text(doc.model_dump_json(), encoding='utf-8')
    query.write_text(json.dumps(run), encoding='utf-8')
    output = tmp_path / 'generation.json'
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'synthetic-key')
    monkeypatch.setattr(generation, 'send_request', lambda *_: response({}, 'max_tokens'))
    monkeypatch.setattr(sys, 'argv', ['generation', str(document), str(query), '--model',
                                    'synthetic-model', '--output', str(output)])
    with pytest.raises(SystemExit) as exc:
        generation.main()
    assert exc.value.code == 1
    assert json.loads(output.read_text(encoding='utf-8'))['status'] == 'invalid_response'
    original = output.read_bytes()
    with pytest.raises(SystemExit):
        generation.main()
    assert output.read_bytes() == original
