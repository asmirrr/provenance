"""Saved-run evaluation uses synthetic responses and makes no live API calls."""
from copy import deepcopy
import json
import sys

import pytest

from src import generation, generation_evaluation as evaluation
from src.benchmark import Benchmark
from test_answer import inputs
from test_generation import response


def benchmark(doc, run, *, answerable=True):
    chunk=doc.chunks[0]
    return Benchmark.model_validate({'version':'0.1.0','status':'draft_pending_human_review',
        'source_sha256':doc.source_sha256,'review_method':'Synthetic fixture',
        'items':[{'id':'fixture','question':run['question'],'category':'factual','answerable':answerable,
                  'expected_answer':'Fixture' if answerable else None,
                  'expected_claims':['Fixture'] if answerable else [],'difficulty':'easy','notes':'Synthetic',
                  'evidence':[{'page':chunk.page,'quote':chunk.text}] if answerable else []}]})


def artifact(doc, run, payload, monkeypatch):
    monkeypatch.setattr(generation,'send_request',lambda *_:response(payload))
    return generation.generate(doc,run,model='synthetic',api_key='synthetic')


def test_revalidates_raw_instead_of_saved_success(inputs,monkeypatch):
    doc,run,payload=inputs
    saved=artifact(doc,run,payload,monkeypatch)
    saved['raw_response']['content'][0]['text']='not JSON'
    report=evaluation.evaluate(doc,benchmark(doc,run),[('fixture',run,saved)],allow_draft=True)
    assert report['outcomes']=={'invalid_response':1}
    assert report['response_behavior']['answerable']=={'selected':1,'answered':0,'abstained':0}
    assert report['claim_accuracy']=='not_scored'


def test_unsupported_answer_is_behavior_not_accuracy(inputs,monkeypatch):
    doc,run,payload=inputs
    saved=artifact(doc,run,payload,monkeypatch)
    saved['validation']={'invented':'do not trust'}
    report=evaluation.evaluate(doc,benchmark(doc,run,answerable=False),[('fixture',run,saved)],allow_draft=True)
    assert report['response_behavior']['unsupported']['answered']==1
    assert report['items'][0]['validation']['claim_support']=='not_checked'
    assert report['items'][0]['human_review']=='pending'


@pytest.mark.parametrize('case',['draft','duplicate','unknown','question','binding','request','missing'])
def test_invalid_inputs(inputs,monkeypatch,case):
    doc,run,payload=inputs
    bench=benchmark(doc,run)
    saved=artifact(doc,run,payload,monkeypatch)
    rows=[('fixture',run,saved)]
    if case=='duplicate': rows*=2
    elif case=='unknown': rows=[('unknown',run,saved)]
    elif case=='question': bench.items[0].question='Different'
    elif case=='binding': saved['query_sha256']='0'*64
    elif case=='request': saved['request']['system']='Changed prompt'
    elif case=='missing': saved['raw_response']=None
    with pytest.raises(ValueError):
        evaluation.evaluate(doc,bench,rows,allow_draft=case!='draft')


@pytest.mark.parametrize('status',['dry_run','provider_error'])
def test_nonanswers_not_counted_as_abstentions(inputs,status):
    doc,run,_=inputs
    saved=generation.generate(doc,run,model='synthetic',dry_run=True)
    saved['status']=status
    report=evaluation.evaluate(doc,benchmark(doc,run),[('fixture',run,saved)],allow_draft=True)
    assert report['outcomes']=={status:1}
    assert report['response_behavior']['answerable']['abstained']==0


def test_cli_relative_paths_and_no_overwrite(inputs,tmp_path,monkeypatch):
    doc,run,_=inputs
    saved=generation.generate(doc,run,model='synthetic',dry_run=True)
    objects={'doc.json':doc.model_dump(mode='json'),'benchmark.json':benchmark(doc,run).model_dump(mode='json'),
             'query.json':run,'generation.json':saved,
             'manifest.json':[{'item_id':'fixture','query':'query.json','generation':'generation.json'}]}
    for name,obj in objects.items():
        (tmp_path/name).write_text(json.dumps(obj),encoding='utf-8')
    output=tmp_path/'report.json'
    monkeypatch.setattr(sys,'argv',['evaluation',str(tmp_path/'doc.json'),str(tmp_path/'benchmark.json'),
                                   str(tmp_path/'manifest.json'),'--allow-draft','--output',str(output)])
    evaluation.main()
    report=json.loads(output.read_text(encoding='utf-8'))
    assert report['artifacts'][0]['query']['sha256']
    assert report['outcomes']=={'dry_run':1}
    before=output.read_bytes()
    with pytest.raises(SystemExit):evaluation.main()
    assert output.read_bytes()==before
