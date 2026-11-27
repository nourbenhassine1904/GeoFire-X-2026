"""Publication contracts: metadata only, no test-year pixels or Git mutation."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_secret_scan_detects_without_disclosing_value(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('release_audit', ROOT/'scripts/audit_release_publication.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    token = 'gh' + 'p_' + 'x' * 36
    (tmp_path/'candidate.txt').write_text('token=' + token)
    finding = module.scan(['candidate.txt'])
    assert finding == [dict(path='candidate.txt', kind='github_token', line=1)]
    assert token not in json.dumps(finding)


def test_all_six_crisp_phases_have_honest_status_and_acceptance():
    text = (ROOT/'docs/CRISP_DM_CLOTURE_20261127.md').read_text(encoding='utf8')
    for phase in ('Business Understanding', 'Data Understanding', 'Data Preparation',
                  'Modeling', 'Evaluation', 'Deployment'):
        row = next(line for line in text.splitlines() if line.startswith('| '+phase+' |'))
        assert row.count('|') == 8 and 'Acceptation' in row
        assert ('**PARTIAL**' if phase == 'Deployment' else '**PASS**') in row
    assert 'machine vierge non essayee' in text and 'local[2]' in text


def test_frozen_source_attributes_cover_serving_identities():
    from geofirex.studio_v2 import APPLICATION_SOURCES
    attrs = (ROOT/'.gitattributes').read_text()
    pins = json.loads((ROOT/'configs/protocol_v2/baseline_c8.json').read_text())['source_sha256']
    for name in set(pins) | set(APPLICATION_SOURCES):
        assert '/'+name+' -text' in attrs


def test_current_readme_separates_archives_and_final_test():
    text = (ROOT/'README.md').read_text(encoding='utf8')
    current, history = text.split('<details>', 1)
    assert '0,5512253057' in current and '75 540' in current
    assert 'test 2021 est **COMPLETE**' in current
    assert '**archive et desactive en inference directe**' in current
    assert 'machine vierge **n\'a pas ete effectuee**' in current
    assert history.rstrip().endswith('</details>')
