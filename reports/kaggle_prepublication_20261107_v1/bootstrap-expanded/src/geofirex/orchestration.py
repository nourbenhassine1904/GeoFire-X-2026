"""Append-only local orchestration of bounded private Kaggle CPU jobs.

The CLI is isolated from the ML environment. No credentials enter job sources.
Large acquisitions and GPU campaigns remain separate, explicitly approved work.
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import io
import json
import re
import subprocess
import time
import uuid
import zipfile
from pathlib import Path

from geofirex.cloud_corpus import check_seal, seal
from geofirex.corpus import json_new, sha256_file


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def redact(value):
    value = re.sub(r"(https?://[^\s?\"']+)\?[^\s\"']+", r"\1?[QUERY_REDACTED]", value)
    value = re.sub(r"(?i)(Bearer\s+)[^\s\"']+", r"\1[REDACTED]", value)
    value = re.sub(r"(?i)((?:access_token|refresh_token|authorization|api_key|KAGGLE_KEY)\s*[=:]\s*)[^\s,]+",
                   r"\1[REDACTED]", value)
    return re.sub(r"(?i)([?&](?:code|token|key)=)[^&\s]+", r"\1[REDACTED]", value)


def event(job, kind, **fields):
    path = Path(job)/"events"/(dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f")+"-"+uuid.uuid4().hex[:8]+".json")
    json_new(path, {"time": timestamp(), "kind": kind, **fields})


def code_bundle(root):
    """Only executable project sources and the pinned official public index."""
    root = Path(root)
    files = sorted((root/"src/geofirex").glob("*.py"))
    files += [root/("scripts/"+name) for name in ("kaggle_publication_probe.py", "kaggle_cloud.py",
              "kaggle_acquire_publish.py", "spark_index_demo.py", "install_spark_bounded.py", "measure_gpu_throughput.py")]
    files += [
              root/"data/manifests/real/corpus_zip_index_v1.json"]
    output = io.BytesIO()
    hashes = {p.relative_to(root).as_posix(): sha256_file(p) for p in files}
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, path.relative_to(root).as_posix())
        z.writestr("code_manifest.json", json.dumps(hashes, sort_keys=True))
    data = output.getvalue()
    if len(data) > 2*1024**2:
        raise ValueError("Code-only job bundle exceeds 2 MiB")
    return data


DRIVER = r'''
import json, os, sys, traceback
from pathlib import Path
from geofirex.cloud_corpus import check_seal
from geofirex.corpus import json_new
from geofirex.kaggle_persistence import (session_id, verify_fresh_session,
    validate_bootstrap, http_status)
output = Path('/kaggle/working/geofirex-result')
output.mkdir(exist_ok=False)
request = json.loads(Path('job-request.json').read_text())
try:
    if request['kind'] in {'session-b', 'verify-batch'}:
        proof = json.loads(Path(request.get('proof_filename', 'manual-adoption.json')).read_text())
        certificate = verify_fresh_session(proof, session=session_id())
        index = json.loads(Path('data/manifests/real/corpus_zip_index_v1.json').read_text())
        validate_bootstrap(certificate, index)
        from geofirex.cloud_corpus import fingerprint
        if (certificate['index_identity'] != proof.get('index_identity', fingerprint(index)) or
                certificate['verification'] != proof.get('verification', proof.get('readback'))):
            raise ValueError('Original proof / official corpus verification differs')
        json_new(output/'bootstrap.json', certificate)
        status = 'PERSISTENCE_VERIFIED_NEW_SESSION'
    elif request['kind'] == 'throughput':
        import torch
        from geofirex.cloud_corpus import prepare_mounted, seal
        from geofirex.kaggle_persistence import KaggleBackend
        from geofirex.runner import load_config
        from scripts.measure_gpu_throughput import measure
        if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
            raise RuntimeError('Requested Tesla T4 unavailable')
        index_path = Path('data/manifests/real/corpus_zip_index_v1.json').resolve()
        index = json.loads(index_path.read_text())
        roots = []
        for certificate in request['certificates']:
            validate_bootstrap(certificate, index)
            roots.append(str(KaggleBackend().mount(certificate['versioned_handle'])))
        prepared = Path('/kaggle/working/geofirex-prepared')
        completeness = prepare_mounted(index, roots, prepared)
        if (completeness['by_split']['train']['events'] < 30
                or completeness['by_split']['val']['events'] < 20):
            raise ValueError('Substantial complete train/validation pilot required')
        results = []
        for model in ['B1', 'B2', 'B3']:
            config = {'model': model, 'device': 'cuda', 'amp': True, 'qualification': 'pilot',
                'data_roots': roots, 'manifest': str(prepared/'manifest.csv'),
                'split': str(prepared/'split.csv'), 'index': str(index_path), 'workers': 0,
                'samples_per_epoch': 4096, 'positive_tile_fraction': 0.5, 'context': 3,
                'batch_size': 8, 'patch_size': 64, 'cpu_threads': 2}
            if model == 'B1':
                config['model_kwargs'] = {'architecture': 'spatial_cnn_v2', 'width': 16}
            path = Path('throughput-'+model+'.json')
            json_new(path, config)
            results.append({'model': model, **measure(load_config(path), steps=100, warmup=10)})
            json_new(output/('throughput-'+model+'.json'), results[-1])
        json_new(output/'throughput.json', seal({'status': 'THROUGHPUT_MEASURED',
            'results': results, 'corpus': completeness, 'quality_metrics': 'not measured; models discarded'}))
        status = 'THROUGHPUT_MEASURED'
    elif request['kind'] == 'bigdata':
        import base64
        from scripts.install_spark_bounded import install_spark
        from scripts.spark_index_demo import run
        import importlib.util, importlib.metadata
        if importlib.util.find_spec('pyspark'):
            tooling = {'artifact_bytes': 0, 'source': 'preinstalled Kaggle image',
                'pyspark_version': importlib.metadata.version('pyspark')}
        elif request['dependency_budget_bytes'] > 0:
            tooling = install_spark('/kaggle/working/spark-dependencies', request['dependency_budget_bytes'])
        else:
            raise RuntimeError('PySpark absent and no remaining dependency download authorization')
        json_new(output/'tooling-downloads.json', tooling)
        qc = Path('observations.parquet')
        qc.write_bytes(base64.b64decode(request['metadata_parquet_b64']))
        from geofirex.corpus import sha256_file
        if sha256_file(qc) != request['metadata_parquet_sha256']:
            raise ValueError('Real raster QC metadata changed')
        report = run('data/manifests/real/corpus_zip_index_v1.json', qc, output/'bigdata')
        json_new(output/'spark-report.json', report)
        status = 'SPARK_EXECUTED'
    elif request['kind'] == 'recover-batch':
        from scripts.kaggle_acquire_publish import recover_published
        index = json.loads(Path('data/manifests/real/corpus_zip_index_v1.json').read_text())
        plan = request['plan']
        receipt = recover_published(index, request['dataset_handle'], output/'publication.json',
            years=sorted({int(e[:4]) for e in plan['events']}), minimum=len(plan['events']),
            maximum=len(plan['events']), session=session_id(), expected_plan=plan)
        if receipt['inventory_identity'] != request['expected_inventory_identity']:
            raise ValueError('Recovered dataset differs from the original attempted publication')
        status = receipt['status']
    elif request['kind'] == 'acquire':
        import shutil
        from geofirex.cloud_corpus import validate_plan, acquire_batch
        from geofirex.kaggle_persistence import publish_package, KaggleBackend, require_owner
        index = json.loads(Path('data/manifests/real/corpus_zip_index_v1.json').read_text())
        plan, certificate = request['plan'], request['bootstrap']
        validate_plan(plan, index)
        validate_bootstrap(certificate, index)
        require_owner(KaggleBackend(), request['dataset_handle'])
        if shutil.disk_usage('/kaggle/working').free < plan['limits']['storage_bytes']:
            raise RuntimeError('Working disk below approved staging allocation; no transfer started')
        payload = Path('/kaggle/working/geofirex-acquisition/payload')
        acquire_batch(plan, index, payload, output/'acquisition.json')
        receipt = publish_package(payload, request['dataset_handle'], output/'publication.json',
            upload_bytes=plan['limits']['upload_bytes'], private_free_bytes=plan['limits']['private_free_bytes'],
            session=session_id(), mode='create', attempts=120, delay=5)
        status = receipt['status']
    elif request['kind'] == 'publication-probe':
        from scripts.kaggle_publication_probe import prepare_probe
        from geofirex.kaggle_persistence import publish_package, KaggleBackend, require_owner
        handle = request['probe_handle']
        require_owner(KaggleBackend(), handle)
        payload = Path('/kaggle/working/geofirex-probe-payload')
        prepare_probe(payload)
        if request.get('expected_inventory_identity'):
            inventory = json.loads((payload/'inventory.json').read_text())
            if inventory['identity'] != request['expected_inventory_identity']:
                raise ValueError('Probe recovery inventory differs from the original upload')
        receipt = publish_package(payload, handle, output/'publication-probe.json',
            upload_bytes=1024**2, private_free_bytes=1024**2, session=session_id(), mode=request.get('mode', 'create'),
            attempts=60, delay=5)
        status = receipt['status']
    else:
        raise ValueError('Unsupported job kind')
    json_new(output/'result.json', {'job_id': request['job_id'], 'kind': request['kind'],
        'status': status, 'success': True})
except Exception as error:
    # Exception messages may include signed URLs. Record type/status only.
    from geofirex.kaggle_persistence import error_evidence
    json_new(output/'result.json', {'job_id': request['job_id'], 'kind': request['kind'],
        'status': 'FAILED', 'success': False, 'error_type': type(error).__name__,
        'http_status': http_status(error), 'stage': request['kind'], 'error': error_evidence(error)})
    print('GeoFire-X failed:', type(error).__name__, 'HTTP', http_status(error))
    sys.exit(1)
'''


def kernel_source(bundle, request, proof_bytes=None):
    # Strings are represented as Python literals; no shell interpolation.
    values = dict(bundle=base64.b64encode(bundle).decode(), digest=hashlib.sha256(bundle).hexdigest(),
                  request=json.dumps(request), driver=DRIVER,
                  proof=base64.b64encode(proof_bytes).decode() if proof_bytes else None)
    return "# GeoFire-X private CPU job; generated, no credentials.\nVALUES = "+repr(values)+r'''
import base64, hashlib, io, json, os, signal, subprocess, sys, time, zipfile
from pathlib import Path, PurePosixPath
request = json.loads(VALUES['request'])
started = time.monotonic()
def deadline_reached(signum, frame):
    raise TimeoutError('GeoFire-X execution deadline reached')
signal.signal(signal.SIGALRM, deadline_reached)
signal.alarm(request['timeout_seconds'])
data = base64.b64decode(VALUES['bundle'])
assert hashlib.sha256(data).hexdigest() == VALUES['digest'], 'Code bundle hash mismatch'
root = Path('/kaggle/working/geofirex-code')
root.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(data)) as archive:
    for entry in archive.infolist():
        name = PurePosixPath(entry.filename)
        if name.is_absolute() or '..' in name.parts or '\\' in entry.filename:
            raise ValueError('Unsafe code archive')
    archive.extractall(root)
for name, digest in json.loads((root/'code_manifest.json').read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
(root/'job-request.json').write_text(VALUES['request'])
if VALUES['proof']:
    proof = base64.b64decode(VALUES['proof'])
    assert hashlib.sha256(proof).hexdigest() == json.loads(VALUES['request'])['proof_sha256']
    (root/json.loads(VALUES['request']).get('proof_filename', 'manual-adoption.json')).write_bytes(proof)
(root/'driver.py').write_text(VALUES['driver'])
# Isolated tooling, no alteration of the image's ML stack.
tooling = '/kaggle/working/geofirex-tooling'
try:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-deps',
        '--target', tooling, 'kagglehub==1.0.3', 'kagglesdk==0.1.38'], check=True,
        timeout=max(1, request['timeout_seconds']-(time.monotonic()-started)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([tooling, str(root/'src'), str(root)]))
    subprocess.run([sys.executable, 'driver.py'], cwd=root, env=env, check=True,
        timeout=max(1, request['timeout_seconds']-(time.monotonic()-started)))
except BaseException as error:
    output = Path('/kaggle/working/geofirex-result')
    output.mkdir(exist_ok=True)
    target = output/'result.json'
    if not target.exists():
        with target.open('x') as result:
            json.dump({'job_id': request['job_id'], 'kind': request['kind'], 'success': False,
                'status': 'FAILED', 'stage': 'bootstrap_or_deadline', 'error_type': type(error).__name__}, result)
    raise
finally:
    signal.alarm(0)
'''


class Orchestrator:
    def __init__(self, root, state=None, cli=None):
        self.root = Path(root).resolve()
        self.state = Path(state) if state else self.root/"artifacts/orchestrator"
        self.cli = Path(cli) if cli else self.state/"kaggle-cli-env/Scripts/kaggle.exe"

    def prepare(self, kind, owner, *, proof=None, resume_receipt=None, plan=None, bootstrap=None,
                dependency_budget_bytes=600*1024**2, certificates=()):
        if kind not in {"session-b", "verify-batch", "publication-probe", "bigdata", "acquire", "recover-batch", "throughput"}:
            raise ValueError("Only bounded CPU certification/probe jobs are authorized here")
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", owner):
            raise ValueError("Invalid Kaggle owner")
        job_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:8]
        label = "probe" if kind == "publication-probe" else kind
        handle = owner+"/geofirex-"+label+"-"+job_id
        request = {"job_id": job_id, "kind": kind, "handle": handle, "created_at": timestamp(),
                   "owner": owner, "timeout_seconds": 900, "accelerator": "CPU", "private": True}
        raw = None
        sources = []
        if kind in {"session-b", "verify-batch"}:
            raw = Path(proof).read_bytes()
            receipt = json.loads(raw)
            check_seal(receipt)
            expected = "MANUAL_ADOPTED_FRESH_SESSION_PENDING" if kind == "session-b" else "REMOTE_VERIFIED_FRESH_SESSION_PENDING"
            if (receipt.get("status") != expected or receipt.get("authenticated_owner") != owner
                    or receipt.get("private_verified") is not True
                    or (kind == "session-b" and (receipt.get("kind") != "manual_adoption" or receipt.get("upload_called") is not False))):
                raise ValueError("Original verified manual-adoption proof required")
            sources = [receipt["handle"]]
            request.update(proof_sha256=hashlib.sha256(raw).hexdigest(),
                           proof_identity=receipt["identity"], inventory_identity=receipt["inventory_identity"],
                           versioned_handle=receipt["versioned_handle"],
                           proof_filename="manual-adoption.json" if kind == "session-b" else "producer-receipt.json")
        elif kind == "throughput":
            from geofirex.cloud_corpus import fingerprint
            index = json.loads((self.root/'data/manifests/real/corpus_zip_index_v1.json').read_text())
            values = [json.loads(Path(p).read_text()) for p in certificates]
            if len(values) < 2:
                raise ValueError("Verified train and validation corpus certificates required")
            for value in values:
                check_seal(value)
                if (value.get('status') != 'PERSISTENCE_VERIFIED_NEW_SESSION' or value.get('private_verified') is not True
                        or not value.get('session') or value['session'] == value.get('producer_session')
                        or value.get('index_identity') != fingerprint(index)
                        or not value.get('versioned_handle', '').startswith(owner+'/')):
                    raise ValueError('Unverified corpus certificate')
            sources = [v['versioned_handle'].split('/versions/')[0] for v in values]
            if len(set(sources)) != len(sources):
                raise ValueError('Duplicate corpus sources')
            request.update(certificates=values, accelerator='NvidiaTeslaT4', authorization_required=True)
        elif kind == "bigdata":
            if not 0 <= dependency_budget_bytes <= 600*1024**2:
                raise ValueError("Dependency download budget outside bounded Spark allowance")
            metadata = self.root/"data/interim/benchmark_metadata_v1/observations.parquet"
            request.update(dependency_budget_bytes=dependency_budget_bytes, timeout_seconds=900,
                           authorization_required=True, metadata_parquet_sha256=sha256_file(metadata),
                           metadata_parquet_b64=base64.b64encode(metadata.read_bytes()).decode())
        elif kind == 'recover-batch':
            from geofirex.cloud_corpus import validate_plan
            index = json.loads((self.root/'data/manifests/real/corpus_zip_index_v1.json').read_text())
            plan = json.loads(Path(plan).read_text())
            validate_plan(plan, index)
            receipt = json.loads(Path(proof).read_text())
            check_seal(receipt)
            if (receipt.get('status') != 'FAILED_FILES_PRESERVED' or receipt.get('upload_called') is not True
                    or receipt.get('authenticated_owner') != owner
                    or receipt.get('files') != plan['observations']
                    or not receipt.get('handle', '').startswith(owner+'/')):
                raise ValueError('Original attempted complete-batch publication receipt required')
            sources = [receipt['handle']]
            request.update(plan=plan, dataset_handle=receipt['handle'],
                expected_inventory_identity=receipt['inventory_identity'],
                original_upload_receipt_identity=receipt['identity'],
                zenodo_transfer_bytes=0, upload_bytes=0)
        elif kind == "acquire":
            from geofirex.cloud_corpus import validate_plan, fingerprint
            index = json.loads((self.root/"data/manifests/real/corpus_zip_index_v1.json").read_text())
            plan = json.loads(Path(plan).read_text())
            validate_plan(plan, index)
            certificate = json.loads(Path(bootstrap).read_text())
            check_seal(certificate)
            if (certificate.get("status") != "PERSISTENCE_VERIFIED_NEW_SESSION"
                    or certificate.get("index_identity") != fingerprint(index)
                    or certificate.get("session") == certificate.get("producer_session")
                    or certificate.get("private_verified") is not True):
                raise ValueError("Completed authentic bootstrap required")
            sources = [certificate["versioned_handle"].split('/versions/')[0]]
            label = "train2018" if all(e.startswith('2018/') for e in plan['events']) else "val2020"
            if label == "val2020" and not all(e.startswith('2020/') for e in plan['events']):
                raise ValueError("Only approved 2018 train and 2020 validation lots supported")
            request.update(plan=plan, bootstrap=certificate, authorization_required=True,
                timeout_seconds=3600, dataset_handle=owner+'/gfx-wsts-'+label+'-'+job_id)
        else:
            request["probe_handle"] = owner+"/geofirex-transport-probe-"+job_id
            request["payload_cap_bytes"] = 1024**2
            if resume_receipt:
                receipt = json.loads(Path(resume_receipt).read_text())
                check_seal(receipt)
                if receipt.get("authenticated_owner") != owner or receipt.get("upload_called") is not True:
                    raise ValueError("An authentic attempted probe publication receipt is required")
                sources = [receipt["handle"]]
                request.update(probe_handle=receipt["handle"], mode="resume",
                               expected_inventory_identity=receipt["inventory_identity"],
                               original_upload_receipt_identity=receipt["identity"])
        bundle = code_bundle(self.root)
        source = kernel_source(bundle, request, raw).encode()
        request.update(code_sha256=hashlib.sha256(source).hexdigest(),
                       bundle_sha256=hashlib.sha256(bundle).hexdigest())
        metadata = {"id": handle, "title": handle.split("/")[1], "code_file": "job.py",
                    "language": "python", "kernel_type": "script", "is_private": True,
                    "enable_gpu": kind == "throughput", "enable_internet": True, "dataset_sources": sources,
                    "competition_sources": [], "kernel_sources": [], "model_sources": []}
        request["metadata_sha256"] = hashlib.sha256(json.dumps(metadata, indent=2).encode()).hexdigest()
        job = self.state/"jobs"/job_id
        job.mkdir(parents=True, exist_ok=False)
        (job/"job.py").write_bytes(source)
        (job/"kernel-metadata.json").write_bytes(json.dumps(metadata, indent=2).encode())
        if raw:
            (job/request['proof_filename']).write_bytes(raw)
        json_new(job/"job.json", seal(request))
        event(job, "PREPARED", qualification="configuration only; not a remote execution")
        return job

    def load(self, job_id):
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
            raise ValueError("Invalid job ID")
        job = self.state/"jobs"/job_id
        request = json.loads((job/"job.json").read_text())
        check_seal(request)
        if sha256_file(job/"job.py") != request["code_sha256"]:
            raise ValueError("Prepared code changed")
        if sha256_file(job/"kernel-metadata.json") != request["metadata_sha256"]:
            raise ValueError("Prepared metadata changed")
        return job, request

    def call(self, job, args, *, timeout=120):
        try:
            result = subprocess.run([str(self.cli), *args], capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=timeout)
            record = {"args": args, "returncode": result.returncode,
                      "stdout": redact(result.stdout), "stderr": redact(result.stderr)}
        except subprocess.TimeoutExpired:
            record = {"args": args, "returncode": None, "error": "TIMEOUT_REMOTE_STATE_UNKNOWN"}
        event(job, "CLI", **record)
        return record

    def submit(self, job_id, *, authorization=None):
        job, request = self.load(job_id)
        if request.get("authorization_required"):
            if not authorization:
                raise PermissionError("Explicit user budget authorization record required")
            permission = json.loads(Path(authorization).read_text())
            check_seal(permission)
            if request["identity"] not in permission.get("job_identities", []):
                raise PermissionError("Authorization does not cover this immutable job")
            event(job, "BUDGET_AUTHORIZATION", identity=permission["identity"])
        previous = [json.loads(p.read_text()) for p in (job/"events").glob("*.json")]
        if any(e["kind"] == "SUBMISSION_ATTEMPT" for e in previous):
            raise ValueError("Submission already attempted; use status/fetch to recover, never blindly resubmit")
        event(job, "SUBMISSION_ATTEMPT")
        args = ["kernels", "push", "-p", str(job), "-t", str(request["timeout_seconds"])]
        if request['accelerator'] == 'NvidiaTeslaT4':
            args += ['--accelerator', 'NvidiaTeslaT4']
        result = self.call(job, args)
        event(job, "SUBMITTED" if result["returncode"] == 0 else "SUBMISSION_UNCONFIRMED")
        return result

    def refresh(self, job_id):
        job, request = self.load(job_id)
        result = self.call(job, ["kernels", "status", request["handle"]])
        text = result.get("stdout", "").lower()
        status = "UNKNOWN"
        if result["returncode"] == 0:
            for label in ("complete", "running", "queued", "error", "cancelacknowledged", "cancelrequested"):
                pattern = label.replace("cancel", "cancel_?", 1) if label.startswith("cancel") else label
                if re.search(r"\b"+pattern+r"\b", text):
                    status = label.upper()
                    break
        event(job, "REMOTE_STATUS", status=status)
        return {"job_id": job_id, "remote_status": status, "detail": result}

    def fetch(self, job_id):
        job, request = self.load(job_id)
        folder = job/"retrievals"/uuid.uuid4().hex
        folder.mkdir(parents=True, exist_ok=False)
        # Explicit evidence files only: never corpus, checkpoint or whole notebook output.
        pattern = r"(^|/)geofirex-result/(result|bootstrap|publication-probe|publication|acquisition|spark-report|tooling-downloads|throughput(?:-B[123])?)\.json$"
        pattern += r"|(^|/)geofirex-result/acquisition\.transfer\.jsonl$"
        if request['kind'] == 'bigdata':
            pattern += r"|(^|/)geofirex-result/bigdata/(index-parquet|annual-parquet|raster-qc-parquet)/.*\.parquet$"
        result = self.call(job, ["kernels", "output", request["handle"], "-p", str(folder),
            "--file-pattern", pattern])
        if result["returncode"] != 0:
            return result
        results = list(folder.rglob("result.json"))
        if len(results) != 1:
            event(job, "EVIDENCE_MISSING", folder=str(folder))
            return {"status": "EVIDENCE_MISSING", "folder": str(folder)}
        result = json.loads(results[0].read_text())
        if (result.get("job_id"), result.get("kind")) != (job_id, request["kind"]):
            raise ValueError("Retrieved result belongs to a different job")
        if result.get("success") is True:
            name = {"session-b": "bootstrap.json", "verify-batch": "bootstrap.json", "publication-probe": "publication-probe.json",
                    "acquire": "publication.json", "recover-batch": "publication.json", "bigdata": "spark-report.json", "throughput": "throughput.json"}[request["kind"]]
            evidence_path = results[0].parent/name
            evidence = json.loads(evidence_path.read_text())
            check_seal(evidence)
            if request["kind"] in {"session-b", "verify-batch"}:
                proof = json.loads((job/request.get('proof_filename', 'manual-adoption.json')).read_text())
                if (evidence.get("status") != "PERSISTENCE_VERIFIED_NEW_SESSION"
                        or evidence.get("producer_receipt_identity") != request["proof_identity"]
                        or evidence.get("inventory_identity") != request["inventory_identity"]
                        or evidence.get("versioned_handle") != request["versioned_handle"]
                        or evidence.get("private_verified") is not True
                        or evidence.get("producer_session") != proof["producer_session"]
                        or not evidence.get("session") or evidence["session"] == proof["producer_session"]
                        or evidence.get("verification") != proof.get("verification", proof.get("readback"))
                        or (proof.get("index_identity") and evidence.get("index_identity") != proof["index_identity"])):
                    raise ValueError("Retrieved bootstrap does not certify the original proof")
            elif request["kind"] == "throughput":
                if (evidence.get('status') != 'THROUGHPUT_MEASURED'
                        or {r['model'] for r in evidence.get('results', [])} != {'B1', 'B2', 'B3'}
                        or any(r.get('device') != 'cuda' or 'T4' not in r.get('gpu', '') for r in evidence['results'])):
                    raise ValueError('Incomplete actual Tesla T4 measurements')
            elif request["kind"] == "bigdata":
                if (evidence.get("status") != "SPARK_EXECUTED" or evidence.get("index_observations") != 13607
                        or evidence.get("index_events") != 607
                        or evidence.get("raster_qc_parquet_sha256") != request["metadata_parquet_sha256"]):
                    raise ValueError("Spark report differs from the prepared real metadata")
                import pyarrow.parquet as pq
                parquet = results[0].parent/'bigdata'
                entries = pq.read_table(parquet/'index-parquet').to_pylist()
                index = json.loads((self.root/'data/manifests/real/corpus_zip_index_v1.json').read_text())
                columns = ['event_id', 'date', 'year', 'path', 'compressed_bytes', 'geotiff_bytes', 'crc32']
                expected = [{k:e[k] for k in columns} for e in index['entries']]
                if sorted(entries, key=lambda r:r['path']) != sorted(expected, key=lambda r:r['path']):
                    raise ValueError('Retrieved Spark Parquet differs from the entire official index')
                annual = pq.read_table(parquet/'annual-parquet').to_pylist()
                if sorted(annual, key=lambda r:r['year']) != evidence['annual']:
                    raise ValueError('Retrieved annual Spark aggregates differ from measured report')
                files = [{'path': p.relative_to(folder).as_posix(), 'bytes': p.stat().st_size,
                          'sha256': sha256_file(p)} for p in sorted(parquet.rglob('*.parquet'))]
                if sum(f['bytes'] for f in files) > 64*1024**2:
                    raise ValueError('Unexpected metadata output volume')
                json_new(folder/'spark-local-verification.json', seal({'scope': 'Actual retrieved Spark Parquet',
                    'index_rows_equal_to_official_source': len(entries), 'annual_aggregates_equal': True,
                    'files': files, 'report_identity': evidence['identity']}))
            elif request['kind'] == 'recover-batch':
                if (evidence.get('status') != 'REMOTE_VERIFIED_FRESH_SESSION_PENDING'
                        or evidence.get('handle') != request['dataset_handle']
                        or evidence.get('inventory_identity') != request['expected_inventory_identity']
                        or evidence.get('upload_called') is not False or evidence.get('private_verified') is not True
                        or evidence.get('kind') != 'remote_recovery'):
                    raise ValueError('Read-only recovery did not verify the original private package')
            elif request["kind"] == "acquire":
                acquisition = json.loads((results[0].parent/'acquisition.json').read_text())
                if (evidence.get("status") != "REMOTE_VERIFIED_FRESH_SESSION_PENDING"
                        or evidence.get("handle") != request["dataset_handle"]
                        or evidence.get("private_verified") is not True
                        or evidence.get("files") != request["plan"]["observations"]
                        or acquisition.get('plan_identity') != request['plan']['identity']
                        or acquisition.get('inventory_identity') != evidence.get('inventory_identity')
                        or acquisition.get('integrity', {}).get('complete') is not True
                        or set(acquisition['integrity']['verified']) != {e['path'] for e in request['plan']['entries']}
                        or acquisition['transfer']['reserved_bytes_including_retries'] > request['plan']['limits']['transfer_bytes']):
                    raise ValueError("Acquisition publication differs from the approved plan")
            elif (evidence.get("status") != "REMOTE_VERIFIED_FRESH_SESSION_PENDING"
                    or evidence.get("handle") != request["probe_handle"]
                    or evidence.get("private_verified") is not True):
                raise ValueError("Publication probe was not verified")
            event(job, "EVIDENCE_VERIFIED", path=str(evidence_path), sha256=sha256_file(evidence_path),
                  status=evidence["status"])
        else:
            event(job, "REMOTE_FAILURE", result=result)
        return {**result, "folder": str(folder)}

    def status(self):
        rows = []
        for path in sorted((self.state/"jobs").glob("*/job.json")):
            try:
                job, request = self.load(path.parent.name)
            except ValueError as error:
                rows.append({"job_id": path.parent.name, "latest_local_event": "LOCAL_INTEGRITY_FAILED",
                             "detail": str(error), "evidence": None})
                continue
            events = [json.loads(p.read_text()) for p in sorted((job/"events").glob("*.json"))]
            remote = [e for e in events if e["kind"] == "REMOTE_STATUS"]
            evidence = [e for e in events if e["kind"] == "EVIDENCE_VERIFIED"]
            rows.append({"job_id": request["job_id"], "kind": request["kind"], "handle": request["handle"],
                         "latest_local_event": events[-1]["kind"],
                         "last_remote_status": remote[-1]["status"] if remote else "NOT_QUERIED",
                         "last_remote_check": remote[-1]["time"] if remote else None,
                         "evidence": evidence[-1] if evidence else None})
        monitors = []
        for started in sorted(self.state.glob('monitor-*/started.json')):
            snapshots = sorted(p for p in started.parent.glob('*.json') if p.name not in {'started.json', 'finished.json'})
            monitors.append({'directory': str(started.parent), **json.loads(started.read_text()),
                'finished': json.loads((started.parent/'finished.json').read_text()) if (started.parent/'finished.json').exists() else None,
                'last_cycle': json.loads(snapshots[-1].read_text())['time'] if snapshots else None})
        chains = []
        for started in sorted(self.state.glob('chain-*/started.json')):
            cycles = sorted(started.parent.glob('cycle-*.json'))
            resumes = sorted(started.parent.glob('resumed-*.json'))
            chains.append({'directory': str(started.parent), **json.loads(started.read_text()),
                'active_process_record': json.loads(resumes[-1].read_text()) if resumes else json.loads(started.read_text()),
                'latest': json.loads((started.parent/'finished.json').read_text()) if (started.parent/'finished.json').exists()
                    else json.loads(cycles[-1].read_text()) if cycles else None})
        demo = self.root/'artifacts/demo-serving/http-verification-v2-validation/verification.json'
        if not demo.exists():
            demo = self.root/'artifacts/demo-serving/http-verification-v1/verification.json'
        comparison = self.root/'artifacts/demo-comparison/real-cpu-v1/comparison.json'
        return {"generated_at": timestamp(), "jobs": rows, "monitors": monitors, 'chains': chains,
                'local_demo_http_verification': json.loads(demo.read_text()) if demo.exists() else None,
                'local_diagnostic_comparison': str(comparison) if comparison.exists() else None,
                "long_gpu": "NOT_AUTHORIZED_NOT_SUBMITTED"}
