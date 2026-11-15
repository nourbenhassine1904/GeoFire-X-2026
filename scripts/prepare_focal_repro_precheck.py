"""Build a small diagnostic companion; never modify the Focal campaign bundle."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/focal_repro_precheck_v1'


def sha(data): return hashlib.sha256(data).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def extract(source, name):
    node = next(n for n in ast.parse(source).body if getattr(n, 'name', None) == name)
    return ast.get_source_segment(source, node)


def no_training(source):
    """Executable source allowlist: no hidden training entry point in payload."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {'step', 'backward', 'optim', 'load', 'load_state_dict'}:
            # np.load is needed for metadata NPY; other load APIs are excluded.
            if node.attr == 'load' and isinstance(node.value, ast.Name) and node.value.id in {'np', 'json'}:
                continue
            raise ValueError('Training/checkpoint API forbidden in diagnostic payload: ' + node.attr)
    return tree


def build(out=OUT):
    out = Path(out).resolve()
    if not out.is_relative_to(ROOT / 'reports'): raise ValueError('Output must be a new reports directory')
    if out.exists(): raise FileExistsError(out)
    focal = ROOT / 'reports/focal_preparation_v1'
    config_bytes = (focal / 'focal_config.json').read_bytes(); c = json.loads(config_bytes)
    prior_build = json.loads((focal / 'build.json').read_text())
    focal_bytes = (focal / 'focal-code.zip').read_bytes()
    if sha(focal_bytes) != prior_build['code_sha256'] or sha(config_bytes) != prior_build['config_sha256']:
        raise ValueError('Previous frozen Focal preparation differs')
    archive = ROOT / 'reports/unet_confirmation_20261112_v1/inputs/GeoFireX_UNet_ABLATION_3_MODELS.zip'
    archive_hash = hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): archive_hash.update(block)
    if archive_hash.hexdigest() != c['historical_archive_sha256']: raise ValueError('Historical archive differs')
    historical_bundle = ROOT / 'reports/loss_ablation_20261112_v1/loss-ablation-code.zip'
    with zipfile.ZipFile(historical_bundle) as z:
        historical_hash_source = extract(z.read('geofirex/loss_ablation.py').decode(), 'model_hash')
        historical_modules = {n:z.read('geofirex/' + n).decode() for n in ('unet.py', 'benchmark_models.py')}
    probe_source = (ROOT / 'scripts/focal_repro_precheck.py').read_text(encoding='utf-8')
    if ast.dump(ast.parse(historical_hash_source), include_attributes=False) != ast.dump(
            ast.parse(extract(probe_source, 'model_hash')), include_attributes=False):
        raise ValueError('Diagnostic hash algorithm differs from historical')
    variants = []
    with zipfile.ZipFile(archive) as z:
        for weight in (100, 30, 10):
            prefix = f'B2_posweight_{weight}/'
            cb, sb = z.read(prefix+'config.json'), z.read(prefix+'selection32.json')
            rc, rs = json.loads(cb), json.loads(sb)
            if rc['initial_state_sha256'] != c['initial_state_sha256'] or rs['initial_state_sha256'] != c['initial_state_sha256']:
                raise ValueError('Expected historical hash provenance differs')
            variants.append(dict(weight=weight, config_member=prefix+'config.json', config_sha256=sha(cb),
                                 selection_member=prefix+'selection32.json', selection_sha256=sha(sb),
                                 initial_sha256=rc['initial_state_sha256'], torch=rc['torch_version'],
                                 numpy=rc['numpy_version'], device=rc['device'],
                                 trained_checkpoint_sha256=rs['checkpoint_sha256']))
    sources = {}; files = {}
    with zipfile.ZipFile(io.BytesIO(focal_bytes)) as z:
        for name in ('__init__.py', 'unet.py', 'benchmark_models.py'):
            source = z.read('geofirex/' + name).decode('utf-8')
            if name in historical_modules and source != historical_modules[name]:
                raise ValueError('Focal/historical model source differs')
            no_training(source); files['geofirex/' + name] = source
        sampler_source = extract(z.read('geofirex/fast_compact.py').decode(), 'LocalitySampler')
    files['metadata_sampler.py'] = 'import numpy as np\nfrom torch.utils.data import Sampler\n\n' + sampler_source + '\n'
    files['focal_repro_precheck.py'] = probe_source
    with io.BytesIO() as buffer:
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
            for name, source in files.items():
                no_training(source); compile(source, name, 'exec')
                sources[name] = sha(source.encode('utf-8')); z.writestr(name, source)
            z.writestr('focal_config.json', config_bytes)
        payload = buffer.getvalue()
    provenance = dict(expected_initial_sha256=c['initial_state_sha256'],
        historical_archive_sha256=archive_hash.hexdigest(), variants=variants,
        historical_preparation_bundle_sha256=sha(historical_bundle.read_bytes()),
        historical_hash_function=historical_hash_source,
        model_sources_identical_between_saved_preparations=True,
        historical_python_version='NOT_RECORDED', historical_platform_build='NOT_RECORDED',
        historical_initial_tensor_snapshot='NOT_SAVED',
        direct_baseline_gpu_name='NOT_RECORDED; config says CUDA',
        related_propagation_environment=dict(torch='2.11.0+cu130', numpy='2.1.3', cuda='13.0', gpu='Tesla T4'),
        provenance_limit='Recorded config/selection hashes agree and prior audit checked checkpoint config; saved preparation source is not a complete attestation of the executing runtime.')
    cells = []
    def md(s): cells.append(dict(cell_type='markdown', metadata={}, source=s))
    def code(s):
        no_training(s); compile(s, 'repro-precheck-cell', 'exec')
        cells.append(dict(cell_type='code', metadata={}, source=s, execution_count=None, outputs=[]))
    md('''# GeoFire-X - INITIALIZATION PRECHECK ONLY
Thin diagnostic companion to the existing frozen Focal notebook. NO campaign.
Use a fresh Kaggle session with the same eleven EXISTING compact inputs, Internet OFF.
No package installation, training, model forward, checkpoint load or authorization.
Cells only inspect environment, initialize small random CPU models, optionally copy
one model to the GPU and back for the historical hash boundary, and read existing
compressed-byte hashes / small coordinate indices. No raster/NPZ pixels are decoded.
PASS concerns technical prerequisites only and never starts or authorizes Focal.
''')
    code('''RUN_TRAINING=False
AUTHORIZATION_FILE=None
VERIFY_ALL_BYTES=True
CHECK_CUDA_TRANSFER=True  # ~0.5 MB of parameters, NO forward/optimization.
import base64, hashlib, io, json, sys, tempfile, zipfile, uuid
from pathlib import Path
payload=base64.b64decode(''' + repr(base64.b64encode(payload).decode()) + ''')
assert hashlib.sha256(payload).hexdigest()==''' + repr(sha(payload)) + '''
workspace=Path(tempfile.mkdtemp(prefix="geofirex-repro-code-",dir="/tmp"))
with zipfile.ZipFile(io.BytesIO(payload)) as z: z.extractall(workspace)
sys.path.insert(0,str(workspace))
config_path=workspace/"focal_config.json"
assert hashlib.sha256(config_path.read_bytes()).hexdigest()==''' + repr(sha(config_bytes)) + '''
config=json.loads(config_path.read_text())
from focal_repro_precheck import environment, initialization, verify_corpus, decision
from metadata_sampler import LocalitySampler
OUT=Path("/kaggle/working")/("geofirex-repro-precheck-"+uuid.uuid4().hex)
OUT.mkdir(exist_ok=False)
''')
    code('''# Environment only; mismatches are reported, NEVER repaired automatically.
env=environment()
print(json.dumps(env,indent=2))
''')
    code('''# Exact historical seeds/factory/hash. NO forward and no trained weights.
try:
    initial=initialization(config,env,cuda_transfer=CHECK_CUDA_TRANSFER)
except Exception as exc:
    initial={"status":"BLOCKED","causes":[type(exc).__name__+": "+str(exc)]}
print(json.dumps({k:v for k,v in initial.items() if k not in ("tensors","initializer_trace")},indent=2))
''')
    code('''# Verify current read-only inputs IN PLACE; never import/publish a dataset.
try:
    maps=list(Path("/kaggle/input").rglob("mount-map.json"))
    corpus=verify_corpus(maps,config,LocalitySampler,all_bytes=VERIFY_ALL_BYTES)
except Exception as exc:
    corpus={"status":"BLOCKED","causes":[type(exc).__name__+": "+str(exc)]}
print(json.dumps(corpus,indent=2))
''')
    code('''assert RUN_TRAINING is False and AUTHORIZATION_FILE is None
report=decision(config,env,initial,corpus)
report["frozen_config_sha256"]=hashlib.sha256(config_path.read_bytes()).hexdigest()
path=OUT/"precheck_report.json"
with path.open("x",encoding="utf-8") as stream:
    json.dump(report,stream,indent=2,allow_nan=False)
with (OUT/"precheck_report.sha256").open("x") as stream:
    stream.write(hashlib.sha256(path.read_bytes()).hexdigest()+"  "+path.name+"\\n")
print(report["status"],report["causes"])
print("Save these diagnostic outputs:",OUT)
print("Training remains disabled; no authorization was generated.")
''')
    md('''Stop here. Download/save the JSON and SHA256 diagnostic outputs.
Do not run the Focal notebook training cell or change any expected hash.
BLOCKED with matching hashes but different versions still requires environment review.
PASS does not certify bitwise training reproducibility or permit an optimizer update.
''')
    notebook = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(name='python3',
                    display_name='Python 3', language='python')),
                    cells=[dict(id=f'repro-{i}', **cell) for i, cell in enumerate(cells)])
    out.mkdir(parents=True, exist_ok=False)
    with (out/'precheck-code.zip').open('xb') as f: f.write(payload)
    write(out/'GEOFIREX_FOCAL_PRECHECK_ONLY.ipynb', notebook)
    write(out/'historical_provenance.json', provenance)
    write(out/'build.json', dict(code_sha256=sha(payload), frozen_config_sha256=sha(config_bytes),
          existing_focal_notebook_sha256=sha((focal/'GEOFIREX_FOCAL_SINGLE_DISABLED.ipynb').read_bytes()),
          sources=sources, optimizer_present=False, training_enabled=False, submitted=False,
          corpus_republished=False))
    print(out/'GEOFIREX_FOCAL_PRECHECK_ONLY.ipynb')


if __name__ == '__main__': build()
