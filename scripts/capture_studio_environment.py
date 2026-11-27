"""Record the tested serving dependency closure; no installation or network."""
from importlib import metadata
import json
import platform
from pathlib import Path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT=Path(__file__).resolve().parents[1]
seeds=['fastapi','uvicorn','torch','numpy','rasterio','matplotlib','scikit-learn',
       'pyyaml','pyarrow','mlflow','pytest']
pending=list(seeds); versions={}
while pending:
    name=canonicalize_name(pending.pop())
    if name in versions:continue
    dist=metadata.distribution(name);versions[name]=dist.version
    for dependency in dist.requires or []:
        r=Requirement(dependency)
        if r.marker is None or r.marker.evaluate({'extra':''}):pending.append(r.name)
out=ROOT/'requirements-studio-cpu.txt'
text='# Tested Windows CPU serving environment. Python '+platform.python_version()+'\n'
text+='# Torch CPU wheels use the additional official index. No training packages are launched.\n'
text+='--extra-index-url https://download.pytorch.org/whl/cpu\n'
text+='\n'.join(name+'=='+version for name,version in sorted(versions.items()))+'\n'
out.write_text(text,encoding='utf-8')
folder=ROOT/'artifacts/studio-v2-deployment';folder.mkdir(parents=True,exist_ok=True)
(folder/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),
    platform=platform.platform(),dependencies=versions),indent=2),encoding='utf-8')
print('Pinned serving dependencies:',len(versions))
