"""Read-only local deployment smoke check; no test-year inference endpoints."""
import json
import time
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError

out=[]
for route in ('/health','/science/v2/deployment-health','/studio/models',
              '/science/v2/comparison','/science/v2/scenes','/science/final-2021','/feu.png'):
    start=time.perf_counter()
    try:
        with urlopen('http://127.0.0.1:8028'+route,timeout=120) as response:
            raw=response.read();row=dict(route=route,status=response.status,bytes=len(raw))
            if 'json' in response.headers.get('Content-Type',''):
                body=json.loads(raw);row['summary']={k:body[k] for k in ('status','events','models','tiles') if k in body}
                if isinstance(row['summary'].get('events'),list):row['summary']['events']=len(row['summary']['events'])
    except Exception as e:row=dict(route=route,error=repr(e))
    row['seconds']=time.perf_counter()-start;out.append(row);print(json.dumps(row,ensure_ascii=True),flush=True)
Path('artifacts/studio-v2-deployment/http-smoke.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
