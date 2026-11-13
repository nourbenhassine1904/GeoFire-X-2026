"""Focused ASGI requests; no TestClient dependency or live server required."""
import asyncio
import json
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles
from geofirex.mission_results import attach_mission
from geofirex.mission_streaming import save

async def main():
    app=FastAPI();app.mount('/',StaticFiles(directory='frontend'),name='frontend');attach_mission(app,'reports/mission_20261111')
    async def request(path):
        messages=[]
        scope=dict(type='http',asgi=dict(version='3.0'),http_version='1.1',method='GET',scheme='http',path=path,raw_path=path.encode(),query_string=b'',headers=[],client=('127.0.0.1',1234),server=('localhost',80),root_path='')
        async def receive():return dict(type='http.request',body=b'',more_body=False)
        async def send(message):messages.append(message)
        await app(scope,receive,send)
        status=next(m['status'] for m in messages if m['type']=='http.response.start')
        body=b''.join(m.get('body',b'') for m in messages if m['type']=='http.response.body')
        return status,json.loads(body)
    status,events=await request('/mission/events');assert status==200 and events['windows_evaluated']==3287
    status,scene=await request('/mission/map/0');assert status==200 and scene['model']=='B0'
    status,_=await request('/mission/map/100');assert status==404
    status,b=await request('/mission/benchmark');assert status==200 and b['matched_pilot']['b1']['samples_seen']==2575
    status,spark=await request('/mission/big-data');assert status==200 and spark['distributed_cluster'] is False
    status,c=await request('/mission/compact');assert status==200 and c['events']>=16
    save('reports/mission_20261111/api-verification.json',dict(status='PASS',requests=6,method='direct ASGI HTTP request execution',root_frontend_mount_preserved=True))
    print('Six ASGI requests PASS, real responses and 404 checked')

if __name__=='__main__':asyncio.run(main())
