"""Exercise aborted local static requests, then verify the listening socket survives."""
import concurrent.futures
import json
import socket
from urllib.request import urlopen


def abandon(_):
    with socket.create_connection(('127.0.0.1',8026),timeout=5) as connection:
        connection.sendall(b'GET /app.js HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n')
        assert connection.recv(512)


with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(abandon,range(120)))
with urlopen('http://127.0.0.1:8026/health',timeout=5) as response:
    assert response.status==200 and json.load(response)['ready']
with urlopen('http://127.0.0.1:8026/',timeout=5) as response:
    assert response.status==200 and b'app.js' in response.read()
print(json.dumps(dict(status='PASS',aborted_static_requests=120,health=200,frontend=200)))
