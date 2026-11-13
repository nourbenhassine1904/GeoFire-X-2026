"""Durable conservative reservations for HTTP request bodies, including retries."""
import json
import os
from pathlib import Path
import threading


class BodyBudget:
    def __init__(self, journal, ceiling):
        self.path, self.ceiling = Path(journal), ceiling
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.used = 0
        # Exclusive ownership avoids two upload processes spending the same budget.
        self.lock_path = self.path.with_suffix(self.path.suffix+'.lock')
        self.fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            if self.path.exists():
                for line in self.path.read_text(encoding='utf-8').splitlines():
                    row = json.loads(line)
                    if row['ceiling'] != ceiling or row['previous'] != self.used:
                        raise ValueError('Budget journal mismatch; never reset consumed bytes')
                    self.used += row['reserved']
            if self.used > ceiling:
                raise ValueError('Previously reserved upload bytes exceed ceiling')
        except BaseException:
            self.close()
            raise

    def reserve(self, n):
        with self.lock:
            if n < 0 or self.used+n > self.ceiling:
                raise RuntimeError(f'Upload budget exhausted: {self.used} reserved / {self.ceiling}; next body {n}')
            with self.path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(dict(previous=self.used, reserved=n, ceiling=self.ceiling))+'\n')
                f.flush()
                os.fsync(f.fileno())
            self.used += n

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
            self.lock_path.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def install_requests_guard(self):
        import requests
        from urllib3.util.retry import Retry
        original = requests.sessions.Session.send
        budget = self

        def guarded(session, request, **kwargs):
            # All adapter retries are visible as new SDK sends or disabled.
            for adapter in session.adapters.values():
                adapter.max_retries = Retry(total=0, redirect=0)
            length = request.headers.get('Content-Length')
            if length is None and request.body is not None:
                raise RuntimeError('Unknown HTTP body length; refusing an unbudgeted upload')
            budget.reserve(int(length or 0))
            return original(session, request, **kwargs)
        requests.sessions.Session.send = guarded
        return lambda: setattr(requests.sessions.Session, 'send', original)
