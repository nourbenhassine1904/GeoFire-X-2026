"""Budgeted ZIP metadata and incremental acquisition; no implicit downloads.

GeoTIFF ZIP-entry bytes are not decoded raster bytes. CRC verifies an entry's
transfer integrity, not dataset authenticity; pin the Zenodo archive MD5 too.
"""
from __future__ import annotations

import binascii
import datetime as dt
import hashlib
import json
import re
import shutil
import time
import urllib.request
import uuid
import zipfile
from pathlib import Path

from geofirex.remote_archive import ARCHIVE_URL, HttpRangeFile, RangeResponseError

ENTRY = re.compile(r"(201[89]|202[01])/(fire_\d+)/(\d{4}-\d{2}-\d{2})\.tif")
ARCHIVE_BYTES = 48_359_369_821
ARCHIVE_MD5 = "dc1a04e63ccc70037b277d585b8fe761"


def json_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_entry(name):
    match = ENTRY.fullmatch(name)
    if not match:
        raise ValueError(f"Unexpected archive entry: {name}")
    dt.date.fromisoformat(match[3])
    return f"{match[1]}/{match[2]}", match[3]


class BudgetRangeFile(HttpRangeFile):
    """Each attempt reserves its full requested bytes before opening a socket.

    Refuses HTTP 200 and reads at most requested+1 bytes, even for a lying server.
    Directory inspection and payload transfer share the same hard session cap.
    """
    def __init__(self, budget_bytes, *, url=ARCHIVE_URL, size=ARCHIVE_BYTES, retries=3):
        super().__init__(url, size, timeout=30, retries=retries)
        self.budget_bytes = int(budget_bytes)
        self.bytes_reserved = 0
        self.requests = 0
        self.ledger_path = None

    def enable_ledger(self, path, plan_identity):
        self.ledger_path = Path(path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open('x', encoding='utf-8') as output:
            output.write(json.dumps({'plan_identity': plan_identity, 'budget_bytes': self.budget_bytes})+'\n')

    def _record_reservation(self, start, length):
        if self.ledger_path is not None:
            import os
            with self.ledger_path.open('a', encoding='utf-8') as output:
                output.write(json.dumps({'reserved_bytes': self.bytes_reserved,
                    'requests': self.requests, 'start': start, 'length': length})+'\n')
                output.flush()
                os.fsync(output.fileno())

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.pos + n, self.size)
        length = end - self.pos
        if length <= 0:
            return b""
        if length > 4 * 1024 * 1024:
            raise RangeResponseError("Single read exceeds 4 MiB; stream ZIP members")
        start = self.pos
        for attempt in range(self.retries):
            # Reserve the extra byte used to detect an overlong malformed body.
            reservation = length + 1
            if self.bytes_reserved + reservation > self.budget_bytes:
                raise RuntimeError("Session transfer budget exhausted before request")
            self.bytes_reserved += reservation
            self.requests += 1
            self._record_reservation(start, length)
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end-1}",
                                                               "Accept-Encoding": "identity"})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    if response.status != 206:
                        raise RangeResponseError("Expected HTTP 206; response body refused")
                    expected = f"bytes {start}-{end-1}/{self.size}"
                    if response.headers.get("Content-Range", "").strip() != expected:
                        raise RangeResponseError("Content-Range or archive size changed")
                    data = response.read(length + 1)
                    self.bytes_fetched += len(data)
                    if len(data) != length:
                        raise RangeResponseError("Range body length mismatch")
                    self.pos = end
                    return data
            except (OSError, RangeResponseError):
                if attempt == self.retries - 1:
                    raise
                time.sleep(0.25 * (attempt + 1))
        raise AssertionError("unreachable")


class MemberCachedRangeFile(BudgetRangeFile):
    """Prefetch only the selected ZIP member, in bounded 4 MiB blocks.

    Avoids separate HTTP connections for its name, extra header and each small
    decompressor read. Never prefetches neighbouring/unselected archive data.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.member_start = self.member_stop = None
        self.cache_start, self.cache = 0, b''

    def begin_member(self, info):
        import struct
        self.member_start = self.member_stop = None
        self.cache = b''
        self.seek(info.header_offset)
        header = super().read(30)
        if len(header) != 30 or header[:4] != b'PK\x03\x04':
            raise RangeResponseError('Invalid ZIP local header')
        name_bytes, extra_bytes = struct.unpack_from('<HH', header, 26)
        self.member_start = info.header_offset
        self.member_stop = info.header_offset+30+name_bytes+extra_bytes+info.compress_size
        if self.member_stop > self.size:
            raise RangeResponseError('ZIP member exceeds pinned archive size')
        self.cache_start, self.cache = info.header_offset, header

    def read(self, n=-1):
        if self.member_start is None or not self.member_start <= self.pos < self.member_stop:
            return super().read(n)
        if n < 0 or n > 4*1024**2:
            raise RangeResponseError('Single read exceeds bounded member cache')
        wanted = min(n, self.member_stop-self.pos)
        result = bytearray()
        while len(result) < wanted:
            offset = self.pos-self.cache_start
            if not 0 <= offset < len(self.cache):
                start = self.pos
                self.cache = super().read(min(4*1024**2, self.member_stop-start))
                self.cache_start = self.pos = start
                offset = 0
            count = min(wanted-len(result), len(self.cache)-offset)
            result.extend(self.cache[offset:offset+count])
            self.pos += count
        return bytes(result)


def index_zip(zf, *, archive_size=None, remote=False):
    entries = []
    seen = set()
    for info in zf.infolist():
        if info.is_dir() or not info.filename.endswith(".tif"):
            continue
        event, date = safe_entry(info.filename)
        if info.filename in seen or info.flag_bits & 1:
            raise ValueError("Duplicate or encrypted archive entry")
        seen.add(info.filename)
        entries.append({"path": info.filename, "event_id": event, "date": date,
                        "year": int(event[:4]), "compressed_bytes": info.compress_size,
                        "geotiff_bytes": info.file_size, "crc32": f"{info.CRC:08x}",
                        "compression": info.compress_type, "header_offset": info.header_offset})
    entries.sort(key=lambda e: e["path"])
    return {"schema_version": 1, "source": ARCHIVE_URL if remote else "local_test_archive",
            "archive_bytes": archive_size, "archive_md5": ARCHIVE_MD5 if remote else None,
            "entries": entries, "events": len({e["event_id"] for e in entries}),
            "observations": len(entries), "compressed_tif_bytes": sum(e["compressed_bytes"] for e in entries),
            "extracted_geotiff_bytes": sum(e["geotiff_bytes"] for e in entries)}


def verify_member(path, entry):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != entry["geotiff_bytes"]:
        return False
    crc = 0
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            crc = binascii.crc32(block, crc)
    return f"{crc & 0xffffffff:08x}" == entry["crc32"]


def acquire_members(zf, entries, destination, *, max_files=None, max_compressed_bytes=None):
    """Validate existing bytes via CRC. Never replace a corrupt historical file.

    Completed writes use a temporary file in the same directory, validated before
    atomic rename. Interrupted .part files are retained and retried at file level.
    """
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    written, skipped, reserved = [], [], 0
    for entry in entries:
        safe_entry(entry["path"])
        output = destination / entry["path"]
        if not output.resolve().is_relative_to(destination):
            raise ValueError("Entry escapes destination")
        if output.exists():
            if not verify_member(output, entry):
                raise ValueError(f"Existing file failed CRC; preserved for manual repair: {output}")
            skipped.append(entry["path"])
            continue
        if max_files is not None and len(written) >= max_files:
            break
        if max_compressed_bytes is not None and reserved + entry["compressed_bytes"] > max_compressed_bytes:
            break
        info = zf.getinfo(entry["path"])
        if (info.CRC, info.file_size, info.compress_size) != (
            int(entry["crc32"], 16), entry["geotiff_bytes"], entry["compressed_bytes"]):
            raise ValueError("Remote directory differs from pinned acquisition plan")
        if shutil.disk_usage(destination).free < entry["geotiff_bytes"] + 1024**3:
            raise RuntimeError("Insufficient space for next file plus 1 GiB reserve")
        reserved += entry["compressed_bytes"]
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + f".{uuid.uuid4().hex}.part")
        if hasattr(zf.fp, 'begin_member'):
            zf.fp.begin_member(info)
        with zf.open(info) as source, temporary.open("xb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        if not verify_member(temporary, entry):
            raise ValueError(f"Downloaded member failed integrity check: {temporary}")
        # rename refuses an existing destination on Windows; never use replace.
        if output.exists():
            raise FileExistsError(output)
        temporary.rename(output)
        written.append(entry["path"])
    return {"written": written, "verified_existing": skipped,
            "planned_compressed_bytes_written": reserved}


def inventory_local(index, root):
    """Exhaustive CRC validation of present members, no network or raster decode."""
    verified, missing, corrupt = [], [], []
    for entry in index["entries"]:
        safe_entry(entry["path"])
        file = Path(root) / entry["path"]
        if not file.exists():
            missing.append(entry["path"])
        elif verify_member(file, entry):
            verified.append(entry["path"])
        else:
            corrupt.append(entry["path"])
    return {"expected": len(index["entries"]), "verified": verified, "missing": missing,
            "corrupt": corrupt, "complete": not missing and not corrupt}
