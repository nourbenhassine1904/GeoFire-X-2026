"""Read the official WildfireSpreadTS.zip (Zenodo 8006177, 48.36 GB) remotely via
HTTP Range requests, without downloading the full archive.

The Zenodo CDN advertises `Accept-Ranges: bytes` (verified manually). Python's
`zipfile` only needs an object with `seek()`/`tell()`/`read()` to parse the
central directory and open individual entries — it never requires the whole
file. This lets us: (1) list the real on-disk structure of the dataset for free
(a few hundred KB of central-directory metadata), and (2) extract a handful of
real events on demand, paying only for the bytes of those entries.

Every Range response is validated (HTTP 206, Content-Range matching the
requested bytes, exact byte count) BEFORE its body is ever read. A server that
silently ignores the Range header and answers 200 (the full file) is refused,
not read - that single safeguard is what prevents an unexpected 200 from
triggering a 48.36 GB download.

Stdlib only (urllib + zipfile) — no new dependency.
"""

from __future__ import annotations

import http.client
import logging
import re
import shutil
import time
import urllib.request
import uuid
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

ZENODO_RECORD_ID = "8006177"
ARCHIVE_URL = (
    f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/WildfireSpreadTS.zip/content"
)

_CONTENT_RANGE_RE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)")


class RangeResponseError(RuntimeError):
    """Raised when a Range request did not get back a valid, matching 206
    response. Raising here means the response body was never read."""


def _backoff_sleep(attempt: int, retries: int, seconds: int = 5) -> None:
    """Sleep before the next retry - but never after the last attempt, so
    giving up doesn't carry a pointless final delay."""
    if attempt < retries - 1:
        time.sleep(seconds)


def _validated_range_get(
    url: str, start: int, end: int, timeout: int = 90, retries: int = 5
) -> bytes:
    """GET bytes [start, end] (inclusive) via a Range request, refusing to read
    the body unless the response is HTTP 206 with a Content-Range that matches
    the requested range, and the body length matches exactly."""
    last_err: Exception | None = None
    expected_len = end - start + 1
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except Exception as e:  # noqa: BLE001 - network errors are retried
            last_err = e
            logger.warning(
                "range request failed (attempt %d/%d): %s", attempt + 1, retries, e
            )
            _backoff_sleep(attempt, retries)
            continue

        try:
            status = getattr(resp, "status", None)
            if status is None:
                status = resp.getcode()
            if status != 206:
                raise RangeResponseError(
                    f"Expected HTTP 206 Partial Content for bytes={start}-{end}, got "
                    f"HTTP {status}. Refusing to read the response body - a 200 here "
                    "would mean the server ignored Range and is about to send the "
                    "full archive."
                )
            content_range = resp.headers.get("Content-Range", "")
            match = _CONTENT_RANGE_RE.fullmatch(content_range.strip())
            if not match or int(match.group(1)) != start or int(match.group(2)) != end:
                raise RangeResponseError(
                    f"Content-Range mismatch: requested bytes={start}-{end}, server "
                    f"reported {content_range!r}"
                )
            data = resp.read()
            if len(data) != expected_len:
                raise RangeResponseError(
                    f"Expected {expected_len} bytes for bytes={start}-{end}, received "
                    f"{len(data)}"
                )
            return data
        except (RangeResponseError, http.client.IncompleteRead, OSError) as e:
            last_err = e
            logger.warning(
                "range response validation failed (attempt %d/%d): %s",
                attempt + 1,
                retries,
                e,
            )
            _backoff_sleep(attempt, retries)
        finally:
            resp.close()
    raise RangeResponseError(f"Giving up after {retries} attempts: {last_err}") from last_err


class HttpRangeFile:
    """Minimal seek/read file-like object backed by validated HTTP Range requests.

    The Zenodo CDN has been observed returning transient 502/504 errors under
    load, so every request is retried with backoff rather than failing fast.
    """

    def __init__(self, url: str, total_size: int, timeout: int = 90, retries: int = 5) -> None:
        self.url = url
        self.size = total_size
        self.pos = 0
        self.timeout = timeout
        self.retries = retries
        self.bytes_fetched = 0  # for honest reporting of what was actually downloaded

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def seekable(self) -> bool:
        return True

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            end = self.size - 1
        else:
            end = min(self.pos + n, self.size) - 1
        if self.pos > end:
            return b""
        data = _validated_range_get(self.url, self.pos, end, timeout=self.timeout, retries=self.retries)
        self.pos += len(data)
        self.bytes_fetched += len(data)
        return data


def get_remote_size(url: str = ARCHIVE_URL, timeout: int = 90, retries: int = 5) -> int:
    """Return the exact size of a remote file via a validated 1-byte Range probe."""
    last_err: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("size probe failed (attempt %d/%d): %s", attempt + 1, retries, e)
            _backoff_sleep(attempt, retries)
            continue
        try:
            status = getattr(resp, "status", None) or resp.getcode()
            if status != 206:
                raise RangeResponseError(
                    f"Expected HTTP 206 Partial Content for the size probe, got HTTP "
                    f"{status}. Refusing to trust an unranged response for sizing."
                )
            content_range = resp.headers.get("Content-Range", "")
            match = _CONTENT_RANGE_RE.fullmatch(content_range.strip())
            if not match or match.group(3) == "*":
                raise RangeResponseError(
                    f"Missing or malformed Content-Range in size probe: {content_range!r}"
                )
            resp.read()  # drain the single probed byte
            return int(match.group(3))
        except RangeResponseError as e:
            last_err = e
            logger.warning("size probe validation failed (attempt %d/%d): %s", attempt + 1, retries, e)
            _backoff_sleep(attempt, retries)
        finally:
            resp.close()
    raise RangeResponseError(f"Giving up after {retries} attempts: {last_err}") from last_err


def open_remote_zip(url: str = ARCHIVE_URL) -> tuple[zipfile.ZipFile, HttpRangeFile]:
    """Open the remote archive's central directory without downloading its body."""
    size = get_remote_size(url)
    f = HttpRangeFile(url, size)
    return zipfile.ZipFile(f), f


def list_real_events(zf: zipfile.ZipFile) -> dict[str, list[str]]:
    """Map event_id ("<year>/fire_<id>") -> sorted list of day filenames (*.tif)."""
    events: dict[str, list[str]] = {}
    for name in zf.namelist():
        if name.endswith("/") or not name.endswith(".tif"):
            continue
        parts = name.split("/")
        if len(parts) != 3:
            continue  # unexpected layout - do not silently misclassify
        year, fire_dir, filename = parts
        event_id = f"{year}/{fire_dir}"
        events.setdefault(event_id, []).append(filename)
    for v in events.values():
        v.sort()
    return events


def missing_files_for_event(
    dest_dir: Path, event_id: str, expected_filenames: list[str]
) -> list[str]:
    """Filenames from `expected_filenames` not already present (and non-empty)
    under dest_dir/event_id. Pure filesystem check - no network, so this is
    cheap to call before deciding whether an event needs (re)fetching."""
    event_dir = Path(dest_dir) / event_id
    if not event_dir.is_dir():
        return list(expected_filenames)
    existing = {p.name for p in event_dir.glob("*.tif") if p.stat().st_size > 0}
    return [name for name in expected_filenames if name not in existing]


def extract_events(
    zf: zipfile.ZipFile,
    event_ids: list[str],
    dest_dir: Path,
) -> list[Path]:
    """Decompress every .tif belonging to the given events into dest_dir,
    preserving the official <year>/fire_<id>/<date>.tif layout. Only the bytes
    of these specific entries are fetched from the remote archive.

    Resumable: a local file that already exists with the exact expected
    (uncompressed) size is left untouched and not re-fetched, so a prior
    partial/interrupted run can be safely re-run to complete it.
    """
    dest_dir = Path(dest_dir)
    written = []
    wanted = set(event_ids)
    for name in zf.namelist():
        if name.endswith("/") or not name.endswith(".tif"):
            continue
        parts = name.split("/")
        if len(parts) != 3:
            continue
        year, fire_dir, filename = parts
        event_id = f"{year}/{fire_dir}"
        if event_id not in wanted:
            continue
        info = zf.getinfo(name)
        out_path = dest_dir / year / fire_dir / filename
        if out_path.exists() and out_path.stat().st_size == info.file_size:
            continue  # already correctly extracted - resume by skipping
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Finish reading (including ZIP CRC validation) before opening the
        # destination. A broken connection must not truncate an existing file
        # or leave a zero-byte TIFF that looks like a successful extraction.
        with zf.open(name) as src:
            data = src.read()
        if out_path.exists():
            # Preserve an incomplete file before repairing it. Valid files
            # were already skipped above.
            backup = out_path.with_name(out_path.name + f".before_repair_{uuid.uuid4().hex}.bak")
            shutil.copy2(out_path, backup)
        out_path.write_bytes(data)
        written.append(out_path)
    return written
