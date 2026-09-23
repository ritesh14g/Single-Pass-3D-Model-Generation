"""Upload the box bundle to the GPU box's Jupyter server, in pieces, from the laptop.

A 1.3 GB file through the Jupyter browser upload stalls or times out. This sends it through the
same server's file API (``/api/contents``) in small chunks instead:

  * the zip is sent as numbered parts (``<zip>.part001`` ...), each in ``--chunk-mb`` chunks, so a
    proxy that limits request size or cuts long requests is not a problem;
  * **resumable**: a part already on the box at the right size is skipped, so after a dropped
    connection just run the same command again;
  * ``<zip>.sha256`` and ``box_restore.sh`` are uploaded too; box_restore.sh joins the parts and
    checks the checksum before unpacking.

    python scripts/box_upload.py --url "https://<host>/user/<you>/lab?token=<token>"
    python scripts/box_upload.py --url https://<host>/user/<you>/ --token <token> [zip]

``--url``: copy the address bar of your Jupyter tab (anything after ``/lab`` or ``/tree`` is ignored).
The token: the ``?token=`` in that URL if there is one, otherwise JupyterHub → Control Panel → Token
→ "Request new API token". It is used for this upload only and never written anywhere; you can
also set it in the ``JUPYTER_TOKEN`` environment variable instead of the command line.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit, urlunsplit

import requests

ROOT = Path(__file__).resolve().parents[1]


def base_and_token(url: str, token: str | None) -> tuple[str, str | None]:
    """Server root (…/user/<name>/ or …/) and the token, from what is in the browser's address bar."""
    parts = urlsplit(url.strip())
    token = token or os.environ.get("JUPYTER_TOKEN") or (parse_qs(parts.query).get("token") or [None])[0]
    path = parts.path
    for marker in ("/lab", "/tree", "/notebooks", "/edit", "/terminals", "/api/"):
        if marker in path:
            path = path[: path.index(marker)]
            break
    return urlunsplit((parts.scheme, parts.netloc, path.rstrip("/") + "/", "", "")), token


class Box:
    def __init__(self, base: str, token: str | None, remote_dir: str, timeout: float, chunk_bytes: int):
        self.base, self.dir, self.timeout = base, remote_dir.strip("/"), timeout
        self.chunk_bytes = chunk_bytes  # shrinks for good if a proxy rejects a request as too large
        self.s = requests.Session()
        if token:
            self.s.headers["Authorization"] = f"token {token}"

    def _url(self, name: str) -> str:
        return f"{self.base}api/contents/{self.dir + '/' if self.dir else ''}{name}"

    def check(self) -> None:
        r = self.s.get(f"{self.base}api/contents/{self.dir}", params={"content": 0}, timeout=self.timeout)
        if r.status_code in (401, 403):
            sys.exit(f"the server refused the token ({r.status_code}). Copy a fresh URL/token from the Jupyter tab.")
        if r.status_code == 404:
            sys.exit(f"remote folder '{self.dir}' does not exist on the box")
        if r.status_code != 200 or "json" not in r.headers.get("content-type", ""):
            sys.exit(f"{self.base} does not look like a Jupyter server (HTTP {r.status_code}); check --url")

    def size(self, name: str) -> int | None:
        r = self.s.get(self._url(name), params={"content": 0}, timeout=self.timeout)
        return r.json().get("size") if r.status_code == 200 else None

    def put_chunk(self, name: str, data: bytes, chunk: int) -> requests.Response:
        """Jupyter's chunked save: chunk 1 creates the file, 2..n append, -1 appends and finishes."""
        body = {"type": "file", "format": "base64", "name": name,
                "path": f"{self.dir + '/' if self.dir else ''}{name}", "chunk": chunk,
                "content": base64.b64encode(data).decode("ascii")}
        return self.s.put(self._url(name), json=body, timeout=self.timeout)

    def upload(self, name: str, read, total: int, label: str) -> None:
        """Send one file (``read(offset, n)`` gives its bytes); restarts it from scratch on any error."""
        for attempt in range(1, 6):
            try:
                sent, n = 0, 0
                started = time.time()
                while sent < total or n == 0:
                    data = read(sent, min(self.chunk_bytes, total - sent))
                    n += 1
                    last = sent + len(data) >= total
                    r = self.put_chunk(name, data, -1 if last and n > 1 else n)
                    if r.status_code == 413 and self.chunk_bytes > 256 * 1024:
                        self.chunk_bytes //= 2
                        print(f"\n   the proxy limits request size: chunks -> {self.chunk_bytes // 1024} KB, "
                              f"restarting {label}")
                        raise RuntimeError("request too large")
                    r.raise_for_status()
                    sent += len(data)
                    rate = sent / 1e6 / max(time.time() - started, 1e-3)
                    print(f"\r   {label}: {sent / 1e6:7.1f} / {total / 1e6:.1f} MB  {rate:5.1f} MB/s", end="", flush=True)
                    if last:
                        break
                print()
                got = self.size(name)
                if got == total:
                    return
                raise RuntimeError(f"size on the box {got} != {total}")
            except (requests.RequestException, RuntimeError) as exc:
                print(f"\n   {label}: attempt {attempt} failed ({exc}); retrying the whole piece")
                time.sleep(min(5 * attempt, 30))
        sys.exit(f"{label} failed 5 times; run the same command again later (finished parts are kept)")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip", nargs="?", help="bundle to send (default: newest in data/box_upload)")
    ap.add_argument("--url", required=True, help="your Jupyter tab's address (with ?token=... if it has one)")
    ap.add_argument("--token", help="Jupyter/JupyterHub API token (or set JUPYTER_TOKEN)")
    ap.add_argument("--remote-dir", default="", help="folder on the box, relative to the Jupyter root (default: home)")
    ap.add_argument("--part-mb", type=int, default=100, help="size of each resumable part")
    ap.add_argument("--chunk-mb", type=float, default=4, help="size of each request (lower it if the proxy rejects)")
    ap.add_argument("--timeout", type=float, default=120)
    args = ap.parse_args()

    if args.zip:
        zip_path = Path(args.zip)
    else:
        found = sorted((ROOT / "data" / "box_upload").glob("box_upload_*.zip"))
        if not found:
            sys.exit("no bundle in data/box_upload: run scripts/box_pack.py first")
        zip_path = found[-1]
    total = zip_path.stat().st_size
    base, token = base_and_token(args.url, args.token)
    box = Box(base, token, args.remote_dir, args.timeout, int(args.chunk_mb * 1024 * 1024))
    box.check()
    print(f"box: {base}  ({'token' if token else 'no token'})")
    print(f"sending {zip_path.name}: {total / 1e6:,.0f} MB")

    digest = sha256(zip_path)
    part_bytes = args.part_mb * 1024 * 1024
    count = (total + part_bytes - 1) // part_bytes
    part_sums = []
    with zip_path.open("rb") as fh:
        for _ in range(count):
            part_sums.append(hashlib.sha256(fh.read(part_bytes)).hexdigest())
    started = time.time()
    with zip_path.open("rb") as fh:
        for i in range(count):
            name = f"{zip_path.name}.part{i + 1:03d}"
            start = i * part_bytes
            size = min(part_bytes, total - start)
            if box.size(name) == size:
                print(f"   {name}: already on the box")
                continue

            def read(offset, n, start=start):
                fh.seek(start + offset)
                return fh.read(n)

            box.upload(name, read, size, f"part {i + 1}/{count}")

    # sha256sum format: the whole zip first, then each part, so the box can name a bad part.
    sidecar = (f"{digest}  {zip_path.name}\n"
               + "".join(f"{h}  {zip_path.name}.part{i + 1:03d}\n" for i, h in enumerate(part_sums))).encode()
    box.upload(f"{zip_path.name}.sha256", lambda o, n: sidecar[o:o + n], len(sidecar), "checksum")
    script = (ROOT / "scripts" / "box_restore.sh").read_bytes()
    box.upload("box_restore.sh", lambda o, n: script[o:o + n], len(script), "box_restore.sh")
    print(f"\nDONE in {(time.time() - started) / 60:.1f} min: {count} parts + checksum + box_restore.sh are in "
          f"~/{args.remote_dir + '/' if args.remote_dir else ''} on the box.")
    print("On the box (Jupyter terminal):")
    print(f"   export HF_TOKEN=hf_...     # fresh token, this shell only")
    print(f"   cd ~/{args.remote_dir} && nohup bash box_restore.sh ~/{args.remote_dir + '/' if args.remote_dir else ''}"
          f"{zip_path.name} > ~/box_setup.log 2>&1 &")
    print("   tail -f ~/box_setup.log    # Ctrl+C stops watching only; the restore keeps running")
    print("   (it joins the parts, checks the sha256, then restores; the parts are deleted once it verifies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
