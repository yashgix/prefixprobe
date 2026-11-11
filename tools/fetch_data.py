"""Download the Multi-step SWE results into data/ and check them against data/MANIFEST.json.

The files have no stated license, so they are never committed: this script is how a reader gets them.
`--verify` only checks files already present and needs no network.
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def problems(manifest):
    return [name for name, entry in manifest["files"].items()
            if not (DATA / name).is_file() or sha256(DATA / name) != entry["sha256"]]  # fmt: skip


def download(manifest):
    for name in manifest["files"]:
        target = DATA / name
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(manifest["source"] + name, headers={"User-Agent": "prefixprobe fetch"})
        with urllib.request.urlopen(request, timeout=60) as response:
            target.write_bytes(response.read())
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="check files only; no network")
    args = parser.parse_args()
    manifest = json.loads((DATA / "MANIFEST.json").read_text())
    if not args.verify:
        download(manifest)
    bad = problems(manifest)
    for name in bad:
        print(f"missing or changed: {name}", file=sys.stderr)
    if bad:
        sys.exit("the site's data no longer matches the pinned snapshot; results may differ")
    print(f"{len(manifest['files'])} files match MANIFEST.json")


if __name__ == "__main__":
    main()
