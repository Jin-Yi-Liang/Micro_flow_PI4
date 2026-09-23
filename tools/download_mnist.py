#!/usr/bin/env python3
"""Download and validate the official MNIST test-set IDX files."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import struct
import urllib.request
from pathlib import Path


BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist"
FILES = {
    "t10k-images-idx3-ubyte.gz": ("9fb629c4189551a2d022fa330f9573f3", 2051),
    "t10k-labels-idx1-ubyte.gz": ("ec29112dd5afa0611ce80d1b7f02629c", 2049),
}


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(name: str, expected_md5: str, destination: Path) -> Path:
    compressed = destination / name
    if not compressed.exists() or md5sum(compressed) != expected_md5:
        print(f"Downloading {name}...")
        request = urllib.request.Request(
            f"{BASE_URL}/{name}", headers={"User-Agent": "MicroFlow/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            with compressed.open("wb") as output:
                shutil.copyfileobj(response, output)
    actual_md5 = md5sum(compressed)
    if actual_md5 != expected_md5:
        raise RuntimeError(
            f"Checksum mismatch for {name}: {actual_md5} != {expected_md5}"
        )
    return compressed


def decompress_and_validate(compressed: Path, expected_magic: int) -> Path:
    output = compressed.with_suffix("")
    with gzip.open(compressed, "rb") as source, output.open("wb") as destination:
        shutil.copyfileobj(source, destination)
    with output.open("rb") as stream:
        magic = struct.unpack(">I", stream.read(4))[0]
    if magic != expected_magic:
        raise RuntimeError(f"Invalid IDX magic in {output}: {magic}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/MNIST/raw"),
        help="Directory for downloaded and decompressed test files",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, (checksum, magic) in FILES.items():
        archive = download_file(name, checksum, args.output_dir)
        output = decompress_and_validate(archive, magic)
        print(f"Ready: {output}")


if __name__ == "__main__":
    main()
