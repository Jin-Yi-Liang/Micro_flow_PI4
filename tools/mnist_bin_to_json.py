#!/usr/bin/env python3
"""Convert a 28x28 MicroFlow .bin image to a Web API JSON request."""

import json
import struct
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <image.bin>", file=sys.stderr)
        return 2

    payload = Path(sys.argv[1]).read_bytes()
    if len(payload) == 784:
        pixels = [value / 255.0 for value in payload]
    elif len(payload) == 784 * 4:
        pixels = list(struct.unpack("<784f", payload))
    else:
        print(f"Expected 784 or 3136 bytes, got {len(payload)}", file=sys.stderr)
        return 1

    if any(not 0.0 <= value <= 1.0 for value in pixels):
        print("Pixel values must be in [0, 1]", file=sys.stderr)
        return 1

    json.dump({"pixels": pixels}, sys.stdout, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
