#!/usr/bin/env python3
"""Convert the bundled FP32 MicroFlow MNIST model to ncnn param/bin files."""

from __future__ import annotations

import argparse
import array
import struct
import sys
from pathlib import Path


HEADER = struct.Struct("<IIIIII64s")
LAYER_HEADER = struct.Struct("<IIIIII")
TENSOR_DESC = struct.Struct("<8I")


def read_tensor(stream) -> tuple[tuple[int, ...], array.array]:
    raw_desc = stream.read(TENSOR_DESC.size)
    if len(raw_desc) != TENSOR_DESC.size:
        raise RuntimeError("Truncated tensor descriptor")
    ndim, *fields = TENSOR_DESC.unpack(raw_desc)
    if ndim == 0 or ndim > 4:
        raise RuntimeError(f"Unsupported tensor rank: {ndim}")
    shape = tuple(fields[:4][:ndim])
    dtype, size, _offset = fields[4:]
    if dtype != 0:
        raise RuntimeError("Only FP32 MicroFlow tensors can be exported to ncnn")
    values = array.array("f")
    values.fromfile(stream, size)
    if sys.byteorder != "little":
        values.byteswap()
    expected_size = 1
    for dimension in shape:
        expected_size *= dimension
    if expected_size != size:
        raise RuntimeError(f"Invalid tensor size: shape={shape}, size={size}")
    return shape, values


def transpose_2d(values: array.array, rows: int, columns: int) -> array.array:
    result = array.array("f", [0.0]) * (rows * columns)
    for row in range(rows):
        source_offset = row * columns
        for column in range(columns):
            result[column * rows + row] = values[source_offset + column]
    return result


def load_weights(path: Path) -> dict[str, array.array]:
    tensors: list[tuple[tuple[int, ...], array.array]] = []
    with path.open("rb") as stream:
        raw_header = stream.read(HEADER.size)
        if len(raw_header) != HEADER.size:
            raise RuntimeError("Truncated MicroFlow model header")
        magic, version, layer_count, _tensor_count, _offset, _size, _description = HEADER.unpack(raw_header)
        if magic != 0x4D464C57 or version != 2:
            raise RuntimeError("Expected a MicroFlow V2 model")
        for _ in range(layer_count):
            raw_layer = stream.read(LAYER_HEADER.size)
            if len(raw_layer) != LAYER_HEADER.size:
                raise RuntimeError("Truncated layer header")
            layer_type = LAYER_HEADER.unpack(raw_layer)[0]
            if layer_type in (1, 13):
                tensors.append(read_tensor(stream))
                tensors.append(read_tensor(stream))

    expected_shapes = [
        (32, 1, 3, 3), (32,),
        (64, 32, 3, 3), (64,),
        (3136, 128), (128,),
        (128, 10), (10,),
    ]
    shapes = [shape for shape, _ in tensors]
    if shapes != expected_shapes:
        raise RuntimeError(f"Unexpected MNIST model tensors: {shapes}")

    return {
        "conv1_weight": tensors[0][1],
        "conv1_bias": tensors[1][1],
        "conv2_weight": tensors[2][1],
        "conv2_bias": tensors[3][1],
        # ncnn InnerProduct stores one contiguous input vector per output.
        "fc1_weight": transpose_2d(tensors[4][1], 3136, 128),
        "fc1_bias": tensors[5][1],
        "fc2_weight": transpose_2d(tensors[6][1], 128, 10),
        "fc2_bias": tensors[7][1],
    }


def write_ncnn(weights: dict[str, array.array], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    param_path = output_prefix.with_suffix(".param")
    bin_path = output_prefix.with_suffix(".bin")
    param_path.write_text(
        "\n".join(
            [
                "7767517",
                "8 8",
                "Input data 0 1 data 0=28 1=28 2=1",
                "Convolution conv1 1 1 data conv1 0=32 1=3 4=1 5=1 6=288 9=1",
                "Pooling pool1 1 1 conv1 pool1 0=0 1=2 2=2",
                "Convolution conv2 1 1 pool1 conv2 0=64 1=3 4=1 5=1 6=18432 9=1",
                "Pooling pool2 1 1 conv2 pool2 0=0 1=2 2=2",
                "InnerProduct fc1 1 1 pool2 fc1 0=128 1=1 2=401408 9=1",
                "InnerProduct fc2 1 1 fc1 fc2 0=10 1=1 2=1280",
                "Softmax prob 1 1 fc2 prob",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with bin_path.open("wb") as stream:
        for layer in ("conv1", "conv2", "fc1", "fc2"):
            stream.write(b"\x00\x00\x00\x00")  # ncnn raw-FP32 weight tag
            weights[f"{layer}_weight"].tofile(stream)
            weights[f"{layer}_bias"].tofile(stream)

    print(f"Wrote {param_path}")
    print(f"Wrote {bin_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output_prefix", type=Path)
    args = parser.parse_args()
    write_ncnn(load_weights(args.model), args.output_prefix)


if __name__ == "__main__":
    main()
