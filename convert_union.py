"""Repack the original Union checkpoint for ComfyUI's MODEL_PATCH loader.

Only the safetensors header changes. Tensor payload bytes and offsets are kept.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct


SOURCE_REVISION = "8a4702014d4dabb5f896fcba917e2ee0a961465f"
EXPECTED_SIZE = 7550979904
EXPECTED_SHA256 = "65d6b66d734da9e7ff5ef04e7db3a133553a52a3f29a7fcb3e9cce8fa21dcfcd"
MODEL_NAME = "Qwen-Image-2.1-Fun-Controlnet-Union-ComfyUI.safetensors"


def output_path(source, requested):
    return requested if requested is not None else source.parent / MODEL_NAME


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(source, output):
    if source.stat().st_size != EXPECTED_SIZE:
        raise ValueError(f"Wrong source size: {source.stat().st_size}, expected {EXPECTED_SIZE}")
    source_digest = sha256(source)
    if source_digest != EXPECTED_SHA256:
        raise ValueError(f"Source SHA256 mismatch: {source_digest}")

    with source.open("rb") as stream:
        old_header_length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(old_header_length))
        key_names = set(header) - {"__metadata__"}
        if header["control_img_in.weight"]["shape"] != [4096, 129]:
            raise ValueError("Expected the Qwen Image 2.1 control input [4096,129]")
        if not all(f"control_blocks.{i}.after_proj.weight" in key_names for i in range(16)):
            raise ValueError("The checkpoint does not contain all 16 control blocks")
        if "control_blocks.16.after_proj.weight" in key_names:
            raise ValueError("Unexpected extra control block")

        header["__metadata__"] = {
            **header.get("__metadata__", {}),
            "format": "pt",
            "comfyui.model_type": "MODEL_PATCH",
            "comfyui.base_model": "Qwen-Image-2.1",
            "comfyui.control_modes": "Canny,Depth,Grayscale,HED,Lineart,MLSD,Pose,Scribble,Inpaint",
            "source.repo": "alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union",
            "source.revision": SOURCE_REVISION,
            "source.sha256": source_digest,
        }
        new_header = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        padding = (-len(new_header)) % 8
        new_header += b" " * padding

        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(output.suffix + ".tmp")
        with temp.open("wb") as dest:
            dest.write(struct.pack("<Q", len(new_header)))
            dest.write(new_header)
            while chunk := stream.read(8 * 1024 * 1024):
                dest.write(chunk)
        os.replace(temp, output)

    with output.open("rb") as stream:
        converted_header_length = struct.unpack("<Q", stream.read(8))[0]
        converted_header = json.loads(stream.read(converted_header_length))
    if set(converted_header) - {"__metadata__"} != key_names:
        raise RuntimeError("Converted tensor names differ from source")
    if any(converted_header[key] != header[key] for key in key_names):
        raise RuntimeError("Converted tensor descriptors differ from source")

    receipt = {
        "source": str(source),
        "source_sha256": source_digest,
        "output": str(output),
        "output_sha256": sha256(output),
        "tensor_count": len(key_names),
        "control_blocks": 16,
        "source_revision": SOURCE_REVISION,
    }
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(convert(args.source, output_path(args.source, args.output)), indent=2))
