"""Reject inputs that the native Qwen control patch would silently truncate."""

import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_node_module():
    folder_paths = types.ModuleType("folder_paths")
    spec = importlib.util.spec_from_file_location("union_node_for_tests", ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"folder_paths": folder_paths}):
        spec.loader.exec_module(module)
    return module


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node = load_node_module()

    def test_rejects_oversized_or_truncated_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.safetensors"
            path.write_bytes(struct.pack("<Q", 1 << 60) + b"{}")
            with self.assertRaisesRegex(ValueError, "header length"):
                self.node._checkpoint_header(path)
            path.write_bytes(b"short")
            with self.assertRaisesRegex(ValueError, "no safetensors header"):
                self.node._checkpoint_header(path)

    def test_rejects_multi_image_conditioning(self):
        many_images = types.SimpleNamespace(shape=(2, 512, 512, 3))
        one_mask = types.SimpleNamespace(shape=(1, 512, 512), ndim=3)
        many_masks = types.SimpleNamespace(shape=(2, 512, 512), ndim=3)
        apply = self.node.QwenImage21UnionApply().apply
        fixed = (None, None, None, "Canny", 1.0, 0.0, 1.0)
        with self.assertRaisesRegex(ValueError, "control_image must contain one"):
            apply(*fixed, control_image=many_images)
        with self.assertRaisesRegex(ValueError, "inpaint_image must contain one"):
            apply(*fixed, inpaint_image=many_images, mask=one_mask)
        with self.assertRaisesRegex(ValueError, "mask must contain one"):
            apply(*fixed, mask=many_masks)

    def test_rejects_multi_image_or_extreme_latent(self):
        make = self.node.QwenImage21UnionLatentFromImage().make_latent
        with self.assertRaisesRegex(ValueError, "Connect one image"):
            make(types.SimpleNamespace(shape=(2, 512, 512, 3)), 1024)
        with self.assertRaisesRegex(ValueError, "neither output side exceeds 4096"):
            make(types.SimpleNamespace(shape=(1, 4096, 1, 3)), 1024)

    def test_reference_requires_single_image(self):
        encode = self.node.QwenImage21UnionReferenceEncode().encode
        with self.assertRaisesRegex(ValueError, "reference_image must contain one"):
            encode(None, None, types.SimpleNamespace(shape=(2, 512, 512, 3)), "", "", 1024)
        with self.assertRaisesRegex(ValueError, "reference_image_2 must contain one"):
            encode(None, None, types.SimpleNamespace(shape=(1, 512, 512, 3)), "", "", 1024,
                   reference_image_2=types.SimpleNamespace(shape=(2, 512, 512, 3)))

    def test_reference_forwards_images_in_order(self):
        first = types.SimpleNamespace(shape=(1, 512, 512, 3))
        second = types.SimpleNamespace(shape=(1, 512, 512, 3))
        captured = {}

        class Native:
            @staticmethod
            def execute(**kwargs):
                captured.update(kwargs)
                return types.SimpleNamespace(result=("positive", "negative", "latent"))

        native_module = types.ModuleType("comfy_extras.nodes_qwen")
        native_module.TextEncodeQwenImage21 = Native
        with patch.dict(sys.modules, {"comfy_extras": types.ModuleType("comfy_extras"),
                                      "comfy_extras.nodes_qwen": native_module}):
            result = self.node.QwenImage21UnionReferenceEncode().encode(
                None, None, first, "edit <image1> using <image2>", "", 1024,
                reference_image_2=second,
            )
        self.assertEqual(result, ("positive", "negative", "latent"))
        self.assertIs(captured["images"]["image_1"], first)
        self.assertIs(captured["images"]["image_2"], second)


if __name__ == "__main__":
    unittest.main()
