"""Offline checks for the published workflows and model paths."""

import json
from pathlib import Path
import unittest

from convert_union import MODEL_NAME, output_path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows"
MODES = ("canny", "depth", "grayscale", "hed", "lineart", "mlsd", "pose", "scribble")


class ReleaseTests(unittest.TestCase):
    def test_converter_keeps_output_beside_source_by_default(self):
        source = Path("ComfyUI/models/controlnet/original.safetensors")
        self.assertEqual(output_path(source, None), source.parent / MODEL_NAME)
        explicit = Path("custom/output.safetensors")
        self.assertEqual(output_path(source, explicit), explicit)

    def test_canvas_and_api_links_match(self):
        ui_files = sorted(path for path in WORKFLOWS.glob("*.json")
                          if not path.name.endswith(".api.json"))
        self.assertEqual(len(ui_files), 11)
        for path in ui_files:
            with self.subTest(workflow=path.name):
                graph = json.loads(path.read_text(encoding="utf-8"))
                prompt = json.loads(path.with_suffix(".api.json").read_text(encoding="utf-8"))
                nodes = {node["id"]: node for node in graph["nodes"]}
                links = {link[0]: link for link in graph["links"]}
                self.assertEqual(len(nodes), len(graph["nodes"]))
                self.assertEqual(len(links), len(graph["links"]))
                self.assertEqual(set(prompt), {str(node_id) for node_id in nodes})
                for link_id, origin, origin_slot, target, target_slot, dtype in graph["links"]:
                    source = nodes[origin]["outputs"][origin_slot]
                    destination = nodes[target]["inputs"][target_slot]
                    self.assertEqual(source["type"], dtype)
                    self.assertEqual(destination["type"], dtype)
                    self.assertIn(link_id, source["links"])
                    self.assertEqual(destination["link"], link_id)
                    self.assertEqual(prompt[str(target)]["inputs"][destination["name"]],
                                     [str(origin), origin_slot])

    def test_referenced_images_are_bundled(self):
        for path in WORKFLOWS.glob("*.api.json"):
            with self.subTest(workflow=path.name):
                prompt = json.loads(path.read_text(encoding="utf-8"))
                for node in prompt.values():
                    if node["class_type"] in ("LoadImage", "LoadImageMask"):
                        image = ROOT / "assets" / node["inputs"]["image"]
                        self.assertTrue(image.is_file(), str(image))
                        self.assertGreater(image.stat().st_size, 0)

    def test_mode_and_latent_connections(self):
        names = (*MODES, "inpaint_pose")
        for name in names:
            with self.subTest(mode=name):
                prompt = json.loads((WORKFLOWS / f"qwen21_union_{name}.api.json")
                                    .read_text(encoding="utf-8"))
                expected_mode = {"hed": "HED", "mlsd": "MLSD", "inpaint_pose": "Pose"}.get(
                    name, name.capitalize()
                )
                self.assertEqual(prompt["7"]["inputs"]["control_mode"], expected_mode)
                self.assertEqual(prompt["4"]["inputs"]["union_model"], MODEL_NAME)
                self.assertEqual(prompt["8"]["inputs"]["latent_image"], ["13", 0])
                image_node = "11" if name == "inpaint_pose" else "5"
                self.assertEqual(prompt["13"]["inputs"]["image"], [image_node, 0])
                if name == "inpaint_pose":
                    self.assertEqual(prompt["7"]["inputs"]["inpaint_image"], ["11", 0])
                    self.assertEqual(prompt["7"]["inputs"]["mask"], ["12", 0])

    def test_reference_edit_workflows(self):
        for name, source in (("pose_reference_edit", "14"),
                             ("pose_reference_inpaint", "11")):
            with self.subTest(workflow=name):
                prompt = json.loads((WORKFLOWS / f"qwen21_union_{name}.api.json")
                                    .read_text(encoding="utf-8"))
                self.assertEqual(prompt["6"]["class_type"], "QwenImage21UnionReferenceEncode")
                self.assertEqual(prompt["6"]["inputs"]["reference_image"], [source, 0])
                self.assertEqual(prompt["8"]["inputs"]["positive"], ["6", 0])
                self.assertEqual(prompt["8"]["inputs"]["negative"], ["6", 1])
                self.assertEqual(prompt["8"]["inputs"]["latent_image"], ["6", 2])
                if name == "pose_reference_inpaint":
                    self.assertEqual(prompt["7"]["inputs"]["inpaint_image"], ["11", 0])
                    self.assertEqual(prompt["7"]["inputs"]["mask"], ["12", 0])
                else:
                    self.assertNotIn("inpaint_image", prompt["7"]["inputs"])
                    self.assertNotIn("mask", prompt["7"]["inputs"])


if __name__ == "__main__":
    unittest.main()
