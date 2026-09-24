# Release audit · 1.0.1

Twenty focused checks were completed for this release: rounds 1–10 by an independent agent, and rounds 11–20 by the maintainer. Automated checks run with `python -m unittest discover -s tests -v` on Python 3.10 or newer. Runtime checks used the native Qwen Image 2.1 Fun ComfyUI branch and the converted checkpoint.

| # | Check | Result |
| ---: | --- | --- |
| 1 | Import and node registration | Three node classes register and appear in `/object_info`. |
| 2 | Input, output, and function contracts | Node signatures match the ComfyUI API graphs. |
| 3 | Model discovery | The loader reads only matching safetensors from `models/controlnet/`. |
| 4 | Safetensors header parsing | Fixed: invalid or oversized lengths are rejected before allocation. |
| 5 | Union constructor | 16 blocks, 4096 inner channels, and 129 control inputs match the checkpoint. |
| 6 | State loading | The checkpoint loads into the native control class without missing or unexpected tensors. |
| 7 | Apply signature | Arguments and return value match native `QwenImage21FunControlNetApply.execute`. |
| 8 | Mode selection | Eight labels match prepared control maps; the Union checkpoint has no mode embedding. |
| 9 | Conditioning batches | Fixed: connected image or mask batches above one are rejected instead of silently truncating. |
| 10 | Latent sizing | Fixed: empty, batched, and extreme aspect ratio inputs now produce clear errors. |
| 11 | Repository packaging | No model weight is tracked by Git; the published package has a root node entry point. |
| 12 | Source examples | All eight control maps match the upstream mode examples. |
| 13 | Workflow graph integrity | All nine canvas graphs and nine API prompts have matching nodes, links, and model names. |
| 14 | Workflow assets | Every referenced sample image is bundled and nonempty. |
| 15 | Canny inference | A post-fix GPU run completed and saved `Qwen21_audit_canny_00001_.png`. |
| 16 | Inpaint inference | A post-fix GPU run completed and saved `Qwen21_audit_inpaint_pose_00001_.png`. |
| 17 | Converted model | The 180-tensor, 16-block local checkpoint matches the published SHA-256 receipt. |
| 18 | Converter output | Fixed: default output stays beside the source checkpoint; `--output` remains explicit. |
| 19 | Documentation | English and Chinese READMEs link to each other, document paths and licensing, and include the supplied links. |
| 20 | Registry workflow | Versioned Comfy Registry publishing runs the unit suite before the official publish action. |

The automated suite has seven tests. It validates release graph wiring, sample assets, converter path selection, and the boundary guards. The canvas JSON files were validated as graph files and through ComfyUI's prompt API; direct mouse drag-and-drop was not part of this audit.
