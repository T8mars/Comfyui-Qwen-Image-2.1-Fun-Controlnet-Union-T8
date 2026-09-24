"""ComfyUI adapter for Alibaba PAI's Qwen Image 2.1 Fun Union branch."""

import json
import struct

import folder_paths


CONTROL_MODES = ("Canny", "Depth", "Grayscale", "HED", "Lineart", "MLSD", "Pose", "Scribble")


def _native_nodes():
    try:
        from comfy_extras.nodes_model_patch import dit_patch_operations
        from comfy_extras.nodes_qwen import QwenImage21FunControlNetApply
        from comfy.ldm.qwen_image21.model import QwenImage21FunControl
    except ImportError as exc:
        raise RuntimeError(
            "Qwen Image 2.1 Fun Union needs ComfyUI with native Qwen Image 2.1 Fun support "
            "(Comfy-Org/ComfyUI PR #16519 or a release containing it)."
        ) from exc
    return dit_patch_operations, QwenImage21FunControl, QwenImage21FunControlNetApply


class QwenImage21UnionLoader:
    @classmethod
    def INPUT_TYPES(cls):
        names = [name for name in folder_paths.get_filename_list("controlnet")
                 if name.startswith("Qwen-Image-2.1-Fun-Controlnet-Union") and name.endswith(".safetensors")]
        return {"required": {"union_model": (names,)}}

    RETURN_TYPES = ("MODEL_PATCH",)
    RETURN_NAMES = ("union_patch",)
    FUNCTION = "load"
    CATEGORY = "Qwen Image 2.1/Union"
    DESCRIPTION = "Loads the 16-block Qwen Image 2.1 Fun Union patch, with checkpoint validation."

    def load(self, union_model):
        path = folder_paths.get_full_path_or_raise("controlnet", union_model)
        with open(path, "rb") as stream:
            header_length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_length))
        keys = set(header)
        required = {"control_img_in.weight", "control_blocks.0.img_mlp.out.weight"}
        if not required.issubset(keys):
            raise ValueError("This is not a Qwen Image 2.1 Fun Union checkpoint.")
        shape = header["control_img_in.weight"]["shape"]
        blocks = sum(f"control_blocks.{i}.after_proj.weight" in keys for i in range(16))
        if shape != [4096, 129] or blocks != 16 or "control_blocks.16.after_proj.weight" in keys:
            raise ValueError(f"Expected Qwen Image 2.1 Fun Union [4096,129] and 16 blocks; got {shape} and {blocks}.")
        import comfy.model_management
        import comfy.model_patcher
        import comfy.utils

        choose_operations, control_class, _ = _native_nodes()
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        dtype, operations = choose_operations(sd)
        model = control_class(
            num_blocks=16,
            control_in_dim=129,
            inner_dim=4096,
            attention_head_dim=sd["control_blocks.0.attn.norm_q.weight"].shape[0],
            mlp_ratio=sd["control_blocks.0.img_mlp.proj.weight"].shape[0] // 4096,
            fused_mlp=False,
            operations=operations,
            device=comfy.model_management.unet_offload_device(),
            dtype=dtype,
        )
        patcher = comfy.model_patcher.CoreModelPatcher(
            model,
            load_device=comfy.model_management.get_torch_device(),
            offload_device=comfy.model_management.unet_offload_device(),
        )
        missing, unexpected = model.load_state_dict(sd, assign=patcher.is_dynamic())
        if missing or unexpected:
            raise RuntimeError(f"UNION checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
        return (patcher,)


class QwenImage21UnionApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "union_patch": ("MODEL_PATCH",),
                "vae": ("VAE",),
                "control_mode": (CONTROL_MODES,),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "control_image": ("IMAGE",),
                "inpaint_image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "Qwen Image 2.1/Union"
    DESCRIPTION = (
        "Apply a prepared Canny, Depth, Grayscale, HED, Lineart, MLSD, Pose or Scribble map. "
        "The mode labels the map; the Union model reads the pixels, with no mode embedding. "
        "For inpainting, connect an image and a white-to-regenerate mask."
    )

    def apply(self, model, union_patch, vae, control_mode, strength, start_percent, end_percent,
              control_image=None, inpaint_image=None, mask=None):
        if control_mode not in CONTROL_MODES:
            raise ValueError(f"Unsupported control mode: {control_mode}")
        if start_percent > end_percent:
            raise ValueError("start_percent must not exceed end_percent")
        if inpaint_image is not None and mask is None:
            raise ValueError("Connect a mask when using inpaint_image (white = regenerate).")
        _, _, native_apply = _native_nodes()
        result = native_apply.execute(
            model=model,
            model_patch=union_patch,
            vae=vae,
            strength=strength,
            start_percent=start_percent,
            end_percent=end_percent,
            control_image=control_image,
            inpaint_image=inpaint_image,
            mask=mask,
        )
        return result.result


class QwenImage21UnionLatentFromImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "resolution": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 32}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "make_latent"
    CATEGORY = "Qwen Image 2.1/Union"
    DESCRIPTION = "Make a 64-channel Qwen Image 2.1 latent at the control image's aspect ratio."

    def make_latent(self, image, resolution):
        import math
        import torch
        import comfy.model_management

        batch, height, width, _ = image.shape
        ratio = width / height
        target_width = max(32, round(math.sqrt(resolution * resolution * ratio) / 32) * 32)
        target_height = max(32, round(math.sqrt(resolution * resolution / ratio) / 32) * 32)
        latent = torch.zeros(
            [batch, 64, target_height // 16, target_width // 16],
            device=comfy.model_management.intermediate_device(),
        )
        return ({"samples": latent},)


NODE_CLASS_MAPPINGS = {
    "QwenImage21UnionLoader": QwenImage21UnionLoader,
    "QwenImage21UnionApply": QwenImage21UnionApply,
    "QwenImage21UnionLatentFromImage": QwenImage21UnionLatentFromImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImage21UnionLoader": "Load Qwen Image 2.1 UNION",
    "QwenImage21UnionApply": "Apply Qwen Image 2.1 UNION",
    "QwenImage21UnionLatentFromImage": "Qwen 2.1 Latent From Control Image",
}
