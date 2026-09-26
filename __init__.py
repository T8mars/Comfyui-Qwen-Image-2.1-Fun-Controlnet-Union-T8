"""ComfyUI adapter for Alibaba PAI's Qwen Image 2.1 Fun Union branch."""

import json
import os
import struct

import folder_paths


CONTROL_MODES = ("Canny", "Depth", "Grayscale", "HED", "Lineart", "MLSD", "Pose", "Scribble")
MAX_HEADER_BYTES = 1024 * 1024


def _checkpoint_header(path):
    with open(path, "rb") as stream:
        file_size = os.fstat(stream.fileno()).st_size
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("UNION checkpoint has no safetensors header.")
        header_length = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_length <= min(file_size - 8, MAX_HEADER_BYTES):
            raise ValueError("UNION checkpoint has an invalid safetensors header length.")
        header_bytes = stream.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError("UNION checkpoint has a truncated safetensors header.")
        return json.loads(header_bytes)


def _native_nodes():
    try:
        from comfy_extras.nodes_model_patch import dit_patch_operations
        from comfy_extras.nodes_qwen import QwenImage21FunControlNetApply
        from comfy.ldm.qwen_image21.model import QwenImage21FunControl
    except ImportError:
        from .union_compat import (
            dit_patch_operations,
            QwenImage21FunControl,
            QwenImage21FunControlNetApply,
        )
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
        header = _checkpoint_header(path)
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
        "For inpainting, connect an image and a white-to-regenerate mask. "
        "Condition images and mask must each contain one image."
    )

    def apply(self, model, union_patch, vae, control_mode, strength, start_percent, end_percent,
              control_image=None, inpaint_image=None, mask=None):
        if control_mode not in CONTROL_MODES:
            raise ValueError(f"Unsupported control mode: {control_mode}")
        if start_percent > end_percent:
            raise ValueError("start_percent must not exceed end_percent")
        if inpaint_image is not None and mask is None:
            raise ValueError("Connect a mask when using inpaint_image (white = regenerate).")
        for name, image in (("control_image", control_image), ("inpaint_image", inpaint_image)):
            if image is not None and image.shape[0] != 1:
                raise ValueError(f"{name} must contain one image; native Qwen Image 2.1 uses only the first.")
        if mask is not None and mask.ndim >= 3 and mask.shape[0] != 1:
            raise ValueError("mask must contain one image; native Qwen Image 2.1 uses only the first.")
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
        return result.result if hasattr(result, "result") else result


class QwenImage21UnionReferenceEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "reference_image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "resolution": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 32}),
            },
            "optional": {
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "Qwen Image 2.1/Union"
    DESCRIPTION = (
        "reference_image is image_1, the edit target and canvas. Optional image_2 and image_3 "
        "provide visual references. Connect positive, negative and latent to the sampler; "
        "the latent must come from this node to match image_1. Use a prepared map for "
        "UNION control_image. For masked editing also connect the source to inpaint_image."
    )

    def encode(self, clip, vae, reference_image, prompt, negative_prompt, resolution,
               reference_image_2=None, reference_image_3=None):
        images = {}
        for index, image in enumerate((reference_image, reference_image_2, reference_image_3), start=1):
            if image is None:
                continue
            if image.shape[0] != 1:
                name = "reference_image" if index == 1 else f"reference_image_{index}"
                raise ValueError(f"{name} must contain one image.")
            images[f"image_{index}"] = image
        from comfy_extras.nodes_qwen import TextEncodeQwenImage21

        result = TextEncodeQwenImage21.execute(
            clip=clip,
            prompt=prompt,
            negative_prompt=negative_prompt,
            vae=vae,
            resolution=resolution,
            images=images,
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
    DESCRIPTION = "Make a 64-channel Qwen Image 2.1 latent from one image, with a 4096-pixel output-side limit."

    def make_latent(self, image, resolution):
        import math

        batch, height, width, _ = image.shape
        if batch != 1:
            raise ValueError("Connect one image; native Qwen Image 2.1 uses only the first control image.")
        if height == 0 or width == 0:
            raise ValueError("Control image must have nonzero width and height.")
        ratio = width / height
        target_width = max(32, round(math.sqrt(resolution * resolution * ratio) / 32) * 32)
        target_height = max(32, round(math.sqrt(resolution * resolution / ratio) / 32) * 32)
        if max(target_width, target_height) > 4096:
            raise ValueError(
                f"Control image aspect ratio produces {target_width}x{target_height}; "
                "reduce resolution or crop/pad the image so neither output side exceeds 4096."
            )
        import torch
        import comfy.model_management

        latent = torch.zeros(
            [batch, 64, target_height // 16, target_width // 16],
            device=comfy.model_management.intermediate_device(),
        )
        return ({"samples": latent},)


NODE_CLASS_MAPPINGS = {
    "QwenImage21UnionLoader": QwenImage21UnionLoader,
    "QwenImage21UnionApply": QwenImage21UnionApply,
    "QwenImage21UnionReferenceEncode": QwenImage21UnionReferenceEncode,
    "QwenImage21UnionLatentFromImage": QwenImage21UnionLatentFromImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImage21UnionLoader": "Load Qwen Image 2.1 UNION",
    "QwenImage21UnionApply": "Apply Qwen Image 2.1 UNION",
    "QwenImage21UnionReferenceEncode": "Qwen 2.1 UNION Image Reference Encode",
    "QwenImage21UnionLatentFromImage": "Qwen 2.1 Latent From Control Image",
}
