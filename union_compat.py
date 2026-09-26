"""Qwen Image 2.1 Fun control support for stock ComfyUI 0.37.

Adapted from Comfy-Org/ComfyUI PR #16519 by kijai. GPL-3.0-only; see
LICENSE_COMFYUI_GPL-3.0.txt. This module is used only until the upstream
implementation is present in ComfyUI.
"""

import inspect

import torch
import torch.nn.functional as F

import comfy.latent_formats
import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.patcher_extension
import comfy.utils
from comfy.ldm.qwen_image21.model import QwenImage21TransformerBlock


def dit_patch_operations(state):
    quantization = comfy.utils.detect_layer_quantization(state, "")
    if quantization is not None:
        return torch.bfloat16, comfy.ops.mixed_precision_ops(quantization, torch.bfloat16)
    device = comfy.model_management.get_torch_device()
    dtype = comfy.model_management.unet_dtype(
        model_params=-1,
        supported_dtypes=[torch.bfloat16, torch.float32],
        weight_dtype=comfy.utils.weight_dtype(state),
    )
    cast_dtype = comfy.model_management.unet_manual_cast(
        dtype, device, supported_dtypes=[torch.bfloat16, torch.float32]
    )
    return dtype, comfy.ops.pick_operations(dtype, cast_dtype, load_device=device)


class _UnionBlock(QwenImage21TransformerBlock):
    def __init__(self, dim, heads, head_dim, mlp_ratio, fused_mlp, first, dtype, device, operations):
        super().__init__(dim, heads, head_dim, mlp_ratio, fused_mlp=fused_mlp,
                         dtype=dtype, device=device, operations=operations)
        if first:
            self.before_proj = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.after_proj = operations.Linear(dim, dim, dtype=dtype, device=device)


class QwenImage21FunControl(torch.nn.Module):
    def __init__(self, num_blocks=16, control_in_dim=129, inner_dim=4096,
                 attention_head_dim=128, mlp_ratio=3, fused_mlp=False,
                 operations=None, device=None, dtype=None):
        super().__init__()
        self.control_img_in = operations.Linear(control_in_dim, inner_dim, dtype=dtype, device=device)
        self.control_blocks = torch.nn.ModuleList([
            _UnionBlock(inner_dim, inner_dim // attention_head_dim, attention_head_dim,
                        mlp_ratio, fused_mlp, index == 0, dtype, device, operations)
            for index in range(num_blocks)
        ])

    def init_stream(self, base_sequence, control_tokens, prefix_len):
        stream = torch.zeros_like(base_sequence)
        stream[:, prefix_len:] = self.control_img_in(control_tokens)
        return self.control_blocks[0].before_proj(stream) + base_sequence

    def step(self, index, stream, modulation, position, attention, prefix_len, options):
        block = self.control_blocks[index]
        stream = block(stream, modulation, position, attention, prefix_len, options)
        return stream, block.after_proj(stream)


class _ControlRuntime:
    def __init__(self, patcher, vae, control_image, inpaint_image, mask,
                 strength, layers, sigma_start, sigma_end):
        self.patcher = patcher
        self.vae = vae
        self.control_image = control_image
        self.inpaint_image = inpaint_image
        self.mask = mask
        self.strength = strength
        self.layers = layers
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.active = False
        self.control = None
        self.stream = None
        self.pristine = None

    def prepare(self, latent_h, latent_w):
        if self.control is not None and self.control.shape[-2:] == (latent_h, latent_w):
            return
        pixel_w = latent_w * self.vae.spacial_compression_encode()
        pixel_h = latent_h * self.vae.spacial_compression_encode()
        latent_format = comfy.latent_formats.QwenImage21()
        loaded = comfy.model_management.loaded_models(only_currently_used=True)
        try:
            structural = torch.zeros(1, latent_format.latent_channels, latent_h, latent_w)
            if self.control_image is not None:
                image = comfy.utils.common_upscale(
                    self.control_image[:1].movedim(-1, 1), pixel_w, pixel_h, "bicubic", "disabled"
                ).movedim(1, -1)
                structural = latent_format.process_in(self.vae.encode(image)).float().cpu()

            regenerate = torch.ones(1, 1, pixel_h, pixel_w)
            if self.mask is not None:
                mask = self.mask.reshape(-1, 1, *self.mask.shape[-2:])[:1].float().cpu()
                regenerate = (comfy.utils.common_upscale(
                    mask, pixel_w, pixel_h, "bilinear", "disabled"
                ) >= 0.5).float()

            source = torch.zeros_like(structural)
            if self.inpaint_image is not None:
                image = comfy.utils.common_upscale(
                    self.inpaint_image[:1].movedim(-1, 1), pixel_w, pixel_h, "bicubic", "disabled"
                ).float().cpu()
                masked = image * (1 - regenerate) + 0.5 * regenerate
                source = latent_format.process_in(self.vae.encode(masked.movedim(1, -1))).float().cpu()

            keep = 1 - F.interpolate(regenerate, size=(latent_h, latent_w), mode="nearest")
        finally:
            comfy.model_management.load_models_gpu(loaded)
        self.control = torch.cat((structural, keep, source), dim=1)

    def around_model(self, executor, x, timestep, context, ref_latents,
                     image_slots, transformer_options, **kwargs):
        sigma = float(timestep.flatten()[0])
        self.active = self.sigma_end <= sigma <= self.sigma_start
        if self.active:
            with comfy.model_prefetch.pause_malloc_graph():
                self.prepare(*x.shape[-2:])
        else:
            replacements = transformer_options.get("patches_replace", {}).get("dit", {})
            replacements = {
                key: value.previous if isinstance(value, _BlockPatch) and value.runtime is self else value
                for key, value in replacements.items()
            }
            replacements = {key: value for key, value in replacements.items() if value is not None}
            transformer_options = {
                **transformer_options,
                "patches_replace": {**transformer_options.get("patches_replace", {}), "dit": replacements},
            }
        try:
            return executor(x, timestep, context, ref_latents, image_slots, transformer_options, **kwargs)
        finally:
            self.stream = None
            self.pristine = None

    def before_block(self, index, args):
        if self.active and index == self.layers[0]:
            self.pristine = args["img"].clone()

    def after_block(self, index, args, output):
        if not self.active:
            return output
        ordinal = self.layers.index(index)
        if ordinal == 0:
            self.control = self.control.to(output["img"].device, output["img"].dtype)
            tokens = self.control.flatten(2).transpose(1, 2)
            self.stream = self.patcher.model.init_stream(self.pristine, tokens, args["prefix_len"])
            self.pristine = None
        self.stream, skip = self.patcher.model.step(
            ordinal, self.stream, args["mod"], args["pe"], args["attn_fn"],
            args["prefix_len"], args["transformer_options"],
        )
        output["img"].add_(skip, alpha=self.strength)
        return output

    def to(self, device_or_dtype):
        if isinstance(device_or_dtype, torch.device):
            if self.control is not None:
                self.control = self.control.to(device_or_dtype)
            self.stream = None
        return self

    def cleanup(self):
        self.control = self.stream = self.pristine = None
        self.active = False
        return self

    def models(self):
        return [self.patcher]

    def register(self, model):
        model.add_wrapper(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, self.around_model)
        existing = model.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
        for index in self.layers:
            previous = existing.get(("single_block", index))
            model.set_model_patch_replace(_BlockPatch(self, index, previous), "dit", "single_block", index)


class _BlockPatch:
    def __init__(self, runtime, index, previous):
        self.runtime = runtime
        self.index = index
        self.previous = previous

    def __call__(self, args, extra_args):
        original = extra_args["original_block"]
        needed = ("mod", "attn_fn", "prefix_len")
        if all(name in args for name in needed):
            block_args = args
        else:
            # ComfyUI 0.37 captures these arguments in the original block closure.
            closure = inspect.getclosurevars(original).nonlocals
            if any(name not in closure for name in needed):
                raise RuntimeError("Unsupported ComfyUI Qwen Image 2.1 block interface.")
            block_args = {**args, **{name: closure[name] for name in needed}}
        with comfy.model_prefetch.pause_malloc_graph():
            self.runtime.before_block(self.index, block_args)
        output = original(args) if self.previous is None else self.previous(args, extra_args)
        with comfy.model_prefetch.pause_malloc_graph():
            return self.runtime.after_block(self.index, block_args, output)

    def to(self, device_or_dtype):
        self.runtime.to(device_or_dtype)
        if hasattr(self.previous, "to"):
            self.previous = self.previous.to(device_or_dtype)
        return self

    def cleanup(self):
        self.runtime.cleanup()
        if hasattr(self.previous, "cleanup"):
            self.previous.cleanup()
        return self

    def models(self):
        models = self.runtime.models()
        if hasattr(self.previous, "models"):
            models += self.previous.models()
        return models


class QwenImage21FunControlNetApply:
    @staticmethod
    def execute(model, model_patch, vae, strength, start_percent=0.0, end_percent=1.0,
                control_image=None, inpaint_image=None, mask=None):
        if strength == 0 or (control_image is None and mask is None):
            return (model,)
        patched = model.clone()
        num_base = len(model.get_model_object("diffusion_model").transformer_blocks)
        num_control = len(model_patch.model.control_blocks)
        if num_base % num_control:
            raise ValueError("UNION block count does not divide the Qwen Image 2.1 base model.")
        sampling = model.get_model_object("model_sampling")
        runtime = _ControlRuntime(
            model_patch, vae,
            control_image[..., :3] if control_image is not None else None,
            inpaint_image[..., :3] if inpaint_image is not None else None,
            mask, strength, list(range(0, num_base, num_base // num_control)),
            float(sampling.percent_to_sigma(start_percent)),
            float(sampling.percent_to_sigma(end_percent)),
        )
        runtime.register(patched)
        return (patched,)
