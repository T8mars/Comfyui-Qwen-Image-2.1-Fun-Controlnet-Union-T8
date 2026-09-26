# Qwen Image 2.1 Fun ControlNet Union · ComfyUI 节点

[English](README.md) | **简体中文**

基于 [Alibaba PAI Qwen Image 2.1 Fun ControlNet Union](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union) 的 ComfyUI 原生管道节点。一个 Union 权重支持 Canny、Depth、Grayscale、HED、Lineart、MLSD、Pose、Scribble 八种控制图，以及掩码局部重绘。

![Canny 控制效果](assets/canny_result.png)

## 依赖与模型位置

- ComfyUI 0.37.0 或更新版本，需支持 Qwen Image 2.1 基础模型。本插件已内置 Union 兼容实现，无需等待[上游 PR #16519](https://github.com/Comfy-Org/ComfyUI/pull/16519) 合入。
- [Comfy-Org Qwen Image 2.1 基础模型](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)。
- [转换后的 Union 权重](https://huggingface.co/t8star/Qwen-Image-2.1-Fun-Controlnet-Union-Comfy)。

将文件放入 **ComfyUI** 的对应目录：

| 文件 | 目录 |
| --- | --- |
| `Qwen-Image-2.1-Fun-Controlnet-Union-ComfyUI.safetensors` | `models/controlnet/` |
| `qwen_image_2.1_int8_convrot.safetensors` | `models/diffusion_models/` |
| `qwen3vl_8b_int8_convrot.safetensors` | `models/text_encoders/` |
| `qwen_image_2.1_vae_bf16.safetensors` | `models/vae/` |

## 安装与使用

在 ComfyUI Manager 中安装 **Qwen Image 2.1 Fun ControlNet Union (T8)**，或将本仓库克隆到 `ComfyUI/custom_nodes/`，然后重启 ComfyUI。

将 [`assets/`](assets) 的示例文件复制到 `ComfyUI/input/`，再把 [`workflows/`](workflows) 中的画布工作流拖入 ComfyUI。包含八种控制图、Pose 局部重绘、[参考图编辑](workflows/qwen21_union_pose_reference_edit.json)和[参考图加掩码局部重绘](workflows/qwen21_union_pose_reference_inpaint.json)。`.api.json` 文件供 API 调用，不是画布工作流。

**Apply Qwen Image 2.1 UNION** 需要已预处理的控制图；模式下拉框只标识输入类型，不会自动从照片提取边缘、深度或姿态。图像编辑时，将原图接入 **Qwen 2.1 UNION Image Reference Encode** 的 `reference_image`，再把其 `positive`、`negative`、`latent` 三个输出接到采样器。输出画布尺寸按参考图确定。指定局部区域时，还要把原图接到 `inpaint_image`，并将掩码接到 `mask`；**白色区域引导重绘，黑色区域引导保留**。该掩码仅作为模型条件，不能保证遮罩外的像素完全不变。

| 输入 | 应连接的图片 |
| --- | --- |
| `control_image` | 已预处理的边缘、深度、姿态等控制图 |
| `reference_image` | 要编辑的原图或视觉参考图 |
| `inpaint_image` + `mask` | 局部重绘原图与白色为重绘区域的引导掩码 |

各条件输入（控制图、重绘原图或掩码）均应为单张；原生控制管道对批次只使用首张。若要用同一控制图生成多个随机种子，可在宽高比 latent 节点后重复 latent。极端宽高比若使输出任一边超过 4096 像素，节点会提示降低分辨率或裁剪、填充输入图。

加载器会校验 16 个控制块。上图 Canny 示例采用默认 40 步，输出 800×1312。

## 许可与来源

原有适配节点代码采用 [MIT 许可](LICENSE)；内置的 ComfyUI Union 兼容实现基于[上游 PR #16519](https://github.com/Comfy-Org/ComfyUI/pull/16519)，采用 [GPL-3.0](LICENSE_COMFYUI_GPL-3.0.txt)，本发行包整体按 GPL-3.0 发布。模型权重遵循 [Qwen Research License](MODEL_LICENSE.txt)，**仅限非商业用途**；商业用途需向 Qwen 单独取得许可。转换仅修改 safetensors 头部元数据，张量数据字节保持不变。校验值见 [`MODEL_SHA256.json`](MODEL_SHA256.json)，转换脚本见 [`convert_union.py`](convert_union.py)。

## T8 链接

[B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [免费画廊](https://www.openzhenzhen.com) · [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
