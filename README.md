# ComfyUI-TCD

<p align="center">
<a href="./README.md">English</a> | <a href="./README.zh-CN.md">简体中文</a>
</p>

> If my work helps you, consider giving it a star. 

This repository is the [ComfyUI](https://github.com/comfyanonymous/ComfyUI) custom node implementation of TCD Sampler mentioned in the [TCD paper](https://github.com/jabir-zheng/TCD).

TCD, inspired by Consistency Models, is a novel distillation technology that enables the distillation of knowledge from pre-trained diffusion models into a few-step sampler. It is well known LCM has some problems in generating clear and detailed images. TCD significantly improves image quality than LCM, also with fewer steps. And TCD allows for adjusting the level of random noise in the samples, producing results with varying degrees of detail.

That is:
- TCD generates better details than LCM in same denoise steps.
- TCD can control the richness of details through parameters.
- In addition, TCD will also produce better results than LCM when steps are large.

Some of my other projects that may help you.
- [ComfyUI-TCD](https://github.com/JettHu/ComfyUI-TCD)
- [ComfyUI_TGate](https://github.com/JettHu/ComfyUI_TGate)
- [ComfyUI-ELLA](https://github.com/TencentQQGYLab/ComfyUI-ELLA)

## :star2: Changelog
- **[2024.6.03]** :wrench: sampling correctly when gamma is 0.
- **[2024.4.28]** :rocket: official [PR](https://github.com/comfyanonymous/ComfyUI/pull/3370) WIP.
- **[2024.4.28]** Initial repo.

## Example workflows

The [examples directory](./examples/) has workflow example. There are images generated with TCD and LCM in the [assets](./assets/) folder.

![tcd](./examples/tcd_example_workflow.png)

![tcd_with_low_NFEs](./examples/tcd_with_low_NFEs.png)

TCD maintains superior generative quality at high NFEs (steps).

![tcd_with_high_NFEs](./examples/tcd_with_high_NFEs.png)


| | TCD result | LCM result |
| ---: | :---: | :---: |
| low NFEs | ![](./assets/tcd_step4.png) | ![](./assets/lcm_ste4.png) |
| high NFEs | ![](./assets/tcd_step30.png) | ![](./assets/lcm_step30.png) |

> [!NOTE]  
> Except for `cfg`, `step`, and `sampler`, other parameters remain the same.

| ddim 30 step | dpmpp_2m 30 step | TCD 4 step | LCM 4 step |
| :---: | :---: | :---: | :---: |
| <img src="./assets/ori_ddim_step30.png" width="150"/> | <img src="./assets/ori_dpmpp_2m_step30.png" width="150"/> | <img src="./assets/tcd_step4.png" width="150"/> | <img src="./assets/lcm_ste4.png" width="150"/> |

The above comparison results show that TCD can produce better results than LCM details, no longer so blurry and bland.

## LoRAs

Some LoRAs available on TCD.

[TCD LoRAs](https://mhh0318.github.io/tcd/) from TCD Team.
- [TCD-SD15-LoRA.safetensors](https://huggingface.co/h1t/TCD-SD15-LoRA/blob/main/pytorch_lora_weights.safetensors)
- [TCD-SDXL-LoRA.safetensors](https://huggingface.co/h1t/TCD-SDXL-LoRA/blob/main/pytorch_lora_weights.safetensors)

[Hyper-SD 1-Step Unified LoRAs](https://hyper-sd.github.io/) from ByteDance Team.
- [Hyper-SD15-1step-lora.safetensors](https://huggingface.co/ByteDance/Hyper-SD/blob/main/Hyper-SD15-1step-lora.safetensors)
- [Hyper-SDXL-1step-lora.safetensors](https://huggingface.co/ByteDance/Hyper-SD/blob/main/Hyper-SDXL-1step-lora.safetensors)

## INSTALL
```bash
git clone https://github.com/JettHu/ComfyUI-TCD

# Or use ComfyUI-Manager
```

## Nodes reference

### TCD Model Sampling Discrete

#### Inputs
- **model**, model loaded by `Load Checkpoint` and other MODEL loaders.

#### Configuration parameters
- **steps**: The number of steps to use during denoising (same as KSampler node).
- **scheduler**: The type of schedule to use (same as KSampler node).
  - I only kept `simple` and `sgm_uniform`. `simple` behaves the same as diffusers. And `sgm_uniform` is another scheduler recommended by comfyui author using lcm.
- **denoise**: How much information of the latents should be erased by noise (same as KSampler node).
- **eta**: A stochastic parameter (referred to as `gamma` in the paper) used to control the stochasticity (detail-richness of the results) in every step. When eta = 0, it represents deterministic sampling, whereas eta = 1 indicates full stochastic sampling. It is recommended to fine-tune this parameter when adjusting steps larger and using different LoRAs. The default is 0.3.
