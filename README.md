# ComfyUI-TCD

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) implementation for [TCD](https://github.com/jabir-zheng/TCD).

## Example workflows

The [examples directory](./examples/) has workflow example. There are images generated with TCD and LCM in the [assets](./assets/) folder.

![tcd_with_low_NFEs](./examples/tcd_with_low_NFEs.png)

TCD maintains superior generative quality at high NFEs (steps).

![tcd_with_high_NFEs](./examples/tcd_with_high_NFEs.png)


| | TCD result | LCM result |
| ---: | :---: | :---: |
| low NFEs | ![](./assets/tcd_step4.png) | ![](./assets/lcm_ste4.png) |
| high NFEs | ![](./assets/tcd_step30.png) | ![](./assets/lcm_step30.png) |

## INSTALL
```bash
git clone https://github.com/JettHu/ComfyUI-TCD
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
- **eta**: A stochastic parameter (referred to as `gamma` in the paper) used to control the stochasticity in every step. When eta = 0, it represents deterministic sampling, whereas eta = 1 indicates full stochastic sampling. 
