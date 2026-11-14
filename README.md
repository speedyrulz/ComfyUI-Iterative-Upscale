# ComfyUI-Iterative-Upscale

**Custom node pack for ComfyUI**

The Iterative Upscale nodes from [ComfyUI Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack),
cut down to those nodes and the provider, hook and pipe nodes needed to use them. The detector,
detailer, SEGS, wildcard, SAM, bridge and logic nodes of Impact Pack are not included.

Every node ID has a `RulZ` prefix (for example `RulZIterativeLatentUpscale`, shown as
"RulZ Iterative Upscale (Latent/on Pixel Space)"), and the nodes sit under `RulZ/` in the node
menu, so this pack can be installed next to Impact Pack without clashing. Inputs and widgets are
the same as in Impact Pack. Workflows saved with Impact Pack's versions of these nodes still use
Impact Pack's IDs: they need Impact Pack, or the nodes swapped for the RulZ versions.

In the node list below, add `RulZ` to the front of each name to find the node.

## How To Install

Clone this repository into `ComfyUI/custom_nodes` and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/speedyrulz/ComfyUI-Iterative-Upscale
```

No extra Python packages are needed beyond ComfyUI itself.

Optional node packs, only for the nodes that name them:

* [BlenderNeko/ComfyUI_TiledKSampler](https://github.com/BlenderNeko/ComfyUI_TiledKSampler) for
  `PixelTiledKSampleUpscalerProvider`, `PixelTiledKSampleUpscalerProviderPipe` and `TiledKSamplerProvider`.
* [BlenderNeko/ComfyUI_Noise](https://github.com/BlenderNeko/ComfyUI_Noise), optional for `NoiseInjectionHookProvider`:
  when installed it makes the noise for 4-channel latents (SD1.5, SDXL), so results match Impact Pack.
  Without it, and for other latents (Flux, SD3, Qwen-Image, ...), the hook makes its own seeded noise.
* [Fannovel16/comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) (`TilePreprocessor`)
  when a tile ControlNet is connected to `tile_cnet_opt`.

## Nodes

### Iterative Upscale nodes
* `Iterative Upscale (Latent/on Pixel Space)` - Splits `upscale_factor` into `steps` and upscales
  step by step with the connected `UPSCALER`. Latent in, latent out.
  * `step_mode`: `simple` grows the scale by the same amount each step; `geometric` by the same ratio.
  * `vae_compression`: the VAE's spatial downscale factor (8 for SD1/SDXL), used to work out pixel sizes.
* `Iterative Upscale (Image)` - The same, image in and image out. Internally it uses the latent node.

### Upscaler providers (`UPSCALER`)
* `PixelKSampleUpscalerProvider` - At each step: VAE decode, upscale in pixel space, VAE encode, then k-sample.
  * If `upscale_model_opt` is connected, the image is upscaled with the model and then scaled down
    to the target size with `scale_method`. An x2 model can cut the number of steps needed.
* `PixelKSampleUpscalerProviderPipe` - The same, taking model, VAE and conditioning from a `BASIC_PIPE`.
  It also accepts a tile ControlNet.
* `PixelTiledKSampleUpscalerProvider` / `...Pipe` - Like `PixelKSampleUpscalerProvider`, but uses
  ComfyUI_TiledKSampler and tiled VAE decode/encode to avoid running out of VRAM at high resolutions.
* `TwoSamplersForMaskUpscalerProvider` / `...Pipe` - Applies `base_sampler` where the mask is 0 and
  `mask_sampler` where it is 1. `full_sample_schedule` picks the steps where the whole image is
  sampled instead.

### Samplers for `TwoSamplersForMaskUpscalerProvider` (`KSAMPLER`)
* `KSamplerProvider` - Wraps a KSampler as `KSAMPLER`.
* `TiledKSamplerProvider` - Wraps ComfyUI_TiledKSampler as `KSAMPLER`.
* `GITSSchedulerFuncProvider` - Supplies a GITS noise schedule (`SCHEDULER_FUNC`) for
  `KSamplerProvider` and `PixelKSampleUpscalerProvider`.

### PK_HOOK
* `DenoiseScheduleHookProvider` - Moves denoise towards `target_denoise` as the steps progress.
* `CfgScheduleHookProvider` - Moves cfg towards `target_cfg` as the steps progress.
* `StepsScheduleHookProvider` - Moves the sampling steps towards `target_steps` as the steps progress.
* `NoiseInjectionHookProvider` - Injects noise into the latent at each step, from `start_strength`
  on the first step to `end_strength` on the last. Works with any latent type.
* `UnsamplerHookProvider` - Applies the Unsampler at each step (bundled, no extra pack needed).
* `PixelKSampleHookCombine` - Chains two PK_HOOKs; hook1 runs first, then hook2.

### Other
* `Latent Scale (on Pixel Space)` - One-off version of the upscale: VAE decode, upscale (optionally
  with an upscale model), VAE encode.
* `ToBasicPipe`, `FromBasicPipe`, `FromBasicPipe_v2` - Bundle model, clip, vae, positive and negative
  into a `BASIC_PIPE` for the pipe versions of the providers, or take them apart again.
* `EditBasicPipe` - Replaces some elements of a `BASIC_PIPE` (for example the model or prompts).

## Tutorials (Impact Pack)

* [Extreme Highresolution Upscale](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/extreme-upscale.md)
* [Advanced Iterative Upscale: PK_HOOK](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/pk_hook.md)
* [Advanced Iterative Upscale: TwoSamplersForMask Upscale Provider](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/TwoSamplersUpscale.md)

![upscale-workflow-example](https://github.com/ltdrdata/ComfyUI-extension-tutorials/raw/Main/ComfyUI-Impact-Pack/images/upscale-workflow.png)

## Credits

ltdrdata/[ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) - The original pack these nodes come from (GPL-3.0).

ComfyUI/[ComfyUI](https://github.com/comfyanonymous/ComfyUI) - A powerful and modular stable diffusion GUI.

BlenderNeko/[ComfyUI-TiledKSampler](https://github.com/BlenderNeko/ComfyUI_TiledKSampler) - Tiled sampling at high resolutions with low VRAM.

BlenderNeko/[ComfyUI_Noise](https://github.com/BlenderNeko/ComfyUI_Noise) - Noise injection and the bundled Unsampler code.
