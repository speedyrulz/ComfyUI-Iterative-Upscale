import inspect
import logging

import comfy.samplers
import nodes
from nodes import MAX_RESOLUTION

from . import core
from . import hooks
from . import utils


class LatentPixelScale:
    upscale_methods = ["nearest-exact", "bilinear", "lanczos", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "samples": ("LATENT", ),
                     "scale_method": (s.upscale_methods,),
                     "scale_factor": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10000, "step": 0.05}),
                     "vae": ("VAE", ),
                     "use_tiled_vae": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                    }
                }

    RETURN_TYPES = ("LATENT", "IMAGE")
    FUNCTION = "doit"

    CATEGORY = "RulZ/Upscale"

    def doit(self, samples, scale_method, scale_factor, vae, use_tiled_vae, upscale_model_opt=None):
        if upscale_model_opt is None:
            latimg = core.latent_upscale_on_pixel_space2(samples, scale_method, scale_factor, vae, use_tile=use_tiled_vae)
        else:
            latimg = core.latent_upscale_on_pixel_space_with_model2(samples, scale_method, upscale_model_opt, scale_factor, vae, use_tile=use_tiled_vae)
        return latimg


class CfgScheduleHookProvider:
    schedules = ["simple"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "schedule_for_iteration": (s.schedules,),
                     "target_cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0}),
                    },
                }

    RETURN_TYPES = ("PK_HOOK",)
    FUNCTION = "doit"

    CATEGORY = "RulZ/Upscale"

    def doit(self, schedule_for_iteration, target_cfg):
        hook = None
        if schedule_for_iteration == "simple":
            hook = hooks.SimpleCfgScheduleHook(target_cfg)

        return (hook, )


class UnsamplerHookProvider:
    schedules = ["simple"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "steps": ("INT", {"default": 25, "min": 1, "max": 10000}),
                     "start_end_at_step": ("INT", {"default": 21, "min": 0, "max": 10000}),
                     "end_end_at_step": ("INT", {"default": 24, "min": 0, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                     "normalize": (["disable", "enable"], ),
                     "positive": ("CONDITIONING", ),
                     "negative": ("CONDITIONING", ),
                     "schedule_for_iteration": (s.schedules,),
                     }}

    RETURN_TYPES = ("PK_HOOK",)
    FUNCTION = "doit"

    CATEGORY = "RulZ/Upscale"

    def doit(self, model, steps, start_end_at_step, end_end_at_step, cfg, sampler_name,
             scheduler, normalize, positive, negative, schedule_for_iteration):
        # The Unsampler code is bundled (noise_nodes.py), so no extra node pack is needed.
        hook = None
        if schedule_for_iteration == "simple":
            hook = hooks.UnsamplerHook(model, steps, start_end_at_step, end_end_at_step, cfg, sampler_name,
                                       scheduler, normalize, positive, negative)

        return (hook, )


class NoiseInjectionHookProvider:
    schedules = ["simple"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "schedule_for_iteration": (s.schedules,),
                     "source": (["CPU", "GPU"],),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "start_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 200.0, "step": 0.01}),
                     "end_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 200.0, "step": 0.01}),
                    },
                }

    RETURN_TYPES = ("PK_HOOK",)
    FUNCTION = "doit"

    CATEGORY = "RulZ/Upscale"

    def doit(self, schedule_for_iteration, source, seed, start_strength, end_strength):
        hook = None
        if schedule_for_iteration == "simple":
            hook = hooks.InjectNoiseHook(source, seed, start_strength, end_strength)

        return (hook, )


class DenoiseScheduleHookProvider:
    schedules = ["simple"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "schedule_for_iteration": (s.schedules,),
                     "target_denoise": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                    },
                }

    RETURN_TYPES = ("PK_HOOK",)
    FUNCTION = "doit"

    CATEGORY = "RulZ/Upscale"

    def doit(self, schedule_for_iteration, target_denoise):
        hook = None
        if schedule_for_iteration == "simple":
            hook = hooks.SimpleDenoiseScheduleHook(target_denoise)

        return (hook, )


class StepsScheduleHookProvider:
    schedules = ["simple"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "schedule_for_iteration": (s.schedules,),
                     "target_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    },
                }

    RETURN_TYPES = ("PK_HOOK",)
    FUNCTION = "doit"

    CATEGORY = "RulZ/Upscale"

    def doit(self, schedule_for_iteration, target_steps):
        hook = None
        if schedule_for_iteration == "simple":
            hook = hooks.SimpleStepsScheduleHook(target_steps)

        return (hook, )


class PixelKSampleHookCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "hook1": ("PK_HOOK",),
                     "hook2": ("PK_HOOK",),
                    },
                }

    RETURN_TYPES = ("PK_HOOK",)
    FUNCTION = "doit"

    CATEGORY = "RulZ/Upscale"

    def doit(self, hook1, hook2):
        hook = hooks.PixelKSampleHookCombine(hook1, hook2)
        return (hook, )


class PixelTiledKSampleUpscalerProvider:
    upscale_methods = ["nearest-exact", "bilinear", "lanczos", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "scale_method": (s.upscale_methods,),
                    "model": ("MODEL",),
                    "vae": ("VAE",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "tile_width": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                    "tile_height": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                    "tiling_strategy": (["random", "padded", 'simple'], ),
                    },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                        "pk_hook_opt": ("PK_HOOK", ),
                        "tile_cnet_opt": ("CONTROL_NET", ),
                        "tile_cnet_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                    }
                }

    RETURN_TYPES = ("UPSCALER",)
    FUNCTION = "doit"

    CATEGORY = "RulZ/Upscale"

    def doit(self, scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, tile_width, tile_height, tiling_strategy, upscale_model_opt=None,
             pk_hook_opt=None, tile_cnet_opt=None, tile_cnet_strength=1.0, overlap=64):
        if "BNK_TiledKSampler" in nodes.NODE_CLASS_MAPPINGS:
            upscaler = core.PixelTiledKSampleUpscaler(scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise,
                                                      tile_width, tile_height, tiling_strategy, upscale_model_opt, pk_hook_opt, tile_cnet_opt,
                                                      tile_size=max(tile_width, tile_height), tile_cnet_strength=tile_cnet_strength, overlap=overlap)
            return (upscaler, )
        else:
            utils.try_install_custom_node('https://github.com/BlenderNeko/ComfyUI_TiledKSampler',
                                          "To use 'PixelTiledKSampleUpscalerProvider' node, 'BlenderNeko/ComfyUI_TiledKSampler' extension is required.")

            raise Exception("[ERROR] PixelTiledKSampleUpscalerProvider: ComfyUI_TiledKSampler custom node isn't installed. You must install BlenderNeko/ComfyUI_TiledKSampler extension to use this node.")


class PixelTiledKSampleUpscalerProviderPipe:
    upscale_methods = ["nearest-exact", "bilinear", "lanczos", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "scale_method": (s.upscale_methods,),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "tile_width": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                    "tile_height": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                    "tiling_strategy": (["random", "padded", 'simple'], ),
                    "basic_pipe": ("BASIC_PIPE",)
                    },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                        "pk_hook_opt": ("PK_HOOK", ),
                        "tile_cnet_opt": ("CONTROL_NET", ),
                        "tile_cnet_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    }
                }

    RETURN_TYPES = ("UPSCALER",)
    FUNCTION = "doit"

    CATEGORY = "RulZ/Upscale"

    def doit(self, scale_method, seed, steps, cfg, sampler_name, scheduler, denoise, tile_width, tile_height, tiling_strategy, basic_pipe, upscale_model_opt=None, pk_hook_opt=None,
             tile_cnet_opt=None, tile_cnet_strength=1.0):
        if "BNK_TiledKSampler" in nodes.NODE_CLASS_MAPPINGS:
            model, _, vae, positive, negative = basic_pipe
            upscaler = core.PixelTiledKSampleUpscaler(scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise,
                                                      tile_width, tile_height, tiling_strategy, upscale_model_opt, pk_hook_opt, tile_cnet_opt,
                                                      tile_size=max(tile_width, tile_height), tile_cnet_strength=tile_cnet_strength)
            return (upscaler, )
        else:
            logging.error("[Impact Pack] PixelTiledKSampleUpscalerProviderPipe: ComfyUI_TiledKSampler custom node isn't installed. You must install BlenderNeko/ComfyUI_TiledKSampler extension to use this node.")
            raise Exception("[Impact Pack] PixelTiledKSampleUpscalerProviderPipe: ComfyUI_TiledKSampler custom node isn't installed. You must install BlenderNeko/ComfyUI_TiledKSampler extension to use this node.")


class PixelKSampleUpscalerProvider:
    upscale_methods = ["nearest-exact", "bilinear", "lanczos", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "scale_method": (s.upscale_methods,),
                    "model": ("MODEL",),
                    "vae": ("VAE",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (core.get_schedulers(), ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "use_tiled_vae": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "tile_size": ("INT", {"default": 512, "min": 320, "max": 4096, "step": 64}),
                    },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                        "pk_hook_opt": ("PK_HOOK", ),
                        "scheduler_func_opt": ("SCHEDULER_FUNC",),
                    }
                }

    RETURN_TYPES = ("UPSCALER",)
    FUNCTION = "doit"

    CATEGORY = "RulZ/Upscale"

    def doit(self, scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise,
             use_tiled_vae, upscale_model_opt=None, pk_hook_opt=None, tile_size=512, scheduler_func_opt=None):
        upscaler = core.PixelKSampleUpscaler(scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler,
                                             positive, negative, denoise, use_tiled_vae, upscale_model_opt, pk_hook_opt,
                                             tile_size=tile_size, scheduler_func=scheduler_func_opt)
        return (upscaler, )


class PixelKSampleUpscalerProviderPipe(PixelKSampleUpscalerProvider):
    upscale_methods = ["nearest-exact", "bilinear", "lanczos", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "scale_method": (s.upscale_methods,),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (core.get_schedulers(), ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "use_tiled_vae": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "basic_pipe": ("BASIC_PIPE",),
                    "tile_size": ("INT", {"default": 512, "min": 320, "max": 4096, "step": 64}),
                    },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                        "pk_hook_opt": ("PK_HOOK", ),
                        "scheduler_func_opt": ("SCHEDULER_FUNC",),
                        "tile_cnet_opt": ("CONTROL_NET", ),
                        "tile_cnet_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    }
                }

    RETURN_TYPES = ("UPSCALER",)
    FUNCTION = "doit_pipe"

    CATEGORY = "RulZ/Upscale"

    def doit_pipe(self, scale_method, seed, steps, cfg, sampler_name, scheduler, denoise,
                  use_tiled_vae, basic_pipe, upscale_model_opt=None, pk_hook_opt=None,
                  tile_size=512, scheduler_func_opt=None, tile_cnet_opt=None, tile_cnet_strength=1.0):
        model, _, vae, positive, negative = basic_pipe
        upscaler = core.PixelKSampleUpscaler(scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler,
                                             positive, negative, denoise, use_tiled_vae, upscale_model_opt, pk_hook_opt,
                                             tile_size=tile_size, scheduler_func=scheduler_func_opt,
                                             tile_cnet_opt=tile_cnet_opt, tile_cnet_strength=tile_cnet_strength)
        return (upscaler, )


class TwoSamplersForMaskUpscalerProvider:
    upscale_methods = ["nearest-exact", "bilinear", "lanczos", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "scale_method": (s.upscale_methods,),
                     "full_sample_schedule": (
                         ["none", "interleave1", "interleave2", "interleave3",
                          "last1", "last2",
                          "interleave1+last1", "interleave2+last1", "interleave3+last1",
                          ],),
                     "use_tiled_vae": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                     "base_sampler": ("KSAMPLER", ),
                     "mask_sampler": ("KSAMPLER", ),
                     "mask": ("MASK", ),
                     "vae": ("VAE",),
                     "tile_size": ("INT", {"default": 512, "min": 320, "max": 4096, "step": 64}),
                     },
                "optional": {
                        "full_sampler_opt": ("KSAMPLER",),
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                        "pk_hook_base_opt": ("PK_HOOK", ),
                        "pk_hook_mask_opt": ("PK_HOOK", ),
                        "pk_hook_full_opt": ("PK_HOOK", ),
                    }
                }

    RETURN_TYPES = ("UPSCALER", )
    FUNCTION = "doit"

    CATEGORY = "RulZ/Upscale"

    def doit(self, scale_method, full_sample_schedule, use_tiled_vae, base_sampler, mask_sampler, mask, vae,
             full_sampler_opt=None, upscale_model_opt=None,
             pk_hook_base_opt=None, pk_hook_mask_opt=None, pk_hook_full_opt=None, tile_size=512):
        upscaler = core.TwoSamplersForMaskUpscaler(scale_method, full_sample_schedule, use_tiled_vae,
                                                   base_sampler, mask_sampler, mask, vae, full_sampler_opt, upscale_model_opt,
                                                   pk_hook_base_opt, pk_hook_mask_opt, pk_hook_full_opt, tile_size=tile_size)
        return (upscaler, )


class TwoSamplersForMaskUpscalerProviderPipe:
    upscale_methods = ["nearest-exact", "bilinear", "lanczos", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "scale_method": (s.upscale_methods,),
                     "full_sample_schedule": (
                         ["none", "interleave1", "interleave2", "interleave3",
                          "last1", "last2",
                          "interleave1+last1", "interleave2+last1", "interleave3+last1",
                          ],),
                     "use_tiled_vae": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                     "base_sampler": ("KSAMPLER", ),
                     "mask_sampler": ("KSAMPLER", ),
                     "mask": ("MASK", ),
                     "basic_pipe": ("BASIC_PIPE",),
                     "tile_size": ("INT", {"default": 512, "min": 320, "max": 4096, "step": 64}),
                     },
                "optional": {
                        "full_sampler_opt": ("KSAMPLER",),
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                        "pk_hook_base_opt": ("PK_HOOK", ),
                        "pk_hook_mask_opt": ("PK_HOOK", ),
                        "pk_hook_full_opt": ("PK_HOOK", ),
                    }
                }

    RETURN_TYPES = ("UPSCALER", )
    FUNCTION = "doit"

    CATEGORY = "RulZ/Upscale"

    def doit(self, scale_method, full_sample_schedule, use_tiled_vae, base_sampler, mask_sampler, mask, basic_pipe,
             full_sampler_opt=None, upscale_model_opt=None,
             pk_hook_base_opt=None, pk_hook_mask_opt=None, pk_hook_full_opt=None, tile_size=512):

        mask = utils.make_2d_mask(mask)

        _, _, vae, _, _ = basic_pipe
        upscaler = core.TwoSamplersForMaskUpscaler(scale_method, full_sample_schedule, use_tiled_vae,
                                                   base_sampler, mask_sampler, mask, vae, full_sampler_opt, upscale_model_opt,
                                                   pk_hook_base_opt, pk_hook_mask_opt, pk_hook_full_opt, tile_size=tile_size)
        return (upscaler, )


def vae_spatial_compression(vae, default=8):
    try:
        return int(vae.spacial_compression_decode())
    except Exception:
        return default


class IterativeLatentUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "samples": ("LATENT", ),
                     "upscale_factor": ("FLOAT", {"default": 1.5, "min": 1, "max": 10000, "step": 0.1}),
                     "steps": ("INT", {"default": 3, "min": 1, "max": 10000, "step": 1}),
                     "temp_prefix": ("STRING", {"default": ""}),
                     "upscaler": ("UPSCALER",),
                     "step_mode": (["simple", "geometric"], {"default": "simple"}),
                     "vae_compression": ("INT", {"default": 8, "min": 0, "max": 256, "step": 8})
                    },
                "hidden": {"unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("LATENT", "VAE")
    RETURN_NAMES = ("latent", "vae")
    FUNCTION = "doit"

    CATEGORY = "RulZ/Upscale"

    # dim_reduction_factor=8 for SD1/SDXL, used to calculate actual dims from latents based on VAE
    def doit(self, samples, upscale_factor, steps, temp_prefix, upscaler, step_mode="simple", vae_compression=8, unique_id=None):
        if vae_compression <= 0:
            # 0 (the widget minimum) used to make every target size 0x0, which ImageScale
            # ignores, so nothing was upscaled. Read the factor from the VAE instead.
            vae_compression = vae_spatial_compression(getattr(upscaler, 'vae', None))

        h, w = samples['samples'].shape[-2:]
        w, h = w * vae_compression, h * vae_compression

        def snap(size):
            # The VAE centre-crops pixels to multiples of its factor, cutting off the image
            # edges at every step; scaling straight to a multiple keeps the whole image.
            return max(vae_compression, int(size) // vae_compression * vae_compression)

        if temp_prefix == "":
            temp_prefix = None

        if step_mode == "geometric":
            upscale_factor_unit = pow(upscale_factor, 1.0/steps)
        else:  # simple
            upscale_factor_unit = max(0, (upscale_factor - 1.0) / steps)

        current_latent = samples
        noise_mask = current_latent.get('noise_mask')
        scale = 1

        try:
            for i in range(steps-1):
                if step_mode == "geometric":
                    scale *= upscale_factor_unit
                else:  # simple
                    scale += upscale_factor_unit

                new_w = snap(w*scale)
                new_h = snap(h*scale)
                core.update_node_status(unique_id, f"{i+1}/{steps} steps | x{scale:.2f}", (i+1)/steps)
                logging.info(f"IterativeLatentUpscale[{i+1}/{steps}]: {new_w:.1f}x{new_h:.1f} (scale:{scale:.2f}) ")
                step_info = i, steps
                current_latent = upscaler.upscale_shape(step_info, current_latent, new_w, new_h, temp_prefix)
                if noise_mask is not None:
                    current_latent['noise_mask'] = noise_mask

            # The final step always runs, so `steps` passes are made even at upscale_factor 1.
            new_w = snap(w*upscale_factor)
            new_h = snap(h*upscale_factor)
            core.update_node_status(unique_id, f"Final step | x{upscale_factor:.2f}", 1.0)
            logging.info(f"IterativeLatentUpscale[Final]: {new_w:.1f}x{new_h:.1f} (scale:{upscale_factor:.2f}) ")
            step_info = steps-1, steps
            current_latent = upscaler.upscale_shape(step_info, current_latent, new_w, new_h, temp_prefix)
        finally:
            core.update_node_status(unique_id, "", None)

        return current_latent, upscaler.vae


class IterativeImageUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "pixels": ("IMAGE", ),
                     "upscale_factor": ("FLOAT", {"default": 1.5, "min": 1, "max": 10000, "step": 0.1}),
                     "steps": ("INT", {"default": 3, "min": 1, "max": 10000, "step": 1}),
                     "temp_prefix": ("STRING", {"default": ""}),
                     "upscaler": ("UPSCALER",),
                     "vae": ("VAE",),
                     "step_mode": (["simple", "geometric"], {"default": "simple"}),
                     "vae_compression": ("INT", {"default": 8, "min": 0, "max": 256, "step": 8})
                    },
                "hidden": {"unique_id": "UNIQUE_ID"}
                }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "doit"

    CATEGORY = "RulZ/Upscale"

    def doit(self, pixels, upscale_factor, steps, temp_prefix, upscaler, vae, step_mode="simple", vae_compression=8, unique_id=None):
        if temp_prefix == "":
            temp_prefix = None

        # Tiled VAE for the first encode and final decode whenever the upscaler uses it per step
        # (the tiled sampler always does; the other upscalers when use_tiled_vae is on).
        use_tile = upscaler.is_tiled or getattr(upscaler, 'use_tiled_vae', False)
        overlap = core.clamp_overlap(upscaler.tile_size, getattr(upscaler, 'overlap', 64))

        try:
            core.update_node_status(unique_id, "VAEEncode (first)", 0)
            if use_tile:
                encoder = nodes.VAEEncodeTiled()
                if 'overlap' in inspect.signature(encoder.encode).parameters:
                    latent = encoder.encode(vae, pixels, upscaler.tile_size, overlap=overlap)[0]
                else:
                    latent = encoder.encode(vae, pixels, upscaler.tile_size)[0]
            else:
                latent = nodes.VAEEncode().encode(vae, pixels)[0]

            refined_latent = IterativeLatentUpscale().doit(latent, upscale_factor, steps, temp_prefix, upscaler, step_mode, vae_compression, unique_id)

            core.update_node_status(unique_id, "VAEDecode (final)", 1.0)
            if use_tile:
                decoder = nodes.VAEDecodeTiled()
                if 'overlap' in inspect.signature(decoder.decode).parameters:
                    pixels = decoder.decode(vae, refined_latent[0], upscaler.tile_size, overlap=overlap)[0]
                else:
                    pixels = decoder.decode(vae, refined_latent[0], upscaler.tile_size)[0]
            else:
                pixels = nodes.VAEDecode().decode(vae, refined_latent[0])[0]
        finally:
            core.update_node_status(unique_id, "", None)

        return (pixels, )
