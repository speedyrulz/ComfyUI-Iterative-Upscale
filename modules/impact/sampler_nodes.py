import comfy
import comfy.samplers
import nodes
from nodes import MAX_RESOLUTION

from . import core
from .impact_sampling import KSamplerWrapper


class TiledKSamplerProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed to use for generating CPU noise for sampling."}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "total sampling steps"}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "tooltip": "classifier free guidance value"}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "sampler"}),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "noise schedule"}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of noise to remove. This amount is the noise added at the start, and the higher it is, the more the input latent will be modified before being returned."}),
                    "tile_width": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64, "tooltip": "Sets the width of the tile to be used in TiledKSampler."}),
                    "tile_height": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64, "tooltip": "Sets the height of the tile to be used in TiledKSampler."}),
                    "tiling_strategy": (["random", "padded", 'simple'], {"tooltip": "Sets the tiling strategy for TiledKSampler."} ),
                    "basic_pipe": ("BASIC_PIPE", {"tooltip": "basic_pipe input for sampling"})
                    }}

    OUTPUT_TOOLTIPS = ("sampler wrapper. (Can be used when generating a regional_prompt.)", )

    RETURN_TYPES = ("KSAMPLER",)
    FUNCTION = "doit"

    CATEGORY = "RulZ/Sampler"

    @staticmethod
    def doit(seed, steps, cfg, sampler_name, scheduler, denoise,
             tile_width, tile_height, tiling_strategy, basic_pipe):
        model, _, _, positive, negative = basic_pipe
        sampler = core.TiledKSamplerWrapper(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise,
                                            tile_width, tile_height, tiling_strategy)
        return (sampler, )


class KSamplerProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed to use for generating CPU noise for sampling."}),
                                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "total sampling steps"}),
                                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "tooltip": "classifier free guidance value"}),
                                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "sampler"}),
                                "scheduler": (core.get_schedulers(), {"tooltip": "noise schedule"}),
                                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of noise to remove. This amount is the noise added at the start, and the higher it is, the more the input latent will be modified before being returned."}),
                                "basic_pipe": ("BASIC_PIPE", {"tooltip": "basic_pipe input for sampling"})
                             },
                "optional": {
                    "scheduler_func_opt": ("SCHEDULER_FUNC", {"tooltip": "[OPTIONAL] Noise schedule generation function. If this is set, the scheduler widget will be ignored."}),
                    }
                }

    OUTPUT_TOOLTIPS = ("sampler wrapper. (Can be used when generating a regional_prompt.)",)

    RETURN_TYPES = ("KSAMPLER",)
    FUNCTION = "doit"

    CATEGORY = "RulZ/Sampler"

    @staticmethod
    def doit(seed, steps, cfg, sampler_name, scheduler, denoise, basic_pipe, scheduler_func_opt=None):
        model, _, _, positive, negative = basic_pipe
        sampler = KSamplerWrapper(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, scheduler_func=scheduler_func_opt)
        return (sampler, )


class GITSSchedulerFuncProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "coeff": ("FLOAT", {"default": 1.20, "min": 0.80, "max": 1.50, "step": 0.05, "tooltip": "coeff factor of GITS Scheduler"}),
                        "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "denoise amount for noise schedule"}),
                    }
                }

    OUTPUT_TOOLTIPS = ("Returns a function that generates a noise schedule using GITSScheduler. This can be used in place of a predetermined noise schedule to dynamically generate a noise schedule based on the steps.",)

    RETURN_TYPES = ("SCHEDULER_FUNC",)
    CATEGORY = "RulZ/sampling"

    FUNCTION = "doit"

    @staticmethod
    def doit(coeff, denoise):
        def f(model, sampler, steps):
            if 'GITSScheduler' not in nodes.NODE_CLASS_MAPPINGS:
                raise Exception("[Impact Pack] ComfyUI is an outdated version. Cannot use GITSScheduler.")

            scheduler = nodes.NODE_CLASS_MAPPINGS['GITSScheduler']()
            return scheduler.get_sigmas(coeff, steps, denoise)[0]

        return (f, )
