"""
@title: Iterative Upscale
@nickname: Iterative Upscale
@description: The iterative upscaler nodes from ComfyUI Impact Pack, with only the provider, hook and pipe nodes they need.
"""

import logging

from .modules.impact.upscale_nodes import (
    LatentPixelScale,
    CfgScheduleHookProvider,
    UnsamplerHookProvider,
    NoiseInjectionHookProvider,
    DenoiseScheduleHookProvider,
    StepsScheduleHookProvider,
    PixelKSampleHookCombine,
    PixelTiledKSampleUpscalerProvider,
    PixelTiledKSampleUpscalerProviderPipe,
    PixelKSampleUpscalerProvider,
    PixelKSampleUpscalerProviderPipe,
    TwoSamplersForMaskUpscalerProvider,
    TwoSamplersForMaskUpscalerProviderPipe,
    IterativeLatentUpscale,
    IterativeImageUpscale,
)
from .modules.impact.sampler_nodes import KSamplerProvider, TiledKSamplerProvider, GITSSchedulerFuncProvider
from .modules.impact.pipe import ToBasicPipe, FromBasicPipe, FromBasicPipe_v2, EditBasicPipe

logging.info("### Loading: ComfyUI-Iterative-Upscale")

# Node IDs carry a RulZ prefix so this pack can be installed next to Impact Pack.
NODE_CLASS_MAPPINGS = {
    "RulZIterativeLatentUpscale": IterativeLatentUpscale,
    "RulZIterativeImageUpscale": IterativeImageUpscale,

    "RulZPixelKSampleUpscalerProvider": PixelKSampleUpscalerProvider,
    "RulZPixelKSampleUpscalerProviderPipe": PixelKSampleUpscalerProviderPipe,
    "RulZPixelTiledKSampleUpscalerProvider": PixelTiledKSampleUpscalerProvider,
    "RulZPixelTiledKSampleUpscalerProviderPipe": PixelTiledKSampleUpscalerProviderPipe,
    "RulZTwoSamplersForMaskUpscalerProvider": TwoSamplersForMaskUpscalerProvider,
    "RulZTwoSamplersForMaskUpscalerProviderPipe": TwoSamplersForMaskUpscalerProviderPipe,

    "RulZPixelKSampleHookCombine": PixelKSampleHookCombine,
    "RulZDenoiseScheduleHookProvider": DenoiseScheduleHookProvider,
    "RulZStepsScheduleHookProvider": StepsScheduleHookProvider,
    "RulZCfgScheduleHookProvider": CfgScheduleHookProvider,
    "RulZNoiseInjectionHookProvider": NoiseInjectionHookProvider,
    "RulZUnsamplerHookProvider": UnsamplerHookProvider,

    "RulZLatentPixelScale": LatentPixelScale,

    "RulZKSamplerProvider": KSamplerProvider,
    "RulZTiledKSamplerProvider": TiledKSamplerProvider,
    "RulZGITSSchedulerFuncProvider": GITSSchedulerFuncProvider,

    "RulZToBasicPipe": ToBasicPipe,
    "RulZFromBasicPipe": FromBasicPipe,
    "RulZFromBasicPipe_v2": FromBasicPipe_v2,
    "RulZEditBasicPipe": EditBasicPipe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RulZIterativeLatentUpscale": "RulZ Iterative Upscale (Latent/on Pixel Space)",
    "RulZIterativeImageUpscale": "RulZ Iterative Upscale (Image)",
    "RulZPixelKSampleUpscalerProvider": "RulZ PixelKSampleUpscalerProvider",
    "RulZPixelKSampleUpscalerProviderPipe": "RulZ PixelKSampleUpscalerProviderPipe",
    "RulZPixelTiledKSampleUpscalerProvider": "RulZ PixelTiledKSampleUpscalerProvider",
    "RulZPixelTiledKSampleUpscalerProviderPipe": "RulZ PixelTiledKSampleUpscalerProviderPipe",
    "RulZTwoSamplersForMaskUpscalerProvider": "RulZ TwoSamplersForMask Upscaler Provider",
    "RulZTwoSamplersForMaskUpscalerProviderPipe": "RulZ TwoSamplersForMask Upscaler Provider (pipe)",
    "RulZPixelKSampleHookCombine": "RulZ PixelKSampleHookCombine",
    "RulZDenoiseScheduleHookProvider": "RulZ DenoiseScheduleHookProvider",
    "RulZStepsScheduleHookProvider": "RulZ StepsScheduleHookProvider",
    "RulZCfgScheduleHookProvider": "RulZ CfgScheduleHookProvider",
    "RulZNoiseInjectionHookProvider": "RulZ NoiseInjectionHookProvider",
    "RulZUnsamplerHookProvider": "RulZ UnsamplerHookProvider",
    "RulZLatentPixelScale": "RulZ Latent Scale (on Pixel Space)",
    "RulZKSamplerProvider": "RulZ KSamplerProvider",
    "RulZTiledKSamplerProvider": "RulZ TiledKSamplerProvider",
    "RulZGITSSchedulerFuncProvider": "RulZ GITSScheduler Func Provider",
    "RulZToBasicPipe": "RulZ ToBasicPipe",
    "RulZFromBasicPipe": "RulZ FromBasicPipe",
    "RulZFromBasicPipe_v2": "RulZ FromBasicPipe_v2",
    "RulZEditBasicPipe": "RulZ Edit BasicPipe",
}

WEB_DIRECTORY = "js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
