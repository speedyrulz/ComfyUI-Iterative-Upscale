"""Widget names and order must never change (AGENTS.md): saved workflows map
`widgets_values` to inputs by position."""

EXPECTED = {
    "RulZIterativeLatentUpscale": (
        ["samples", "upscale_factor", "steps", "temp_prefix", "upscaler", "step_mode", "vae_compression"],
        [],
    ),
    "RulZIterativeImageUpscale": (
        ["pixels", "upscale_factor", "steps", "temp_prefix", "upscaler", "vae", "step_mode", "vae_compression"],
        [],
    ),
    "RulZPixelKSampleUpscalerProvider": (
        ["scale_method", "model", "vae", "seed", "steps", "cfg", "sampler_name", "scheduler", "positive", "negative", "denoise", "use_tiled_vae", "tile_size"],
        ["upscale_model_opt", "pk_hook_opt", "scheduler_func_opt"],
    ),
    "RulZPixelKSampleUpscalerProviderPipe": (
        ["scale_method", "seed", "steps", "cfg", "sampler_name", "scheduler", "denoise", "use_tiled_vae", "basic_pipe", "tile_size"],
        ["upscale_model_opt", "pk_hook_opt", "scheduler_func_opt", "tile_cnet_opt", "tile_cnet_strength"],
    ),
    "RulZPixelTiledKSampleUpscalerProvider": (
        ["scale_method", "model", "vae", "seed", "steps", "cfg", "sampler_name", "scheduler", "positive", "negative", "denoise", "tile_width", "tile_height", "tiling_strategy"],
        ["upscale_model_opt", "pk_hook_opt", "tile_cnet_opt", "tile_cnet_strength", "overlap"],
    ),
    "RulZPixelTiledKSampleUpscalerProviderPipe": (
        ["scale_method", "seed", "steps", "cfg", "sampler_name", "scheduler", "denoise", "tile_width", "tile_height", "tiling_strategy", "basic_pipe"],
        ["upscale_model_opt", "pk_hook_opt", "tile_cnet_opt", "tile_cnet_strength"],
    ),
    "RulZTwoSamplersForMaskUpscalerProvider": (
        ["scale_method", "full_sample_schedule", "use_tiled_vae", "base_sampler", "mask_sampler", "mask", "vae", "tile_size"],
        ["full_sampler_opt", "upscale_model_opt", "pk_hook_base_opt", "pk_hook_mask_opt", "pk_hook_full_opt"],
    ),
    "RulZTwoSamplersForMaskUpscalerProviderPipe": (
        ["scale_method", "full_sample_schedule", "use_tiled_vae", "base_sampler", "mask_sampler", "mask", "basic_pipe", "tile_size"],
        ["full_sampler_opt", "upscale_model_opt", "pk_hook_base_opt", "pk_hook_mask_opt", "pk_hook_full_opt"],
    ),
    "RulZPixelKSampleHookCombine": (
        ["hook1", "hook2"],
        [],
    ),
    "RulZDenoiseScheduleHookProvider": (
        ["schedule_for_iteration", "target_denoise"],
        [],
    ),
    "RulZStepsScheduleHookProvider": (
        ["schedule_for_iteration", "target_steps"],
        [],
    ),
    "RulZCfgScheduleHookProvider": (
        ["schedule_for_iteration", "target_cfg"],
        [],
    ),
    "RulZNoiseInjectionHookProvider": (
        ["schedule_for_iteration", "source", "seed", "start_strength", "end_strength"],
        [],
    ),
    "RulZUnsamplerHookProvider": (
        ["model", "steps", "start_end_at_step", "end_end_at_step", "cfg", "sampler_name", "scheduler", "normalize", "positive", "negative", "schedule_for_iteration"],
        [],
    ),
    "RulZLatentPixelScale": (
        ["samples", "scale_method", "scale_factor", "vae", "use_tiled_vae"],
        ["upscale_model_opt"],
    ),
    "RulZKSamplerProvider": (
        ["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise", "basic_pipe"],
        ["scheduler_func_opt"],
    ),
    "RulZTiledKSamplerProvider": (
        ["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise", "tile_width", "tile_height", "tiling_strategy", "basic_pipe"],
        [],
    ),
    "RulZGITSSchedulerFuncProvider": (
        ["coeff", "denoise"],
        [],
    ),
    "RulZToBasicPipe": (
        ["model", "clip", "vae", "positive", "negative"],
        [],
    ),
    "RulZFromBasicPipe": (
        ["basic_pipe"],
        [],
    ),
    "RulZFromBasicPipe_v2": (
        ["basic_pipe"],
        [],
    ),
    "RulZEditBasicPipe": (
        ["basic_pipe"],
        ["model", "clip", "vae", "positive", "negative"],
    ),
}


def test_node_ids_unchanged(pack):
    assert list(pack.NODE_CLASS_MAPPINGS) == list(EXPECTED)
    assert set(pack.NODE_DISPLAY_NAME_MAPPINGS) == set(EXPECTED)
    assert all(name.startswith("RulZ ") for name in pack.NODE_DISPLAY_NAME_MAPPINGS.values())


def test_widget_names_and_order_unchanged(pack):
    for node_id, (required, optional) in EXPECTED.items():
        inputs = pack.NODE_CLASS_MAPPINGS[node_id].INPUT_TYPES()
        assert list(inputs.get("required", {})) == required, node_id
        assert list(inputs.get("optional", {})) == optional, node_id
