"""Regression tests for the 2026-11 audit fixes. Each test failed on the code before the fix."""

import sys

import pytest
import torch


# ---------------------------------------------------------------- helpers

def latent(b=1, c=4, h=8, w=8, fill=0.0):
    return {"samples": torch.full((b, c, h, w), fill)}


class RecordingSampler:
    """A KSAMPLER stand-in that records the latent (and its noise_mask) it is asked to sample."""

    def __init__(self):
        self.calls = []

    def sample(self, latent_image, hook=None):
        mask = latent_image.get("noise_mask")
        self.calls.append({"mask": None if mask is None else mask.clone(), "hook": hook})
        return latent_image


def fake_pixel_space(calls):
    """Replacement for core.latent_upscale_on_pixel_space_shape2: records its keyword args."""

    def fn(samples, scale_method, w, h, vae, *args, **kwargs):
        calls.append(kwargs)
        return latent(h=int(h) // 8, w=int(w) // 8), torch.zeros(1, int(h), int(w), 3)

    return fn


# ---------------------------------------------------------------- B1: TwoSamplers mask layout

def test_two_samplers_mask_keeps_spatial_layout(impact):
    # Left half of the mask is 1 (mask sampler), right half 0 (base sampler).
    mask = torch.zeros(1, 64, 64)
    mask[:, :, :32] = 1.0
    base, masked = RecordingSampler(), RecordingSampler()
    upscaler = impact.core.TwoSamplersForMaskUpscaler("bilinear", "none", False, base, masked, mask, vae=None)

    upscaler.do_samples((0, 3), base, masked, "none", upscaler.params[5], latent(h=16, w=16))

    mask_used = masked.calls[0]["mask"]
    inv_used = base.calls[0]["mask"]
    assert mask_used.shape == (1, 1, 16, 16)
    assert inv_used.shape == (1, 1, 16, 16)
    # Columns on the left are masked, columns on the right are not, on every row.
    assert torch.all(mask_used[..., :, :6] > 0.99)
    assert torch.all(mask_used[..., :, 10:] < 0.01)
    assert torch.equal(inv_used, (mask_used != 1.0).float())


# ---------------------------------------------------------------- B2: denoise 0

def test_impact_sample_denoise_zero_returns_latent_unchanged(impact):
    src = latent(fill=0.5)
    out = impact.impact_sampling.impact_sample(None, 0, 20, 7.0, "euler", "normal", [], [], src, denoise=0.0)
    assert out is src


# ---------------------------------------------------------------- B3: GITS scheduler func on V3 GITSScheduler

def test_gits_scheduler_func_works_with_v3_node(impact, node_mappings):
    seen = {}

    class GITSScheduler:  # ComfyUI 0.45: V3 node, only a classmethod execute returning NodeOutput
        @classmethod
        def execute(cls, coeff, steps, denoise):
            seen.update(coeff=coeff, steps=steps, denoise=denoise)
            return (torch.tensor([1.0, 0.5, 0.0]),)

    node_mappings["GITSScheduler"] = GITSScheduler
    (fn,) = impact.sampler_nodes.GITSSchedulerFuncProvider.doit(1.2, 0.8)
    sigmas = fn(None, "euler", 2)
    assert torch.equal(sigmas, torch.tensor([1.0, 0.5, 0.0]))
    assert seen == {"coeff": 1.2, "steps": 2, "denoise": 0.8}


# ---------------------------------------------------------------- B4: LTXV schedule length

def test_ltxv_scheduler_uses_requested_steps(impact, node_mappings):
    class LTXVScheduler:
        @classmethod
        def execute(cls, steps, max_shift, base_shift, stretch, terminal, latent=None):
            return (torch.linspace(1.0, 0.0, steps + 1),)

    node_mappings["LTXVScheduler"] = LTXVScheduler
    sigmas = impact.impact_sampling.calculate_sigmas(None, "euler", "LTXV[default]", 40)
    assert len(sigmas) == 41


# ---------------------------------------------------------------- B5 + B10: tiled upscaler hooks and overlap

def test_tiled_upscaler_applies_pk_hook_and_overlap(impact, node_mappings, monkeypatch):
    sampled = []

    class BNK_TiledKSampler:
        def sample(self, model, seed, tile_width, tile_height, tiling_strategy, steps, cfg, sampler_name,
                   scheduler, positive, negative, latent_image, denoise):
            sampled.append({"steps": steps, "cfg": cfg, "denoise": denoise})
            return (latent_image,)

    node_mappings["BNK_TiledKSampler"] = BNK_TiledKSampler
    vae_calls = []
    monkeypatch.setattr(impact.core, "latent_upscale_on_pixel_space_shape2", fake_pixel_space(vae_calls))

    hook = impact.hooks.SimpleDenoiseScheduleHook(0.2)
    upscaler = impact.core.PixelTiledKSampleUpscaler("bilinear", "model", "vae", 0, 20, 7.0, "euler", "normal",
                                                     [], [], 0.6, 512, 512, "random", hook_opt=hook, overlap=96)
    upscaler.upscale_shape((0, 3), latent(), 128, 128)
    upscaler.upscale_shape((2, 3), latent(), 192, 192)

    assert [s["denoise"] for s in sampled] == pytest.approx([0.6, 0.2])
    assert all(call["overlap"] == 96 for call in vae_calls)


# ---------------------------------------------------------------- B6: full-sample schedules

@pytest.mark.parametrize("schedule, expected", [
    ("none", [False, False, False, False, False, False]),
    ("interleave1", [False, True, False, True, False, True]),
    ("interleave2", [False, False, True, False, False, True]),
    ("interleave3", [False, False, False, True, False, False]),
    ("last1", [False, False, False, False, False, True]),
    ("last2", [False, False, False, False, True, True]),
    ("interleave1+last1", [False, True, False, True, False, True]),
    ("interleave2+last1", [False, False, True, False, False, True]),
    ("interleave3+last1", [False, False, False, True, False, True]),
])
def test_full_sample_schedule(impact, schedule, expected):
    upscaler = impact.core.TwoSamplersForMaskUpscaler("bilinear", schedule, False, None, None, torch.zeros(8, 8), None)
    steps = 6
    got = [upscaler.is_full_sample_time((i, steps), schedule) for i in range(steps)]
    assert got == expected


# ---------------------------------------------------------------- B7: hook latent forwarded

def test_sampler_wrappers_use_latent_returned_by_hook(impact, node_mappings, monkeypatch):
    swapped = latent(fill=9.0)

    class SwapHook(impact.hooks.PixelKSampleHook):
        def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise):
            return model, seed, steps, cfg, sampler_name, scheduler, positive, negative, swapped, denoise

    got = []
    monkeypatch.setattr(impact.impact_sampling, "impact_sample", lambda *a, **k: got.append(a[8]) or a[8])
    impact.impact_sampling.KSamplerWrapper("m", 0, 20, 7.0, "euler", "normal", [], [], 0.5).sample(latent(), SwapHook())
    assert len(got) == 1 and got[0] is swapped

    class BNK_TiledKSampler:
        def sample(self, *args):
            got.append(args[11])
            return (args[11],)

    node_mappings["BNK_TiledKSampler"] = BNK_TiledKSampler
    impact.core.TiledKSamplerWrapper("m", 0, 20, 7.0, "euler", "normal", [], [], 0.5, 512, 512, "random").sample(latent(), SwapHook())
    assert got[-1] is swapped


# ---------------------------------------------------------------- B8: TwoSamplers VAE hook does not depend on the upscale model

def test_two_samplers_uses_base_hook_for_vae_with_and_without_model(impact, monkeypatch):
    seen = []
    monkeypatch.setattr(impact.core, "latent_upscale_on_pixel_space_shape",
                        lambda *a, **k: seen.append(("plain", k["hook"])) or latent(h=16, w=16))
    monkeypatch.setattr(impact.core, "latent_upscale_on_pixel_space_with_model_shape",
                        lambda *a, **k: seen.append(("model", k["hook"])) or latent(h=16, w=16))
    base_hook, mask_hook = impact.hooks.PixelKSampleHook(), impact.hooks.PixelKSampleHook()
    sampler = RecordingSampler()
    for model in (None, "upscale-model"):
        upscaler = impact.core.TwoSamplersForMaskUpscaler("bilinear", "none", False, sampler, sampler, torch.zeros(8, 8),
                                                          None, upscale_model_opt=model,
                                                          hook_base_opt=base_hook, hook_mask_opt=mask_hook)
        upscaler.upscale_shape((0, 2), latent(), 128, 128)
    assert seen == [("plain", base_hook), ("model", base_hook)]


# ---------------------------------------------------------------- B9: image node honours use_tiled_vae

def test_iterative_image_upscale_uses_tiled_vae_when_upscaler_does(impact, monkeypatch):
    nodes = sys.modules["nodes"]
    used = []

    class Enc:
        def encode(self, vae, pixels, *a, **k):
            used.append(("encode", type(self).__name__))
            return (latent(),)

    class Dec:
        def decode(self, vae, samples, *a, **k):
            used.append(("decode", type(self).__name__))
            return (torch.zeros(1, 64, 64, 3),)

    monkeypatch.setattr(nodes, "VAEEncode", type("VAEEncode", (Enc,), {}))
    monkeypatch.setattr(nodes, "VAEEncodeTiled", type("VAEEncodeTiled", (Enc,), {}))
    monkeypatch.setattr(nodes, "VAEDecode", type("VAEDecode", (Dec,), {}))
    monkeypatch.setattr(nodes, "VAEDecodeTiled", type("VAEDecodeTiled", (Dec,), {}))
    monkeypatch.setattr(impact.upscale_nodes.IterativeLatentUpscale, "doit", lambda self, lat, *a: (lat, None))

    for upscaler in (
        impact.core.PixelKSampleUpscaler("bilinear", "m", "vae", 0, 20, 7.0, "euler", "normal", [], [], 0.5, True),
        impact.core.TwoSamplersForMaskUpscaler("bilinear", "none", True, None, None, torch.zeros(8, 8), "vae"),
    ):
        used.clear()
        impact.upscale_nodes.IterativeImageUpscale().doit(torch.zeros(1, 64, 64, 3), 2.0, 2, "", upscaler, "vae")
        assert used == [("encode", "VAEEncodeTiled"), ("decode", "VAEDecodeTiled")], type(upscaler).__name__


# ---------------------------------------------------------------- B11: cfg schedule keeps fractions

def test_cfg_schedule_hook_keeps_fractional_cfg(impact):
    hook = impact.hooks.SimpleCfgScheduleHook(3.5)
    got = []
    for i in range(3):
        hook.set_steps((i, 3))
        got.append(hook.pre_ksample("m", 0, 20, 7.5, "euler", "normal", [], [], {}, 0.5)[3])
    assert got == pytest.approx([7.5, 5.5, 3.5])


# ---------------------------------------------------------------- B12: noise / unsampler schedules reach their end value

def test_unsampler_hook_reaches_end_step(impact, monkeypatch):
    seen = []

    class Unsampler:
        def unsampler(self, model, cfg, sampler_name, steps, end_at_step, *a):
            seen.append(end_at_step)
            return (a[-1],)

    monkeypatch.setattr(impact.hooks.noise_nodes, "Unsampler", Unsampler)
    hook = impact.hooks.UnsamplerHook("m", 25, 21, 24, 1.0, "euler", "normal", "disable", [], [])
    for i in range(4):
        hook.set_steps((i, 4))
        hook.post_encode(latent())
    assert seen == [21, 22, 23, 24]


def test_noise_injection_strength_reaches_end(impact, node_mappings):
    strengths = []

    class BNK_NoisyLatentImage:
        def create_noisy_latents(self, source, seed, width, height, batch_size):
            return ({"samples": torch.zeros(batch_size, 4, height // 8, width // 8)},)

    class BNK_InjectNoise:
        def inject_noise(self, latents, strength, noise=None, mask=None):
            strengths.append(strength)
            return (dict(latents),)

    node_mappings.update(BNK_NoisyLatentImage=BNK_NoisyLatentImage, BNK_InjectNoise=BNK_InjectNoise)
    hook = impact.hooks.InjectNoiseHook("CPU", 0, 1.0, 0.25)
    for i in range(4):
        hook.set_steps((i, 4))
        hook.post_encode(latent())
    assert strengths == pytest.approx([1.0, 0.75, 0.5, 0.25])


# ---------------------------------------------------------------- B13: factor 1 still runs every step

def test_iterative_latent_upscale_runs_all_steps_at_factor_one(impact, status):
    calls = []

    class CountingUpscaler:
        vae = "vae"

        def upscale_shape(self, step_info, samples, w, h, prefix=None):
            calls.append((step_info, w, h))
            return dict(samples)

    impact.upscale_nodes.IterativeLatentUpscale().doit(latent(), 1.0, 1, "", CountingUpscaler())
    assert calls == [((0, 1), 64, 64)]


# ---------------------------------------------------------------- B14: progress badge cleared on error

def test_status_cleared_when_upscaler_raises(impact, status):
    class FailingUpscaler:
        vae = "vae"

        def upscale_shape(self, *a, **k):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        impact.upscale_nodes.IterativeLatentUpscale().doit(latent(), 2.0, 3, "", FailingUpscaler(), unique_id="7")
    assert status[-1][1]["text"] == ""


# ---------------------------------------------------------------- B15: noise injection on non-4-channel latents

def test_noise_injection_works_on_16_channel_latent(impact, node_mappings):
    # ComfyUI_Noise's NoisyLatentImage always makes 4-channel noise and InjectNoise
    # silently skips on a shape mismatch, so 16-channel models (Flux, SD3, Qwen-Image) got no noise.
    class BNK_NoisyLatentImage:
        def create_noisy_latents(self, source, seed, width, height, batch_size):
            return ({"samples": torch.ones(batch_size, 4, height // 8, width // 8)},)

    class BNK_InjectNoise:
        def inject_noise(self, latents, strength, noise=None, mask=None):
            s = latents.copy()
            if latents["samples"].shape != noise["samples"].shape:
                return (s,)
            s["samples"] = latents["samples"] + noise["samples"] * strength
            return (s,)

    node_mappings.update(BNK_NoisyLatentImage=BNK_NoisyLatentImage, BNK_InjectNoise=BNK_InjectNoise)
    hook = impact.hooks.InjectNoiseHook("CPU", 42, 1.0, 1.0)
    hook.set_steps((0, 2))
    src = latent(c=16, h=8, w=8)
    out = hook.post_encode(src)
    assert out["samples"].shape == src["samples"].shape
    assert out["samples"].std() > 0.5  # noise was really added

    # Same seed and step give the same noise.
    again = hook.post_encode(latent(c=16, h=8, w=8))
    assert torch.equal(out["samples"], again["samples"])

    # 4-channel latents still go through ComfyUI_Noise, so existing results do not change.
    four = hook.post_encode(latent(c=4))
    assert torch.equal(four["samples"], torch.ones(1, 4, 8, 8))


def test_noise_injection_without_comfyui_noise_installed(impact, node_mappings):
    hook = impact.hooks.InjectNoiseHook("CPU", 1, 0.5, 0.5)
    hook.set_steps((0, 1))
    mask = torch.zeros(1, 8, 8)
    mask[:, :, :4] = 1.0
    out = hook.post_encode({"samples": torch.zeros(1, 16, 8, 8), "noise_mask": mask})
    assert out["samples"][..., :4].abs().sum() > 0
    assert torch.all(out["samples"][..., 4:] == 0)  # unmasked area untouched
    assert out["noise_mask"] is mask


# ---------------------------------------------------------------- B16: dead upscale() path honours tile_size

def test_pixel_ksample_upscaler_upscale_uses_tile_size(impact, monkeypatch):
    seen = []
    monkeypatch.setattr(impact.core, "latent_upscale_on_pixel_space2",
                        lambda *a, **k: seen.append(k["tile_size"]) or (latent(), torch.zeros(1, 8, 8, 3)))
    monkeypatch.setattr(impact.impact_sampling, "impact_sample", lambda *a, **k: a[8])
    upscaler = impact.core.PixelKSampleUpscaler("bilinear", "m", "vae", 0, 20, 7.0, "euler", "normal", [], [], 0.5,
                                                True, tile_size=1024)
    upscaler.upscale((0, 1), latent(), 2.0)
    assert seen == [1024]


# ---------------------------------------------------------------- B17: provider errors are not swallowed

def test_unsampler_provider_does_not_swallow_errors(impact, monkeypatch):
    def broken(*a, **k):
        raise ValueError("bad input")

    monkeypatch.setattr(impact.hooks, "UnsamplerHook", broken)
    with pytest.raises(ValueError):
        impact.upscale_nodes.UnsamplerHookProvider().doit("m", 25, 21, 24, 1.0, "euler", "normal", "disable", [], [], "simple")


# ---------------------------------------------------------------- B18: tiled upscaler keeps noise_mask

def test_tiled_upscaler_passes_noise_mask_to_sampler(impact, node_mappings, monkeypatch):
    seen = []

    class BNK_TiledKSampler:
        def sample(self, *args):
            seen.append(args[11].get("noise_mask"))
            return (args[11],)

    node_mappings["BNK_TiledKSampler"] = BNK_TiledKSampler
    monkeypatch.setattr(impact.core, "latent_upscale_on_pixel_space_shape2", fake_pixel_space([]))
    upscaler = impact.core.PixelTiledKSampleUpscaler("bilinear", "model", "vae", 0, 20, 7.0, "euler", "normal",
                                                     [], [], 0.6, 512, 512, "random")
    mask = torch.ones(1, 8, 8)
    upscaler.upscale_shape((0, 2), {"samples": torch.zeros(1, 4, 8, 8), "noise_mask": mask}, 128, 128)
    assert seen == [mask]


# ---------------------------------------------------------------- B19: step sizes are VAE-factor multiples

def test_iterative_latent_upscale_sizes_are_vae_multiples(impact, status):
    sizes = []

    class SizeUpscaler:
        vae = "vae"

        def upscale_shape(self, step_info, samples, w, h, prefix=None):
            sizes.append((w, h))
            return {"samples": torch.zeros(1, 4, int(h) // 8, int(w) // 8)}

    # 512x768 pixels at x1.5 over 3 steps: 597.33 / 682.67 / 768 wide before the fix.
    impact.upscale_nodes.IterativeLatentUpscale().doit(latent(h=96, w=64), 1.5, 3, "", SizeUpscaler())
    assert sizes == [(592, 896), (680, 1024), (768, 1152)]
    assert all(w % 8 == 0 and h % 8 == 0 for w, h in sizes)


# ---------------------------------------------------------------- B20: overlap too large for the tile is clamped

def test_tiled_vae_overlap_clamped_to_quarter_tile(impact, monkeypatch):
    nodes = sys.modules["nodes"]
    seen = []

    class VAEEncodeTiled:
        def encode(self, vae, pixels, tile_size, overlap, temporal_size=64, temporal_overlap=8):
            seen.append(("encode", tile_size, overlap))
            return (latent(),)

    class VAEDecodeTiled:
        def decode(self, vae, samples, tile_size, overlap=64, temporal_size=64, temporal_overlap=8):
            seen.append(("decode", tile_size, overlap))
            return (torch.zeros(1, 64, 64, 3),)

    monkeypatch.setattr(nodes, "VAEEncodeTiled", VAEEncodeTiled)
    monkeypatch.setattr(nodes, "VAEDecodeTiled", VAEDecodeTiled)
    impact.core.vae_encode("vae", torch.zeros(1, 64, 64, 3), True, None, tile_size=512, overlap=256)
    impact.core.vae_decode("vae", latent(), True, None, tile_size=512, overlap=256)
    impact.core.vae_encode("vae", torch.zeros(1, 64, 64, 3), True, None, tile_size=512, overlap=64)
    assert seen == [("encode", 512, 128), ("decode", 512, 128), ("encode", 512, 64)]

    monkeypatch.setattr(impact.upscale_nodes.IterativeLatentUpscale, "doit", lambda self, lat, *a: (lat, None))
    upscaler = impact.core.PixelTiledKSampleUpscaler("bilinear", "m", "vae", 0, 20, 7.0, "euler", "normal",
                                                     [], [], 0.5, 512, 512, "random", tile_size=512, overlap=1024)
    seen.clear()
    impact.upscale_nodes.IterativeImageUpscale().doit(torch.zeros(1, 64, 64, 3), 2.0, 2, "", upscaler, "vae")
    assert seen == [("encode", 512, 128), ("decode", 512, 128)]


# ---------------------------------------------------------------- B21: vae_compression 0 reads the VAE's factor

def test_vae_compression_zero_uses_vae_factor(impact, status):
    sizes = []

    class VAE:
        def spacial_compression_decode(self):
            return 16

    class SizeUpscaler:
        vae = VAE()

        def upscale_shape(self, step_info, samples, w, h, prefix=None):
            sizes.append((w, h))
            return dict(samples)

    impact.upscale_nodes.IterativeLatentUpscale().doit(latent(h=4, w=4), 2.0, 1, "", SizeUpscaler(), vae_compression=0)
    assert sizes == [(128, 128)]
