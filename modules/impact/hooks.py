import logging

import nodes

from . import utils
from . import noise_nodes


class PixelKSampleHook:
    cur_step = 0
    total_step = 0

    def __init__(self):
        pass

    def set_steps(self, info):
        self.cur_step, self.total_step = info

    def post_decode(self, pixels):
        return pixels

    def post_upscale(self, pixels, mask=None):
        return pixels

    def post_encode(self, samples):
        return samples

    def pre_decode(self, samples):
        return samples

    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent,
                    denoise):
        return model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise

    def post_crop_region(self, w, h, item_bbox, crop_region):
        return crop_region

    def touch_scaled_size(self, w, h):
        return w, h


class PixelKSampleHookCombine(PixelKSampleHook):
    hook1 = None
    hook2 = None

    def __init__(self, hook1, hook2):
        super().__init__()
        self.hook1 = hook1
        self.hook2 = hook2

    def set_steps(self, info):
        self.hook1.set_steps(info)
        self.hook2.set_steps(info)

    def pre_decode(self, samples):
        return self.hook2.pre_decode(self.hook1.pre_decode(samples))

    def post_decode(self, pixels):
        return self.hook2.post_decode(self.hook1.post_decode(pixels))

    def post_upscale(self, pixels, mask=None):
        return self.hook2.post_upscale(self.hook1.post_upscale(pixels, mask), mask)

    def post_encode(self, samples):
        return self.hook2.post_encode(self.hook1.post_encode(samples))

    def post_crop_region(self, w, h, item_bbox, crop_region):
        crop_region = self.hook1.post_crop_region(w, h, item_bbox, crop_region)
        return self.hook2.post_crop_region(w, h, item_bbox, crop_region)

    def touch_scaled_size(self, w, h):
        w, h = self.hook1.touch_scaled_size(w, h)
        return self.hook2.touch_scaled_size(w, h)

    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent,
                    denoise):
        model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise = \
            self.hook1.pre_ksample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                   upscaled_latent, denoise)

        return self.hook2.pre_ksample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                      upscaled_latent, denoise)


class SimpleCfgScheduleHook(PixelKSampleHook):
    target_cfg = 0

    def __init__(self, target_cfg):
        super().__init__()
        self.target_cfg = target_cfg

    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise):
        if self.total_step > 1:
            progress = self.cur_step / (self.total_step - 1)
            gap = self.target_cfg - cfg
            current_cfg = int(cfg + gap * progress)
        else:
            current_cfg = self.target_cfg

        return model, seed, steps, current_cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise


class SimpleDenoiseScheduleHook(PixelKSampleHook):
    def __init__(self, target_denoise):
        super().__init__()
        self.target_denoise = target_denoise

    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise):
        if self.total_step > 1:
            progress = self.cur_step / (self.total_step - 1)
            gap = self.target_denoise - denoise
            current_denoise = denoise + gap * progress
        else:
            current_denoise = self.target_denoise

        return model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, current_denoise


class SimpleStepsScheduleHook(PixelKSampleHook):
    def __init__(self, target_steps):
        super().__init__()
        self.target_steps = target_steps

    def pre_ksample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise):
        if self.total_step > 1:
            progress = self.cur_step / (self.total_step - 1)
            gap = self.target_steps - steps
            current_steps = int(steps + gap * progress)
        else:
            current_steps = self.target_steps

        return model, seed, current_steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise


# REQUIREMENTS: BlenderNeko/ComfyUI Noise
class InjectNoiseHook(PixelKSampleHook):
    def __init__(self, source, seed, start_strength, end_strength):
        super().__init__()
        self.source = source
        self.seed = seed
        self.start_strength = start_strength
        self.end_strength = end_strength

    def post_encode(self, samples):
        cur_step = self.cur_step

        size = samples['samples'].shape
        seed = cur_step + self.seed + cur_step

        if "BNK_NoisyLatentImage" in nodes.NODE_CLASS_MAPPINGS and "BNK_InjectNoise" in nodes.NODE_CLASS_MAPPINGS:
            NoisyLatentImage = nodes.NODE_CLASS_MAPPINGS["BNK_NoisyLatentImage"]
            InjectNoise = nodes.NODE_CLASS_MAPPINGS["BNK_InjectNoise"]
        else:
            utils.try_install_custom_node('https://github.com/BlenderNeko/ComfyUI_Noise',
                                          "To use 'NoiseInjectionHookProvider', 'ComfyUI Noise' extension is required.")
            raise Exception("'BNK_NoisyLatentImage', 'BNK_InjectNoise' nodes are not installed.")

        noise = NoisyLatentImage().create_noisy_latents(self.source, seed, size[3] * 8, size[2] * 8, size[0])[0]

        # inj noise
        mask = None
        if 'noise_mask' in samples:
            mask = samples['noise_mask']

        strength = self.start_strength + (self.end_strength - self.start_strength) * cur_step / self.total_step
        samples = InjectNoise().inject_noise(samples, strength, noise, mask)[0]
        logging.info(f"[Impact Pack] InjectNoiseHook: strength = {strength}")

        if mask is not None:
            samples['noise_mask'] = mask

        return samples


class UnsamplerHook(PixelKSampleHook):
    def __init__(self, model, steps, start_end_at_step, end_end_at_step, cfg, sampler_name,
                 scheduler, normalize, positive, negative):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.steps = steps
        self.start_end_at_step = start_end_at_step
        self.end_end_at_step = end_end_at_step
        self.scheduler = scheduler
        self.normalize = normalize
        self.positive = positive
        self.negative = negative

    def post_encode(self, samples):
        cur_step = self.cur_step

        Unsampler = noise_nodes.Unsampler

        end_at_step = self.start_end_at_step + (self.end_end_at_step - self.start_end_at_step) * cur_step / self.total_step
        end_at_step = int(end_at_step)

        logging.info(f"[Impact Pack] UnsamplerHook: end_at_step = {end_at_step}")

        # inj noise
        mask = None
        if 'noise_mask' in samples:
            mask = samples['noise_mask']

        samples = Unsampler().unsampler(self.model, self.cfg, self.sampler_name, self.steps, end_at_step,
                                        self.scheduler, self.normalize, self.positive, self.negative, samples)[0]

        if mask is not None:
            samples['noise_mask'] = mask

        return samples
