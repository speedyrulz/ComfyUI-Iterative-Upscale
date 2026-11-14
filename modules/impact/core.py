import inspect
import logging
import warnings

import torch
import torch.nn.functional as F

import comfy
import comfy.samplers
import nodes
from server import PromptServer

from . import impact_sampling
from . import utils


ADDITIONAL_SCHEDULERS = ['AYS SDXL', 'AYS SD1', 'AYS SVD', 'GITS[coeff=1.2]', 'LTXV[default]', 'OSS FLUX', 'OSS Wan', 'OSS Chroma']


def get_schedulers():
    return list(comfy.samplers.SCHEDULER_HANDLERS) + ADDITIONAL_SCHEDULERS


def clamp_overlap(tile_size, overlap):
    # Same rule as VAEDecodeTiled. VAEEncodeTiled has no such check, and ComfyUI's tiled
    # encoder also uses tile_size // 2 tiles, so overlap >= tile_size / 2 divides by zero.
    if tile_size < overlap * 4:
        overlap = tile_size // 4
    return overlap


def vae_decode(vae, samples, use_tile, hook, tile_size=512, overlap=64):
    overlap = clamp_overlap(tile_size, overlap)
    if use_tile:
        decoder = nodes.VAEDecodeTiled()
        if 'overlap' in inspect.signature(decoder.decode).parameters:
            pixels = decoder.decode(vae, samples, tile_size, overlap=overlap)[0]
        else:
            logging.warning("[Impact Pack] Your ComfyUI is outdated.")
            pixels = decoder.decode(vae, samples, tile_size)[0]
    else:
        pixels = nodes.VAEDecode().decode(vae, samples)[0]

    if hook is not None:
        pixels = hook.post_decode(pixels)

    return pixels


def vae_encode(vae, pixels, use_tile, hook, tile_size=512, overlap=64):
    overlap = clamp_overlap(tile_size, overlap)
    if use_tile:
        encoder = nodes.VAEEncodeTiled()
        if 'overlap' in inspect.signature(encoder.encode).parameters:
            samples = encoder.encode(vae, pixels, tile_size, overlap=overlap)[0]
        else:
            logging.warning("[Impact Pack] Your ComfyUI is outdated.")
            samples = encoder.encode(vae, pixels, tile_size)[0]
    else:
        samples = nodes.VAEEncode().encode(vae, pixels)[0]

    if hook is not None:
        samples = hook.post_encode(samples)

    return samples


def latent_upscale_on_pixel_space_shape(samples, scale_method, w, h, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    return latent_upscale_on_pixel_space_shape2(samples, scale_method, w, h, vae, use_tile, tile_size, save_temp_prefix, hook, overlap=overlap)[0]


def latent_upscale_on_pixel_space_shape2(samples, scale_method, w, h, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    pixels = vae_decode(vae, samples, use_tile, hook, tile_size=tile_size, overlap=overlap)

    if save_temp_prefix is not None:
        nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(w), int(h), False)[0]

    old_pixels = pixels
    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return vae_encode(vae, pixels, use_tile, hook, tile_size=tile_size, overlap=overlap), old_pixels


def latent_upscale_on_pixel_space(samples, scale_method, scale_factor, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    return latent_upscale_on_pixel_space2(samples, scale_method, scale_factor, vae, use_tile, tile_size, save_temp_prefix, hook, overlap=overlap)[0]


def latent_upscale_on_pixel_space2(samples, scale_method, scale_factor, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    pixels = vae_decode(vae, samples, use_tile, hook, tile_size=tile_size, overlap=overlap)

    if save_temp_prefix is not None:
        nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

    w = pixels.shape[2] * scale_factor
    h = pixels.shape[1] * scale_factor
    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(w), int(h), False)[0]

    old_pixels = pixels
    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return vae_encode(vae, pixels, use_tile, hook, tile_size=tile_size, overlap=overlap), old_pixels


def latent_upscale_on_pixel_space_with_model_shape(samples, scale_method, upscale_model, new_w, new_h, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    return latent_upscale_on_pixel_space_with_model_shape2(samples, scale_method, upscale_model, new_w, new_h, vae, use_tile, tile_size, save_temp_prefix, hook, overlap=overlap)[0]


def latent_upscale_on_pixel_space_with_model_shape2(samples, scale_method, upscale_model, new_w, new_h, vae, use_tile=False, tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    pixels = vae_decode(vae, samples, use_tile, hook, tile_size=tile_size, overlap=overlap)

    if save_temp_prefix is not None:
        nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

    w = pixels.shape[2]

    # upscale by model upscaler
    current_w = w
    while current_w < new_w:
        model_upscaler = nodes.NODE_CLASS_MAPPINGS['ImageUpscaleWithModel']()
        if hasattr(model_upscaler, 'execute'):
            pixels = model_upscaler.execute(upscale_model, pixels)[0]
        else:
            pixels = model_upscaler.upscale(upscale_model, pixels)[0]

        current_w = pixels.shape[2]
        if current_w == w:
            logging.info("[latent_upscale_on_pixel_space_with_model] x1 upscale model selected")
            break

    # downscale to target scale
    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(new_w), int(new_h), False)[0]

    old_pixels = pixels
    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return vae_encode(vae, pixels, use_tile, hook, tile_size=tile_size, overlap=overlap), old_pixels


def latent_upscale_on_pixel_space_with_model(samples, scale_method, upscale_model, scale_factor, vae, use_tile=False,
                                             tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    return latent_upscale_on_pixel_space_with_model2(samples, scale_method, upscale_model, scale_factor, vae, use_tile, tile_size, save_temp_prefix, hook, overlap=overlap)[0]


def latent_upscale_on_pixel_space_with_model2(samples, scale_method, upscale_model, scale_factor, vae, use_tile=False,
                                              tile_size=512, save_temp_prefix=None, hook=None, overlap=64):
    pixels = vae_decode(vae, samples, use_tile, hook, tile_size=tile_size, overlap=overlap)

    if save_temp_prefix is not None:
        nodes.PreviewImage().save_images(pixels, filename_prefix=save_temp_prefix)

    w = pixels.shape[2]
    h = pixels.shape[1]

    new_w = w * scale_factor
    new_h = h * scale_factor

    # upscale by model upscaler
    current_w = w
    while current_w < new_w:
        model_upscaler = nodes.NODE_CLASS_MAPPINGS['ImageUpscaleWithModel']()
        if hasattr(model_upscaler, 'execute'):
            pixels = model_upscaler.execute(upscale_model, pixels)[0]
        else:
            pixels = model_upscaler.upscale(upscale_model, pixels)[0]

        current_w = pixels.shape[2]
        if current_w == w:
            logging.info("[latent_upscale_on_pixel_space_with_model] x1 upscale model selected")
            break

    # downscale to target scale
    pixels = nodes.ImageScale().upscale(pixels, scale_method, int(new_w), int(new_h), False)[0]

    old_pixels = pixels
    if hook is not None:
        pixels = hook.post_upscale(pixels)

    return vae_encode(vae, pixels, use_tile, hook, tile_size=tile_size, overlap=overlap), old_pixels


class TwoSamplersForMaskUpscaler:
    def __init__(self, scale_method, sample_schedule, use_tiled_vae, base_sampler, mask_sampler, mask, vae,
                 full_sampler_opt=None, upscale_model_opt=None, hook_base_opt=None, hook_mask_opt=None,
                 hook_full_opt=None,
                 tile_size=512):

        mask = utils.make_2d_mask(mask)

        mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))

        self.params = scale_method, sample_schedule, use_tiled_vae, base_sampler, mask_sampler, mask, vae
        self.upscale_model = upscale_model_opt
        self.full_sampler = full_sampler_opt
        self.hook_base = hook_base_opt
        self.hook_mask = hook_mask_opt
        self.hook_full = hook_full_opt
        self.use_tiled_vae = use_tiled_vae
        self.tile_size = tile_size
        self.is_tiled = False
        self.vae = vae

    def upscale(self, step_info, samples, upscale_factor, save_temp_prefix=None):
        scale_method, sample_schedule, use_tiled_vae, base_sampler, mask_sampler, mask, vae = self.params

        mask = utils.make_2d_mask(mask)

        self.prepare_hook(step_info)

        # upscale latent
        if self.upscale_model is None:
            upscaled_latent = latent_upscale_on_pixel_space(samples, scale_method, upscale_factor, vae,
                                                            use_tile=self.use_tiled_vae,
                                                            save_temp_prefix=save_temp_prefix,
                                                            hook=self.hook_base, tile_size=self.tile_size)
        else:
            upscaled_latent = latent_upscale_on_pixel_space_with_model(samples, scale_method, self.upscale_model,
                                                                       upscale_factor, vae,
                                                                       use_tile=self.use_tiled_vae,
                                                                       save_temp_prefix=save_temp_prefix,
                                                                       hook=self.hook_base, tile_size=self.tile_size)

        return self.do_samples(step_info, base_sampler, mask_sampler, sample_schedule, mask, upscaled_latent)

    def prepare_hook(self, step_info):
        if self.hook_base is not None:
            self.hook_base.set_steps(step_info)
        if self.hook_mask is not None:
            self.hook_mask.set_steps(step_info)
        if self.hook_full is not None:
            self.hook_full.set_steps(step_info)

    def upscale_shape(self, step_info, samples, w, h, save_temp_prefix=None):
        scale_method, sample_schedule, use_tiled_vae, base_sampler, mask_sampler, mask, vae = self.params

        mask = utils.make_2d_mask(mask)

        self.prepare_hook(step_info)

        # upscale latent
        if self.upscale_model is None:
            upscaled_latent = latent_upscale_on_pixel_space_shape(samples, scale_method, w, h, vae,
                                                                  use_tile=self.use_tiled_vae,
                                                                  save_temp_prefix=save_temp_prefix,
                                                                  hook=self.hook_base,
                                                                  tile_size=self.tile_size)
        else:
            upscaled_latent = latent_upscale_on_pixel_space_with_model_shape(samples, scale_method, self.upscale_model,
                                                                             w, h, vae,
                                                                             use_tile=self.use_tiled_vae,
                                                                             save_temp_prefix=save_temp_prefix,
                                                                             hook=self.hook_base,
                                                                             tile_size=self.tile_size)

        return self.do_samples(step_info, base_sampler, mask_sampler, sample_schedule, mask, upscaled_latent)

    def is_full_sample_time(self, step_info, sample_schedule):
        cur_step, total_step = step_info

        # step_info is (zero-based step index, number of steps): count steps from 1.
        cur_step += 1
        is_last = cur_step >= total_step

        if sample_schedule == "none":
            return False

        elif sample_schedule == "interleave1":
            return cur_step % 2 == 0

        elif sample_schedule == "interleave2":
            return cur_step % 3 == 0

        elif sample_schedule == "interleave3":
            return cur_step % 4 == 0

        elif sample_schedule == "last1":
            return is_last

        elif sample_schedule == "last2":
            return cur_step >= total_step - 1

        elif sample_schedule == "interleave1+last1":
            return cur_step % 2 == 0 or is_last

        elif sample_schedule == "interleave2+last1":
            return cur_step % 3 == 0 or is_last

        elif sample_schedule == "interleave3+last1":
            return cur_step % 4 == 0 or is_last

        return False

    def do_samples(self, step_info, base_sampler, mask_sampler, sample_schedule, mask, upscaled_latent):
        mask = utils.make_2d_mask(mask)

        if self.is_full_sample_time(step_info, sample_schedule):
            logging.info(f"step_info={step_info} / full time")

            upscaled_latent = base_sampler.sample(upscaled_latent, self.hook_base)
            sampler = self.full_sampler if self.full_sampler is not None else base_sampler
            return sampler.sample(upscaled_latent, self.hook_full)

        else:
            logging.info(f"step_info={step_info} / non-full time")
            # upscale mask: F.interpolate needs [B, C, H, W]
            if mask.ndim == 2:
                mask = mask[None, None, :, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
            upscaled_mask = F.interpolate(mask, size=(upscaled_latent['samples'].shape[2], upscaled_latent['samples'].shape[3]), mode='bilinear', align_corners=True)
            upscaled_mask = upscaled_mask[:, :, :upscaled_latent['samples'].shape[2], :upscaled_latent['samples'].shape[3]]

            # base sampler
            upscaled_inv_mask = torch.where(upscaled_mask != 1.0, torch.tensor(1.0), torch.tensor(0.0))
            upscaled_latent['noise_mask'] = upscaled_inv_mask
            upscaled_latent = base_sampler.sample(upscaled_latent, self.hook_base)

            # mask sampler
            upscaled_latent['noise_mask'] = upscaled_mask
            upscaled_latent = mask_sampler.sample(upscaled_latent, self.hook_mask)

            # remove mask
            del upscaled_latent['noise_mask']
            return upscaled_latent


class PixelKSampleUpscaler:
    def __init__(self, scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise,
                 use_tiled_vae, upscale_model_opt=None, hook_opt=None, tile_size=512, scheduler_func=None,
                 tile_cnet_opt=None, tile_cnet_strength=1.0):
        self.params = scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise
        self.upscale_model = upscale_model_opt
        self.hook = hook_opt
        self.use_tiled_vae = use_tiled_vae
        self.tile_size = tile_size
        self.is_tiled = False
        self.vae = vae
        self.scheduler_func = scheduler_func
        self.tile_cnet = tile_cnet_opt
        self.tile_cnet_strength = tile_cnet_strength

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise, images):
        if self.tile_cnet is not None:
            image_batch, image_w, image_h, _ = images.shape
            if image_batch > 1:
                warnings.warn('Multiple latents in batch, Tile ControlNet being ignored')
            else:
                if 'TilePreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
                    raise RuntimeError("'TilePreprocessor' node (from comfyui_controlnet_aux) isn't installed.")
                preprocessor = nodes.NODE_CLASS_MAPPINGS['TilePreprocessor']()
                # might add capacity to set pyrUp_iters later, not needed for now though
                preprocessed = preprocessor.execute(images, pyrUp_iters=3, resolution=min(image_w, image_h))[0]
                positive, negative = nodes.ControlNetApplyAdvanced().apply_controlnet(positive=positive,
                                                                                      negative=negative,
                                                                                      control_net=self.tile_cnet,
                                                                                      image=preprocessed,
                                                                                      strength=self.tile_cnet_strength,
                                                                                      start_percent=0,
                                                                                      end_percent=1.0,
                                                                                      vae=self.vae)

        refined_latent = impact_sampling.impact_sample(model, seed, steps, cfg, sampler_name, scheduler,
                                                       positive, negative, upscaled_latent, denoise, scheduler_func=self.scheduler_func)

        return refined_latent

    def upscale(self, step_info, samples, upscale_factor, save_temp_prefix=None):
        scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params

        if self.hook is not None:
            self.hook.set_steps(step_info)

        if self.upscale_model is None:
            upscaled_latent, upscaled_images = \
                latent_upscale_on_pixel_space2(samples, scale_method, upscale_factor, vae,
                                               use_tile=self.use_tiled_vae,
                                               save_temp_prefix=save_temp_prefix, hook=self.hook, tile_size=self.tile_size)
        else:
            upscaled_latent, upscaled_images = \
                latent_upscale_on_pixel_space_with_model2(samples, scale_method, self.upscale_model,
                                                          upscale_factor, vae,
                                                          use_tile=self.use_tiled_vae,
                                                          save_temp_prefix=save_temp_prefix,
                                                          hook=self.hook,
                                                          tile_size=self.tile_size)

        if self.hook is not None:
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise = \
                self.hook.pre_ksample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                      upscaled_latent, denoise)

        if 'noise_mask' in samples:
            upscaled_latent['noise_mask'] = samples['noise_mask']

        refined_latent = self.sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise, upscaled_images)
        return refined_latent

    def upscale_shape(self, step_info, samples, w, h, save_temp_prefix=None):
        scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params

        if self.hook is not None:
            self.hook.set_steps(step_info)

        if self.upscale_model is None:
            upscaled_latent, upscaled_images = \
                latent_upscale_on_pixel_space_shape2(samples, scale_method, w, h, vae,
                                                     use_tile=self.use_tiled_vae,
                                                     save_temp_prefix=save_temp_prefix, hook=self.hook,
                                                     tile_size=self.tile_size)
        else:
            upscaled_latent, upscaled_images = \
                latent_upscale_on_pixel_space_with_model_shape2(samples, scale_method, self.upscale_model,
                                                                w, h, vae,
                                                                use_tile=self.use_tiled_vae,
                                                                save_temp_prefix=save_temp_prefix,
                                                                hook=self.hook,
                                                                tile_size=self.tile_size)

        if self.hook is not None:
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise = \
                self.hook.pre_ksample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                      upscaled_latent, denoise)

        if 'noise_mask' in samples:
            upscaled_latent['noise_mask'] = samples['noise_mask']

        refined_latent = self.sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise, upscaled_images)
        return refined_latent


# REQUIREMENTS: BlenderNeko/ComfyUI_TiledKSampler
class TiledKSamplerWrapper:
    params = None

    def __init__(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise,
                 tile_width, tile_height, tiling_strategy):
        self.params = model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, tile_width, tile_height, tiling_strategy

    def sample(self, latent_image, hook=None):
        if "BNK_TiledKSampler" in nodes.NODE_CLASS_MAPPINGS:
            TiledKSampler = nodes.NODE_CLASS_MAPPINGS['BNK_TiledKSampler']
        else:
            utils.try_install_custom_node('https://github.com/BlenderNeko/ComfyUI_TiledKSampler',
                                          "To use 'TiledKSamplerProvider', 'Tiled sampling for ComfyUI' extension is required.")
            raise Exception("'BNK_TiledKSampler' node isn't installed.")

        model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, tile_width, tile_height, tiling_strategy = self.params

        if hook is not None:
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise = \
                hook.pre_ksample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                 denoise)

        return TiledKSampler().sample(model, seed, tile_width, tile_height, tiling_strategy, steps, cfg, sampler_name,
                                      scheduler, positive, negative, latent_image, denoise)[0]


class PixelTiledKSampleUpscaler:
    def __init__(self, scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                 denoise,
                 tile_width, tile_height, tiling_strategy,
                 upscale_model_opt=None, hook_opt=None, tile_cnet_opt=None, tile_size=512, tile_cnet_strength=1.0, overlap=64):
        self.params = scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise
        self.vae = vae
        self.tile_params = tile_width, tile_height, tiling_strategy
        self.upscale_model = upscale_model_opt
        self.hook = hook_opt
        self.tile_cnet = tile_cnet_opt
        self.tile_size = tile_size
        self.is_tiled = True
        self.tile_cnet_strength = tile_cnet_strength
        self.overlap = overlap

    def tiled_ksample(self, latent, images):
        if "BNK_TiledKSampler" in nodes.NODE_CLASS_MAPPINGS:
            TiledKSampler = nodes.NODE_CLASS_MAPPINGS['BNK_TiledKSampler']
        else:
            utils.try_install_custom_node('https://github.com/BlenderNeko/ComfyUI_TiledKSampler',
                                          "To use 'PixelTiledKSampleUpscalerProvider', 'Tiled sampling for ComfyUI' extension is required.")
            raise RuntimeError("'BNK_TiledKSampler' node isn't installed.")

        scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params
        tile_width, tile_height, tiling_strategy = self.tile_params

        # PK_HOOK schedules (cfg, denoise, steps) apply here as in PixelKSampleUpscaler.
        if self.hook is not None:
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise = \
                self.hook.pre_ksample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                      latent, denoise)

        if self.tile_cnet is not None:
            image_batch, image_w, image_h, _ = images.shape
            if image_batch > 1:
                warnings.warn('Multiple latents in batch, Tile ControlNet being ignored')
            else:
                if 'TilePreprocessor' not in nodes.NODE_CLASS_MAPPINGS:
                    raise RuntimeError("'TilePreprocessor' node (from comfyui_controlnet_aux) isn't installed.")
                preprocessor = nodes.NODE_CLASS_MAPPINGS['TilePreprocessor']()
                # might add capacity to set pyrUp_iters later, not needed for now though
                preprocessed = preprocessor.execute(images, pyrUp_iters=3, resolution=min(image_w, image_h))[0]

                positive, negative = nodes.ControlNetApplyAdvanced().apply_controlnet(positive=positive,
                                                                                      negative=negative,
                                                                                      control_net=self.tile_cnet,
                                                                                      image=preprocessed,
                                                                                      strength=self.tile_cnet_strength,
                                                                                      start_percent=0, end_percent=1.0,
                                                                                      vae=self.vae)

        return TiledKSampler().sample(model, seed, tile_width, tile_height, tiling_strategy, steps, cfg, sampler_name,
                                      scheduler, positive, negative, latent, denoise)[0]

    def upscale(self, step_info, samples, upscale_factor, save_temp_prefix=None):
        scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params

        if self.hook is not None:
            self.hook.set_steps(step_info)

        if self.upscale_model is None:
            upscaled_latent, upscaled_images = \
                latent_upscale_on_pixel_space2(samples, scale_method, upscale_factor, vae,
                                               use_tile=True, save_temp_prefix=save_temp_prefix,
                                               hook=self.hook, tile_size=self.tile_size, overlap=self.overlap)
        else:
            upscaled_latent, upscaled_images = \
                latent_upscale_on_pixel_space_with_model2(samples, scale_method, self.upscale_model,
                                                          upscale_factor, vae, use_tile=True,
                                                          save_temp_prefix=save_temp_prefix,
                                                          hook=self.hook, tile_size=self.tile_size, overlap=self.overlap)

        if 'noise_mask' in samples:
            upscaled_latent['noise_mask'] = samples['noise_mask']

        refined_latent = self.tiled_ksample(upscaled_latent, upscaled_images)

        return refined_latent

    def upscale_shape(self, step_info, samples, w, h, save_temp_prefix=None):
        scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params

        if self.hook is not None:
            self.hook.set_steps(step_info)

        if self.upscale_model is None:
            upscaled_latent, upscaled_images = \
                latent_upscale_on_pixel_space_shape2(samples, scale_method, w, h, vae,
                                                     use_tile=True, save_temp_prefix=save_temp_prefix,
                                                     hook=self.hook, tile_size=self.tile_size, overlap=self.overlap)
        else:
            upscaled_latent, upscaled_images = \
                latent_upscale_on_pixel_space_with_model_shape2(samples, scale_method,
                                                                self.upscale_model, w, h, vae,
                                                                use_tile=True,
                                                                save_temp_prefix=save_temp_prefix,
                                                                hook=self.hook,
                                                                tile_size=self.tile_size, overlap=self.overlap)

        if 'noise_mask' in samples:
            upscaled_latent['noise_mask'] = samples['noise_mask']

        refined_latent = self.tiled_ksample(upscaled_latent, upscaled_images)

        return refined_latent


def update_node_status(node, text, progress=None):
    if PromptServer.instance.client_id is None:
        return

    PromptServer.instance.send_sync("rulz-iterative-upscale/update_status", {
        "node": node,
        "progress": progress,
        "text": text
    }, PromptServer.instance.client_id)
