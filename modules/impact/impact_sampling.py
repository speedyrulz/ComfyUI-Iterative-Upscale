import logging

import nodes
from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy import samplers
from comfy_extras import nodes_custom_sampler
import latent_preview
import comfy
import torch
import math
import comfy.model_management as mm


try:
    from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise, Noise_RandomNoise
    import node_helpers
except Exception:
    logging.warning("\n#############################################\n[Impact Pack] ComfyUI is an outdated version.\n#############################################\n")
    raise Exception("[Impact Pack] ComfyUI is an outdated version.")


def calculate_sigmas(model, sampler, scheduler, steps):
    discard_penultimate_sigma = False
    if sampler in ['dpm_2', 'dpm_2_ancestral', 'uni_pc', 'uni_pc_bh2']:
        steps += 1
        discard_penultimate_sigma = True

    if scheduler.startswith('AYS'):
        sigmas = nodes.NODE_CLASS_MAPPINGS['AlignYourStepsScheduler']().get_sigmas(scheduler[4:], steps, denoise=1.0)[0]
    elif scheduler.startswith('GITS[coeff='):
        sigmas = nodes.NODE_CLASS_MAPPINGS['GITSScheduler']().execute(float(scheduler[11:-1]), steps, denoise=1.0)[0]
    elif scheduler == 'LTXV[default]':
        # LTXVScheduler's default shift/stretch/terminal, with the requested number of steps
        sigmas = nodes.NODE_CLASS_MAPPINGS['LTXVScheduler']().execute(steps, 2.05, 0.95, True, 0.1)[0]
    elif scheduler.startswith('OSS'):
        sigmas = nodes.NODE_CLASS_MAPPINGS['OptimalStepsScheduler']().execute(scheduler[4:], steps, denoise=1.0)[0]
    else:
        sigmas = samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps)

    if discard_penultimate_sigma:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
    return sigmas


def get_noise_sampler(x, cpu, total_sigmas, **kwargs):
    if 'extra_args' in kwargs and 'seed' in kwargs['extra_args']:
        sigma_min, sigma_max = total_sigmas[total_sigmas > 0].min(), total_sigmas.max()
        seed = kwargs['extra_args'].get("seed", None)
        return k_diffusion_sampling.BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=cpu)
    return None


def ksampler(sampler_name, total_sigmas, extra_options={}, inpaint_options={}):
    if sampler_name in ["dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu"]:
        if sampler_name == "dpmpp_sde":
            orig_sampler_function = k_diffusion_sampling.sample_dpmpp_sde
        elif sampler_name == "dpmpp_sde_gpu":
            orig_sampler_function = k_diffusion_sampling.sample_dpmpp_sde_gpu
        elif sampler_name == "dpmpp_2m_sde":
            orig_sampler_function = k_diffusion_sampling.sample_dpmpp_2m_sde
        elif sampler_name == "dpmpp_2m_sde_gpu":
            orig_sampler_function = k_diffusion_sampling.sample_dpmpp_2m_sde_gpu
        elif sampler_name == "dpmpp_3m_sde":
            orig_sampler_function = k_diffusion_sampling.sample_dpmpp_3m_sde
        elif sampler_name == "dpmpp_3m_sde_gpu":
            orig_sampler_function = k_diffusion_sampling.sample_dpmpp_3m_sde_gpu

        def sampler_function_wrapper(model, x, sigmas, **kwargs):
            if 'noise_sampler' not in kwargs:
                kwargs['noise_sampler'] = get_noise_sampler(x, 'gpu' not in sampler_name, total_sigmas, **kwargs)

            return orig_sampler_function(model, x, sigmas, **kwargs)

        sampler_function = sampler_function_wrapper

    else:
        return comfy.samplers.sampler_object(sampler_name)

    return samplers.KSAMPLER(sampler_function, extra_options, inpaint_options)


# modified version of SamplerCustom.sample
def sample_with_custom_noise(model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image, noise=None, callback=None):
    latent = latent_image
    latent_image = latent["samples"]

    if hasattr(comfy.sample, 'fix_empty_latent_channels'):
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    out = latent.copy()
    out['samples'] = latent_image

    if noise is None:
        if not add_noise:
            noise = Noise_EmptyNoise().generate_noise(out)
        else:
            noise = Noise_RandomNoise(noise_seed).generate_noise(out)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    x0_output = {}
    preview_callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

    if callback is not None:
        def touched_callback(step, x0, x, total_steps):
            callback(step, x0, x, total_steps)
            preview_callback(step, x0, x, total_steps)
    else:
        touched_callback = preview_callback

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    device = mm.get_torch_device()

    noise = noise.to(device)
    latent_image = latent_image.to(device)
    if noise_mask is not None:
        noise_mask = noise_mask.to(device)

    if negative != 'NegativePlaceholder':
        # This way is incompatible with Advanced ControlNet, yet.
        # guider = comfy.samplers.CFGGuider(model)
        # guider.set_conds(positive, negative)
        # guider.set_cfg(cfg)
        samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image,
                                             noise_mask=noise_mask, callback=touched_callback,
                                             disable_pbar=disable_pbar, seed=noise_seed)
    else:
        guider = nodes_custom_sampler.Guider_Basic(model)
        positive = node_helpers.conditioning_set_values(positive, {"guidance": cfg})
        guider.set_conds(positive)
        samples = guider.sample(noise, latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=touched_callback, disable_pbar=disable_pbar, seed=noise_seed)

    samples = samples.to(comfy.model_management.intermediate_device())

    out["samples"] = samples
    if "x0" in x0_output:
        out_denoised = latent.copy()
        out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
    else:
        out_denoised = out
    return out, out_denoised


# When sampling one step at a time, it mitigates the problem. (especially for _sde series samplers)
def separated_sample(model, add_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                     latent_image, start_at_step, end_at_step, return_with_leftover_noise, sigma_ratio=1.0, sampler_opt=None, noise=None, callback=None, scheduler_func=None):

    if scheduler_func is not None:
        total_sigmas = scheduler_func(model, sampler_name, steps)
    else:
        if sampler_opt is None:
            total_sigmas = calculate_sigmas(model, sampler_name, scheduler, steps)
        else:
            total_sigmas = calculate_sigmas(model, "", scheduler, steps)

    sigmas = total_sigmas

    if end_at_step is not None and end_at_step < (len(total_sigmas) - 1):
        sigmas = total_sigmas[:end_at_step + 1]
        if not return_with_leftover_noise:
            sigmas[-1] = 0

    if start_at_step is not None:
        if start_at_step < (len(sigmas) - 1):
            sigmas = sigmas[start_at_step:] * sigma_ratio
        else:
            if latent_image is not None:
                return latent_image
            else:
                return {'samples': torch.zeros_like(noise)}

    if sampler_opt is None:
        impact_sampler = ksampler(sampler_name, total_sigmas)
    else:
        impact_sampler = sampler_opt

    if len(sigmas) == 0 or (len(sigmas) == 1 and sigmas[0] == 0):
        return latent_image

    res = sample_with_custom_noise(model, add_noise, seed, cfg, positive, negative, impact_sampler, sigmas, latent_image, noise=noise, callback=callback)

    if return_with_leftover_noise:
        return res[0]
    else:
        return res[1]


def impact_sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, sigma_ratio=1.0, sampler_opt=None, noise=None, scheduler_func=None):
    if denoise <= 0.0:
        # Nothing to denoise; same as ComfyUI's KSampler (an empty sigma schedule).
        return latent_image

    advanced_steps = math.floor(steps / denoise)
    start_at_step = advanced_steps - steps
    end_at_step = start_at_step + steps
    return separated_sample(model, True, seed, advanced_steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                            start_at_step, end_at_step, False, scheduler_func=scheduler_func)


class KSamplerWrapper:
    params = None

    def __init__(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, scheduler_func=None):
        self.params = model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise
        self.scheduler_func = scheduler_func

    def sample(self, latent_image, hook=None):
        model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params

        if hook is not None:
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise = \
                hook.pre_ksample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)

        return impact_sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, scheduler_func=self.scheduler_func)
