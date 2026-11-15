"""Offline test setup: stub the ComfyUI modules the pack imports, then load the pack.

The tests run on plain Python with torch installed (no ComfyUI, no GPU):

    python -m pytest tests

Only the attributes the pack touches at import time are stubbed here. Tests replace the
pieces they exercise (VAE nodes, samplers, schedulers) with recording fakes.
"""

import importlib.util
import pathlib
import sys
import types

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
PACK_NAME = "iterative_upscale_pack"


def _module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(sys.modules[parent], child, mod)
    return mod


class _Unset:
    def __init__(self, *args, **kwargs):
        raise AssertionError(f"{type(self).__name__} must be replaced by the test")


def _reshape_mask(input_mask, output_shape):
    # Same contract as comfy.utils.reshape_mask: resize a mask to a latent's [B, C, H, W].
    dims = len(output_shape) - 2
    if dims == 1:
        scale_mode = "linear"
    elif dims == 2:
        input_mask = input_mask.reshape((-1, 1, input_mask.shape[-2], input_mask.shape[-1]))
        scale_mode = "bilinear"
    else:
        raise ValueError(dims)
    mask = torch.nn.functional.interpolate(input_mask, size=output_shape[2:], mode=scale_mode)
    if mask.shape[1] < output_shape[1]:
        mask = mask.repeat((1, output_shape[1]) + (1,) * dims)[:, :output_shape[1]]
    if mask.shape[0] < output_shape[0]:
        mask = mask.repeat((output_shape[0] - 1) // mask.shape[0] + 1, *([1] * (dims + 1)))[:output_shape[0]]
    return mask


def _install_stubs():
    if "comfy" in sys.modules and getattr(sys.modules["comfy"], "_iu_stub", False):
        return

    comfy = _module("comfy", _iu_stub=True)
    comfy.__path__ = []
    _module("comfy.samplers",
            KSampler=type("KSampler", (), {"SAMPLERS": ["euler", "dpmpp_2m"], "SCHEDULERS": ["normal", "karras"]}),
            SCHEDULER_HANDLERS={"normal": None, "karras": None},
            calculate_sigmas=lambda model_sampling, scheduler, steps: torch.linspace(1.0, 0.0, steps + 1),
            sampler_object=lambda name: ("sampler", name),
            KSAMPLER=lambda fn, extra, inpaint: ("KSAMPLER", fn))
    _module("comfy.model_management",
            get_torch_device=lambda: torch.device("cpu"),
            intermediate_device=lambda: torch.device("cpu"))
    _module("comfy.utils", PROGRESS_BAR_ENABLED=False, reshape_mask=_reshape_mask)
    _module("comfy.sample", sample_custom=_Unset)
    _module("comfy.sampler_helpers")
    _module("comfy.k_diffusion").__path__ = []
    _module("comfy.k_diffusion.sampling", **{n: _Unset for n in [
        "sample_dpmpp_sde", "sample_dpmpp_sde_gpu", "sample_dpmpp_2m_sde", "sample_dpmpp_2m_sde_gpu",
        "sample_dpmpp_3m_sde", "sample_dpmpp_3m_sde_gpu", "BrownianTreeNoiseSampler"]})

    _module("comfy_extras").__path__ = []
    _module("comfy_extras.nodes_custom_sampler", Noise_EmptyNoise=_Unset, Noise_RandomNoise=_Unset, Guider_Basic=_Unset)
    _module("latent_preview", prepare_callback=lambda *a, **k: None)
    _module("node_helpers", conditioning_set_values=lambda c, v: c)

    _module("nodes", MAX_RESOLUTION=16384, NODE_CLASS_MAPPINGS={},
            VAEDecode=_Unset, VAEEncode=_Unset, VAEDecodeTiled=_Unset, VAEEncodeTiled=_Unset,
            ImageScale=_Unset, PreviewImage=_Unset, ControlNetApplyAdvanced=_Unset)

    class _PromptServer:
        instance = None

        def __init__(self):
            self.client_id = None
            self.sent = []

        def send_sync(self, event, data, sid=None):
            self.sent.append((event, data))

    _PromptServer.instance = _PromptServer()
    _module("server", PromptServer=_PromptServer)


def _load_pack():
    if PACK_NAME in sys.modules:
        return sys.modules[PACK_NAME]
    spec = importlib.util.spec_from_file_location(PACK_NAME, REPO / "__init__.py",
                                                  submodule_search_locations=[str(REPO)])
    pack = importlib.util.module_from_spec(spec)
    sys.modules[PACK_NAME] = pack
    spec.loader.exec_module(pack)
    return pack


sys.dont_write_bytecode = True
_install_stubs()
PACK = _load_pack()


@pytest.fixture
def pack():
    return PACK


@pytest.fixture
def impact():
    """The pack's modules: core, hooks, impact_sampling, sampler_nodes, upscale_nodes."""
    ns = types.SimpleNamespace()
    for name in ["core", "hooks", "impact_sampling", "sampler_nodes", "upscale_nodes", "utils"]:
        setattr(ns, name, sys.modules[f"{PACK_NAME}.modules.impact.{name}"])
    return ns


@pytest.fixture
def node_mappings(monkeypatch):
    """A clean nodes.NODE_CLASS_MAPPINGS for one test."""
    mappings = {}
    monkeypatch.setattr(sys.modules["nodes"], "NODE_CLASS_MAPPINGS", mappings)
    return mappings


@pytest.fixture
def status(monkeypatch):
    """Record update_node_status calls."""
    calls = []
    server = sys.modules["server"].PromptServer.instance
    monkeypatch.setattr(server, "client_id", "test-client")
    monkeypatch.setattr(server, "sent", calls)
    return calls
