# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared helpers for the train-inference mismatch diagnostic tool.

This module deliberately avoids importing heavy deps (``rlinf`` / ``openpi``)
at module load time so ``compare`` can run in any Python environment. The model
build helpers import ``rlinf.models.embodiment.openpi`` lazily.
"""

from __future__ import annotations

import copy
import hashlib
import os
import platform
import socket
from contextlib import nullcontext
from typing import Any, ContextManager, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

DEFAULT_ROBOT_PLATFORM = "LIBERO"

# Placeholder ``model_path`` values used by the example configs; treated as
# "unset" so the tool fails fast with a helpful message instead of downloading.
_PLACEHOLDER_MODEL_PATHS = {
    "",
    "/path/to/model/openpi",
    "/path/to/model/RLinf-Pi05-SFT",
}

# Built-in presets mirror the shipped example configs so the tool is usable
# without a resolved Hydra config. Values match
# ``examples/embodiment/config/model/pi0_5.yaml`` + the LIBERO overrides in
# ``examples/embodiment/config/libero_object_ppo_openpi_pi05.yaml``.
PRESETS: dict[str, dict[str, Any]] = {
    "pi05_libero": {
        "model_cfg": {
            "model_type": "openpi",
            "model_path": None,
            "precision": None,
            "num_action_chunks": 5,
            "action_dim": 7,
            "is_lora": False,
            "lora_rank": 32,
            "use_proprio": True,
            "num_steps": 5,
            "add_value_head": True,
            "openpi": {
                "config_name": "pi05_libero",
                "num_images_in_input": 2,
                "noise_level": 0.3,
                "action_chunk": 5,
                "num_steps": 5,
                "train_expert_only": True,
                "action_env_dim": 7,
                "noise_method": "flow_sde",
                "add_value_head": True,
                "value_after_vlm": True,
                "value_vlm_mode": "mean_token",
                "use_dsrl": False,
            },
        },
        "obs": {
            "state_dim": 8,
            "image_hw": 224,
            "num_images_in_input": 2,
            "prompt": "do something",
        },
    },
    # Mirrors examples/embodiment/config/behavior_ppo_openpi_pi05.yaml (R1Pro).
    # BEHAVIOR uses config_name pi05_behavior with extract_state_from_proprio=True,
    # so `states` must be the full proprio vector (indices up to 256), and
    # `wrist_images` stacks the two wrist views as [B, 2, H, W, 3].
    "pi05_behavior": {
        "model_cfg": {
            "model_type": "openpi",
            "model_path": None,
            "precision": None,
            "num_action_chunks": 32,
            "action_dim": 23,
            "proprio_dim": 32,
            "num_images_in_input": 3,
            "is_lora": False,
            "lora_rank": 32,
            "use_proprio": True,
            "num_steps": 10,
            "add_value_head": True,
            "openpi": {
                "config_name": "pi05_behavior",
                "num_images_in_input": 3,
                "noise_level": 0.0,
                "action_chunk": 32,
                "num_steps": 10,
                "train_expert_only": True,
                "action_env_dim": 23,
                "noise_method": "flow_sde",
                "add_value_head": True,
                "detach_critic_input": True,
                "joint_logprob": False,
            },
        },
        "obs": {
            "state_dim": 256,
            "image_hw": 224,
            "num_images_in_input": 3,
            "prompt": "do something",
        },
    },
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_model_cfg(args) -> tuple[DictConfig, Optional[DictConfig]]:
    """Resolve the OpenPI ``actor.model`` config plus its AMP settings.

    Priority: an explicit ``--config`` (a *resolved* config yaml, e.g. the one
    saved under ``logs/<run>/tensorboard/config.yaml``) is most faithful to the
    real training run; otherwise a built-in ``--preset`` is used.

    Returns:
        Tuple of (model_cfg, amp_cfg). ``amp_cfg`` may be ``None``.
    """
    amp_cfg: Optional[DictConfig] = None
    if getattr(args, "config", None):
        cfg_all = OmegaConf.load(args.config)
        if OmegaConf.select(cfg_all, "actor.model") is not None:
            model_cfg = OmegaConf.select(cfg_all, "actor.model")
            amp_cfg = OmegaConf.select(cfg_all, "actor.fsdp_config.amp_autocast")
        elif OmegaConf.select(cfg_all, "model.model_type") is not None:
            model_cfg = cfg_all.model
        elif OmegaConf.select(cfg_all, "model_type") is not None:
            model_cfg = cfg_all
        else:
            raise ValueError(
                f"Could not find `actor.model` in {args.config!r}. Pass a "
                "resolved config (e.g. logs/<run>/tensorboard/config.yaml) or "
                "use --preset."
            )
        # Materialize a standalone, fully-resolved copy.
        model_cfg = OmegaConf.create(
            OmegaConf.to_container(model_cfg, resolve=True)
        )
    else:
        preset = PRESETS[args.preset]
        model_cfg = OmegaConf.create(copy.deepcopy(preset["model_cfg"]))

    _apply_model_overrides(model_cfg, args)

    model_path = model_cfg.get("model_path", None)
    if model_path in _PLACEHOLDER_MODEL_PATHS or model_path is None:
        raise ValueError(
            "model_path is not set (or is a placeholder). Pass --model-path "
            "with your pi-05 checkpoint directory."
        )
    return model_cfg, amp_cfg


def _apply_model_overrides(model_cfg: DictConfig, args) -> None:
    if getattr(args, "model_path", None):
        model_cfg.model_path = args.model_path
    if getattr(args, "num_action_chunks", None):
        model_cfg.num_action_chunks = args.num_action_chunks
        if "openpi" in model_cfg:
            model_cfg.openpi.action_chunk = args.num_action_chunks
    if getattr(args, "num_steps", None):
        model_cfg.num_steps = args.num_steps
        if "openpi" in model_cfg:
            model_cfg.openpi.num_steps = args.num_steps
    # noise_level controls the flow-SDE stochasticity. When it is 0 the policy is
    # deterministic and the action log-probs are identically 0 (so the PPO ratio
    # is trivially 1). Override it to a positive value to probe the stochastic
    # log-prob hardware sensitivity; keep it IDENTICAL at dump and run time.
    if getattr(args, "noise_level", None) is not None and "openpi" in model_cfg:
        model_cfg.openpi.noise_level = args.noise_level


def get_effective_noise_level(model_cfg: DictConfig) -> Optional[float]:
    val = OmegaConf.select(model_cfg, "openpi.noise_level")
    return None if val is None else float(val)


def resolve_obs_params(args, model_cfg: DictConfig) -> dict[str, Any]:
    """Resolve synthetic-observation parameters from args + preset defaults."""
    preset_name = getattr(args, "preset", None) or "pi05_libero"
    defaults = PRESETS.get(preset_name, PRESETS["pi05_libero"])["obs"]
    cfg_num_images = OmegaConf.select(model_cfg, "openpi.num_images_in_input")
    return {
        "batch_size": args.batch_size,
        "state_dim": args.state_dim or defaults["state_dim"],
        "image_hw": args.image_hw or defaults["image_hw"],
        "num_images": args.num_images
        or int(cfg_num_images or defaults["num_images_in_input"]),
        "prompt": args.prompt or defaults["prompt"],
        "seed": args.seed,
    }


# ---------------------------------------------------------------------------
# Model build + synthetic observation
# ---------------------------------------------------------------------------
def build_model(
    model_cfg: DictConfig,
    device: str = "cuda",
    weights_state_dict: Optional[dict] = None,
):
    """Build the OpenPI policy exactly as the actor worker does at train time."""
    os.environ.setdefault("ROBOT_PLATFORM", DEFAULT_ROBOT_PLATFORM)
    from rlinf.models.embodiment.openpi import get_model

    model = get_model(model_cfg, torch_dtype=None)
    if weights_state_dict is not None:
        missing, unexpected = model.load_state_dict(
            weights_state_dict, strict=False
        )
        if missing:
            print(f"[build_model] {len(missing)} missing keys when loading "
                  f"bundled weights (first few: {list(missing)[:3]})")
        if unexpected:
            print(f"[build_model] {len(unexpected)} unexpected keys when "
                  f"loading bundled weights (first few: {list(unexpected)[:3]})")
    return model.to(device).eval()


def make_synthetic_env_obs(
    batch_size: int,
    state_dim: int,
    image_hw: int,
    num_images: int,
    prompt: str,
    seed: int,
) -> dict[str, Any]:
    """Build a fixed-seed ``env_obs`` matching the OpenPI obs contract.

    Keys mirror ``OpenPi0ForRLActionPrediction.obs_processor``. ``main_images``
    is ``[B, H, W, 3]`` uint8 and ``states`` is ``[B, state_dim]`` float32.
    The wrist layout is derived from ``num_images`` (main + wrist views):
    a single wrist view (e.g. LIBERO) is ``[B, H, W, 3]`` while two or more
    (e.g. BEHAVIOR left+right) are stacked as ``[B, n_wrist, H, W, 3]`` to match
    the env. ``wrist_images``/``extra_view_images`` must exist (even as ``None``)
    because the processor indexes them directly.
    """
    rng = np.random.default_rng(seed)
    main = rng.integers(0, 256, size=(batch_size, image_hw, image_hw, 3), dtype=np.uint8)
    obs: dict[str, Any] = {
        "main_images": main,
        "task_descriptions": [prompt] * batch_size,
        "wrist_images": None,
        "extra_view_images": None,
        "states": rng.standard_normal((batch_size, state_dim)).astype(np.float32),
    }
    n_wrist = max(0, num_images - 1)
    if n_wrist == 1:
        obs["wrist_images"] = rng.integers(
            0, 256, size=(batch_size, image_hw, image_hw, 3), dtype=np.uint8
        )
    elif n_wrist >= 2:
        obs["wrist_images"] = rng.integers(
            0, 256, size=(batch_size, n_wrist, image_hw, image_hw, 3), dtype=np.uint8
        )
    return obs


# ---------------------------------------------------------------------------
# Batch chunking (so large --batch-size fits in GPU memory)
# ---------------------------------------------------------------------------
def infer_batch_size(nested: dict) -> int:
    """Infer the leading batch dim from an env_obs or forward_inputs dict."""
    for key in ("main_images", "chains", "observation/image", "tokenized_prompt"):
        v = nested.get(key)
        if isinstance(v, (torch.Tensor, np.ndarray)):
            return int(v.shape[0])
    td = nested.get("task_descriptions")
    if isinstance(td, list):
        return len(td)
    raise ValueError("could not infer batch size from nested dict")


def batch_slice(obj: Any, start: int, end: int) -> Any:
    """Slice the leading (batch) dim of tensors/arrays/lists in a nested obj."""
    if obj is None:
        return None
    if isinstance(obj, (torch.Tensor, np.ndarray, list)):
        return obj[start:end]
    if isinstance(obj, dict):
        return {k: batch_slice(v, start, end) for k, v in obj.items()}
    return obj  # scalars/strings are shared across the batch


def batch_concat(objs: list) -> Any:
    """Concatenate a list of same-structure nested objs along the batch dim."""
    first = objs[0]
    if first is None:
        return None
    if isinstance(first, torch.Tensor):
        return torch.cat(list(objs), dim=0)
    if isinstance(first, np.ndarray):
        return np.concatenate(list(objs), axis=0)
    if isinstance(first, list):
        out: list = []
        for o in objs:
            out.extend(o)
        return out
    if isinstance(first, dict):
        return {k: batch_concat([o[k] for o in objs]) for k in first}
    return first


# ---------------------------------------------------------------------------
# AMP context (mirror the actor's ``self.amp_context``)
# ---------------------------------------------------------------------------
_AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def build_amp_context(
    amp_cfg: Optional[DictConfig], override: str = "config"
) -> tuple[ContextManager, str]:
    """Return the autocast context to wrap ``default_forward``, plus a label.

    ``override`` is one of ``config`` (follow ``amp_cfg``; OpenPI default is
    disabled), ``off``, ``bf16``, ``fp16`` or ``fp32``.
    """
    if override == "off":
        return nullcontext(), "off"
    if override in ("bf16", "fp16"):
        return torch.amp.autocast("cuda", dtype=_AMP_DTYPES[override]), override
    if override == "fp32":
        return nullcontext(), "fp32"
    # override == "config"
    enabled = bool(amp_cfg.get("enabled", False)) if amp_cfg is not None else False
    if not enabled:
        return nullcontext(), "config:off"
    prec = str(amp_cfg.get("precision", "bf16"))
    if prec == "fp32":
        return nullcontext(), "config:fp32"
    return torch.amp.autocast("cuda", dtype=_AMP_DTYPES[prec]), f"config:{prec}"


def apply_runtime_flags(tf32: str, deterministic: bool, seed: int) -> None:
    """Set seeds and precision/determinism flags before building the model.

    ``tf32`` is ``default`` (leave torch defaults untouched, most faithful to
    training), ``on`` or ``off``. Whatever is chosen must be identical on both
    machines; the actual realized values are recorded in the fingerprint.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if tf32 in ("on", "off"):
        allow = tf32 == "on"
        torch.backends.cuda.matmul.allow_tf32 = allow
        torch.backends.cudnn.allow_tf32 = allow
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:  # pragma: no cover - best effort
            print(f"[apply_runtime_flags] deterministic algorithms unavailable: {exc}")


def get_fingerprint(forward_dtype_label: str, amp_label: str) -> dict[str, Any]:
    """Capture the hardware/software environment for reproducibility checks."""
    dev = torch.cuda.current_device()
    cap = torch.cuda.get_device_capability(dev)
    return {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(dev),
        "sm": f"{cap[0]}{cap[1]}",
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "amp": amp_label,
        "forward_dtype": forward_dtype_label,
    }


# ---------------------------------------------------------------------------
# Hashing (prove both machines used identical inputs / weights)
# ---------------------------------------------------------------------------
def _update_bytes(h: "hashlib._Hash", raw: bytes, sample: Optional[int]) -> None:
    h.update(len(raw).to_bytes(8, "little"))
    if sample is not None and len(raw) > 2 * sample:
        h.update(raw[:sample])
        h.update(raw[-sample:])
    else:
        h.update(raw)


def _walk(h: "hashlib._Hash", key: Any, obj: Any, sample: Optional[int]) -> None:
    h.update(str(key).encode())
    if obj is None:
        h.update(b"None")
    elif isinstance(obj, torch.Tensor):
        t = obj.detach().cpu().contiguous()
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        arr = (t.to(torch.float32) if t.is_floating_point() else t.to(torch.int64)).numpy()
        _update_bytes(h, arr.tobytes(), sample)
    elif isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        _update_bytes(h, arr.tobytes(), sample)
    elif isinstance(obj, dict):
        for k in sorted(obj.keys(), key=str):
            _walk(h, k, obj[k], sample)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _walk(h, i, v, sample)
    else:
        h.update(repr(obj).encode())


def hash_nested(obj: Any, sample: Optional[int] = None) -> str:
    """SHA-256 over a nested structure of tensors/arrays/scalars (order-stable).

    ``sample`` limits large byte blobs to the first+last N bytes (a fast
    fingerprint); ``None`` hashes everything.
    """
    h = hashlib.sha256()
    _walk(h, "root", obj, sample)
    return h.hexdigest()


def hash_state_dict(state_dict: dict, sample: Optional[int] = 4096) -> str:
    """Fast fingerprint of model weights (name + shape + dtype + sampled bytes)."""
    h = hashlib.sha256()
    for name in sorted(state_dict.keys()):
        _walk(h, name, state_dict[name], sample)
    return h.hexdigest()
