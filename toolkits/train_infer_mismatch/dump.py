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
"""Stage 1: freeze a fixed rollout batch into a portable artifact.

Builds the pi-05 policy, runs one ``predict_action_batch`` (the rollout path
that produces the noise ``chains`` / ``denoise_inds`` and rollout log-probs),
and saves everything the training recompute path needs. Run this once on any
machine; copy the artifact to every machine you want to compare.

Example:
    python -m toolkits.train_infer_mismatch.dump \
        --preset pi05_libero --model-path /path/to/RLinf-Pi05-SFT \
        --bundle-weights -o artifact.pt
"""

from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf

from rlinf.utils.nested_dict_process import clone_nested_to_cpu
from toolkits.train_infer_mismatch import TOOL_VERSION
from toolkits.train_infer_mismatch.common import (
    batch_concat,
    batch_slice,
    build_model,
    get_effective_noise_level,
    get_fingerprint,
    hash_nested,
    hash_state_dict,
    infer_batch_size,
    load_model_cfg,
    make_synthetic_env_obs,
    resolve_obs_params,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--config", type=str, default=None,
                     help="Resolved config yaml (e.g. logs/<run>/tensorboard/config.yaml).")
    src.add_argument("--preset", type=str, default="pi05_libero",
                     help="Built-in model preset when --config is not given.")
    p.add_argument("--model-path", type=str, default=None,
                   help="pi-05 checkpoint directory (overrides the config).")
    p.add_argument("--num-action-chunks", type=int, default=None)
    p.add_argument("--num-steps", type=int, default=None)
    p.add_argument("--noise-level", type=float, default=0.3,
                   help="Flow-SDE noise_level to inject (default 0.3). Overrides the "
                        "config value so the logprobs are non-degenerate; pass 0 to "
                        "reproduce a deterministic (noise_level=0) policy. run inherits "
                        "whatever value is recorded in the artifact.")
    # Synthetic observation controls (only used at dump time).
    p.add_argument("--batch-size", type=int, default=32,
                   help="Number of frozen inputs to carry in the artifact.")
    p.add_argument("--micro-batch-size", type=int, default=8,
                   help="Forward chunk size so large --batch-size fits in GPU memory.")
    p.add_argument("--state-dim", type=int, default=None)
    p.add_argument("--image-hw", type=int, default=None)
    p.add_argument("--num-images", type=int, default=None)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["train", "eval"], default="train",
                   help="Rollout mode; 'train' matches RL rollout (stochastic + logprobs).")
    p.add_argument("--real-obs", type=str, default=None,
                   help="Optional .pt file with a saved env_obs dict (instead of synthetic).")
    p.add_argument("--bundle-weights", action="store_true",
                   help="Embed the model state_dict in the artifact (portable but large).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("-o", "--output", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    model_cfg, amp_cfg = load_model_cfg(args)
    obs_params = resolve_obs_params(args, model_cfg)

    print(f"[dump] building model from {model_cfg.model_path}")
    model = build_model(model_cfg, device=args.device)

    if args.real_obs:
        print(f"[dump] loading real env_obs from {args.real_obs}")
        # Trusted local artifact produced by this tool / the user.
        env_obs = torch.load(args.real_obs, map_location="cpu", weights_only=False)
    else:
        print(f"[dump] synthesizing env_obs {obs_params}")
        env_obs = make_synthetic_env_obs(**obs_params)

    total_b = infer_batch_size(env_obs)
    micro_bs = max(1, args.micro_batch_size)
    print(f"[dump] rollout: {total_b} inputs in chunks of {micro_bs}")
    fi_list, lp_list, val_list, act_list = [], [], [], []
    for s in range(0, total_b, micro_bs):
        e = min(s + micro_bs, total_b)
        chunk = batch_slice(env_obs, s, e)
        with torch.no_grad():
            act, res = model.predict_action_batch(chunk, mode=args.mode)
        fi_list.append(clone_nested_to_cpu(res["forward_inputs"]))
        lp_list.append(clone_nested_to_cpu(res.get("prev_logprobs")))
        val_list.append(clone_nested_to_cpu(res.get("prev_values")))
        act_list.append(clone_nested_to_cpu(act))

    forward_inputs = batch_concat(fi_list)
    prev_logprobs = batch_concat(lp_list)
    prev_values = batch_concat(val_list)
    actions = batch_concat(act_list)
    input_hash = hash_nested(forward_inputs)
    noise_level = get_effective_noise_level(model_cfg)
    degenerate = bool(
        isinstance(prev_logprobs, torch.Tensor) and torch.count_nonzero(prev_logprobs) == 0
    )

    artifact = {
        "tool_version": TOOL_VERSION,
        "forward_inputs": forward_inputs,
        "prev_logprobs": prev_logprobs,
        "prev_values": prev_values,
        "actions": actions,
        "input_hash": input_hash,
        "meta": {
            "model_cfg": OmegaConf.to_container(model_cfg, resolve=True),
            "amp_cfg": OmegaConf.to_container(amp_cfg, resolve=True)
            if amp_cfg is not None
            else None,
            "obs_params": None if args.real_obs else obs_params,
            "mode": args.mode,
            "batch_size": total_b,
            "noise_level": noise_level,
            "degenerate_logprobs": degenerate,
            "dump_fingerprint": get_fingerprint("dump", "n/a"),
        },
    }

    sd = model.state_dict()
    artifact["weight_hash"] = hash_state_dict(sd)
    if args.bundle_weights:
        print("[dump] bundling weights (this makes the artifact several GB)")
        artifact["weights"] = clone_nested_to_cpu(sd)

    torch.save(artifact, args.output)
    print(f"[dump] wrote {args.output}")
    print(f"[dump] input_hash={input_hash}")
    print(f"[dump] weight_hash={artifact['weight_hash']}")
    print(f"[dump] noise_level={noise_level}")
    if isinstance(prev_logprobs, torch.Tensor):
        print(f"[dump] rollout prev_logprobs shape={tuple(prev_logprobs.shape)}")
    if degenerate:
        print("[dump] WARNING: rollout logprobs are all zero -> deterministic "
              "policy (noise_level=0). The PPO logprob ratio is trivially 1 on "
              "any hardware. Re-dump with --noise-level 0.3 to probe stochastic "
              "logprob hardware sensitivity.")


if __name__ == "__main__":
    main()
