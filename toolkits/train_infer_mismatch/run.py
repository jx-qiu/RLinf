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
"""Stage 2: recompute log-probs from a frozen artifact on the local GPU.

Rebuilds the pi-05 policy and replays the training recompute path
(``default_forward`` -> ``get_log_prob_value``) on the frozen ``chains`` /
``denoise_inds`` from the artifact, under the same AMP context the actor uses.
Run once per machine (RTX 4090, then A100) and keep the two output files.

Example:
    python -m toolkits.train_infer_mismatch.run \
        --preset pi05_libero --model-path /path/to/RLinf-Pi05-SFT \
        --artifact artifact.pt -o out_4090.pt
"""

from __future__ import annotations

import argparse

import torch

from rlinf.utils.nested_dict_process import clone_nested_to_cpu, put_tensor_device
from toolkits.train_infer_mismatch import TOOL_VERSION
from toolkits.train_infer_mismatch.common import (
    apply_runtime_flags,
    batch_slice,
    build_amp_context,
    build_model,
    get_fingerprint,
    hash_nested,
    hash_state_dict,
    infer_batch_size,
    load_model_cfg,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--config", type=str, default=None,
                     help="Resolved config yaml; must match the one used at dump time.")
    src.add_argument("--preset", type=str, default="pi05_libero")
    p.add_argument("--model-path", type=str, default=None,
                   help="pi-05 checkpoint directory (overrides the config).")
    p.add_argument("--num-action-chunks", type=int, default=None)
    p.add_argument("--num-steps", type=int, default=None)
    p.add_argument("--noise-level", type=float, default=None,
                   help="Override flow-SDE noise_level. If unset, inherits the "
                        "value stored in the artifact so recompute sigma matches the dump.")
    p.add_argument("--artifact", type=str, required=True)
    p.add_argument("--ignore-bundled-weights", action="store_true",
                   help="Load weights from --model-path even if the artifact bundles them.")
    p.add_argument("--compute-values", action="store_true",
                   help="Also run the value path (does not affect log-probs).")
    p.add_argument("--micro-batch-size", type=int, default=8,
                   help="Forward chunk size so large artifacts fit in GPU memory.")
    p.add_argument("--amp", choices=["config", "off", "bf16", "fp16", "fp32"],
                   default="config", help="Autocast override; 'config' follows the training config.")
    p.add_argument("--tf32", choices=["default", "on", "off"], default="default",
                   help="TF32 policy; keep IDENTICAL on both machines.")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("-o", "--output", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    apply_runtime_flags(args.tf32, args.deterministic, args.seed)

    # Trusted local artifact produced by the dump stage.
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)

    # Keep the recompute sigma identical to the dump: inherit noise_level from
    # the artifact unless the user explicitly overrides it.
    if args.noise_level is None:
        args.noise_level = artifact.get("meta", {}).get("noise_level", None)

    model_cfg, amp_cfg = load_model_cfg(args)

    # Guard: the frozen inputs must be intact on this machine.
    recomputed_input_hash = hash_nested(artifact["forward_inputs"])
    if recomputed_input_hash != artifact["input_hash"]:
        raise RuntimeError(
            "Artifact input_hash mismatch after load; the file may be corrupt.\n"
            f"  expected {artifact['input_hash']}\n  got      {recomputed_input_hash}"
        )

    weights = None
    if artifact.get("weights") is not None and not args.ignore_bundled_weights:
        print("[run] using weights bundled in the artifact")
        weights = artifact["weights"]

    print(f"[run] building model from {model_cfg.model_path}")
    model = build_model(model_cfg, device=args.device, weights_state_dict=weights)

    amp_ctx, amp_label = build_amp_context(amp_cfg, override=args.amp)
    forward_inputs_cpu = artifact["forward_inputs"]
    total_b = infer_batch_size(forward_inputs_cpu)
    micro_bs = max(1, args.micro_batch_size)

    print(f"[run] recomputing log-probs: {total_b} inputs in chunks of {micro_bs} "
          f"(amp={amp_label}, compute_values={args.compute_values})")
    lp_list, val_list, ent_list = [], [], []
    for s in range(0, total_b, micro_bs):
        e = min(s + micro_bs, total_b)
        fi = put_tensor_device(batch_slice(forward_inputs_cpu, s, e), args.device)
        with torch.no_grad(), amp_ctx:
            out = model(
                forward_inputs=fi,
                compute_logprobs=True,
                compute_entropy=False,
                compute_values=args.compute_values,
                use_cache=False,
            )
        lp_list.append(out["logprobs"].detach().float().cpu())
        v, ent = out.get("values"), out.get("entropy")
        val_list.append(v.detach().float().cpu() if torch.is_tensor(v) else None)
        ent_list.append(ent.detach().float().cpu() if torch.is_tensor(ent) else None)

    logprobs = torch.cat(lp_list, dim=0)
    values = torch.cat(val_list, dim=0) if val_list and val_list[0] is not None else None
    entropy = torch.cat(ent_list, dim=0) if ent_list and ent_list[0] is not None else None
    weight_hash = hash_state_dict(model.state_dict())
    degenerate = bool(torch.count_nonzero(logprobs) == 0)

    result = {
        "tool_version": TOOL_VERSION,
        "logprobs": logprobs,
        "values": values,
        "entropy": entropy,
        "prev_logprobs": clone_nested_to_cpu(artifact.get("prev_logprobs")),
        "input_hash": artifact["input_hash"],
        "weight_hash": weight_hash,
        "fingerprint": get_fingerprint(str(logprobs.dtype), amp_label),
        "compute_values": args.compute_values,
        "noise_level": args.noise_level,
        "micro_batch_size": micro_bs,
        "degenerate_logprobs": degenerate,
    }
    torch.save(result, args.output)

    fp = result["fingerprint"]
    print(f"[run] wrote {args.output}")
    print(f"[run] gpu={fp['gpu_name']} sm={fp['sm']} torch={fp['torch']} "
          f"cuda={fp['cuda']} tf32_matmul={fp['allow_tf32_matmul']}")
    print(f"[run] logprobs shape={tuple(logprobs.shape)} "
          f"mean={logprobs.mean().item():.6f} "
          f"nonzero={int(torch.count_nonzero(logprobs))}/{logprobs.numel()}")
    print(f"[run] weight_hash={weight_hash}")
    if degenerate:
        print("[run] WARNING: recomputed logprobs are all zero -> deterministic "
              "policy (noise_level=0). Re-dump/-run with --noise-level 0.3 to "
              "probe stochastic logprob hardware sensitivity.")


if __name__ == "__main__":
    main()
