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
"""Stage 3: diff two ``run`` outputs to quantify the hardware-induced mismatch.

Reports the log-prob difference and the induced PPO importance ratio
``exp(logprob_b - logprob_a)`` between two machines that recomputed on the
*same* frozen inputs and weights. Interpret the numbers against a same-GPU
baseline (A100-vs-A100), not against a literal 1.0: the OpenPI rollout uses a
different code path from the training recompute, so a small gap exists even on
identical hardware.

Example:
    python -m toolkits.train_infer_mismatch.compare out_4090.pt out_a100.pt
"""

from __future__ import annotations

import argparse
import json

import torch

# Heuristic thresholds on mean|ratio-1|; tune to your tolerance and always
# compare the cross-hardware value against a same-hardware baseline run.
DEFAULT_PASS = 0.02
DEFAULT_WARN = 0.10


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_a", type=str, help="First run output (e.g. out_4090.pt).")
    p.add_argument("output_b", type=str, help="Second run output (e.g. out_a100.pt).")
    p.add_argument("--allow-hash-mismatch", action="store_true",
                   help="Continue even if input/weight hashes differ (result is NOT a pure hardware diff).")
    p.add_argument("--pass-threshold", type=float, default=DEFAULT_PASS)
    p.add_argument("--warn-threshold", type=float, default=DEFAULT_WARN)
    p.add_argument("--json", type=str, default=None, help="Optional path to write a JSON summary.")
    return p.parse_args()


def _tensor_stats(diff: torch.Tensor) -> dict[str, float]:
    absd = diff.abs()
    return {
        "max_abs": absd.max().item(),
        "mean_abs": absd.mean().item(),
        "median_abs": absd.median().item(),
        "p99_abs": torch.quantile(absd.flatten().float(), 0.99).item(),
        "rms": diff.pow(2).mean().sqrt().item(),
    }


def _ratio_stats(diff: torch.Tensor) -> dict[str, float]:
    ratio = torch.exp(diff)
    dev = (ratio - 1.0).abs()
    return {
        "mean_abs_ratio_minus_1": dev.mean().item(),
        "max_abs_ratio_minus_1": dev.max().item(),
        "mean_logratio": diff.mean().item(),
        "mean_abs_logratio": diff.abs().mean().item(),
    }


def _fmt_fp(fp: dict) -> str:
    return (f"{fp.get('gpu_name', '?')} (sm{fp.get('sm', '?')}), "
            f"torch {fp.get('torch', '?')}/cuda {fp.get('cuda', '?')}, "
            f"tf32_matmul={fp.get('allow_tf32_matmul', '?')}, amp={fp.get('amp', '?')}")


def _check_hashes(a: dict, b: dict, allow: bool) -> list[str]:
    warnings = []
    if a.get("input_hash") != b.get("input_hash"):
        msg = "input_hash differs: the two runs used DIFFERENT frozen inputs."
        if not allow:
            raise SystemExit(f"[compare] ERROR: {msg} Use --allow-hash-mismatch to override.")
        warnings.append(msg)
    if a.get("weight_hash") != b.get("weight_hash"):
        msg = ("weight_hash differs: the two runs used DIFFERENT weights, so the "
               "diff is NOT a pure hardware effect.")
        if not allow:
            raise SystemExit(f"[compare] ERROR: {msg} Use --allow-hash-mismatch to override.")
        warnings.append(msg)
    return warnings


def _self_gap(out: dict) -> dict[str, float] | None:
    """Recompute-vs-rollout gap within one run (the algorithmic component)."""
    prev = out.get("prev_logprobs")
    if not isinstance(prev, torch.Tensor):
        return None
    lp = out["logprobs"].float()
    prev = prev.float()
    if prev.shape != lp.shape:
        return None
    return _ratio_stats(lp - prev)


def main() -> None:
    args = _parse_args()
    # Trusted local artifacts produced by the run stage.
    a = torch.load(args.output_a, map_location="cpu", weights_only=False)
    b = torch.load(args.output_b, map_location="cpu", weights_only=False)

    warnings = _check_hashes(a, b, args.allow_hash_mismatch)

    lp_a = a["logprobs"].float()
    lp_b = b["logprobs"].float()
    if lp_a.shape != lp_b.shape:
        raise SystemExit(f"[compare] logprob shape mismatch: {tuple(lp_a.shape)} vs {tuple(lp_b.shape)}")

    diff = lp_b - lp_a  # b relative to a
    tstats = _tensor_stats(diff)
    rstats = _ratio_stats(diff)

    both_zero = bool(
        torch.count_nonzero(lp_a) == 0 and torch.count_nonzero(lp_b) == 0
    )

    mean_dev = rstats["mean_abs_ratio_minus_1"]
    if both_zero:
        verdict = "DEGENERATE"
    elif mean_dev <= args.pass_threshold:
        verdict = "PASS"
    elif mean_dev <= args.warn_threshold:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    fp_a, fp_b = a.get("fingerprint", {}), b.get("fingerprint", {})

    print("=" * 74)
    print("Train-inference mismatch: hardware comparison (recompute log-probs)")
    print("=" * 74)
    print(f"  A: {args.output_a}")
    print(f"     {_fmt_fp(fp_a)}")
    print(f"  B: {args.output_b}")
    print(f"     {_fmt_fp(fp_b)}")
    print(f"  logprobs shape: {tuple(lp_a.shape)}")
    for w in warnings:
        print(f"  [!] {w}")
    print("-" * 74)
    print("  log-prob |B - A|:")
    print(f"    max={tstats['max_abs']:.3e}  mean={tstats['mean_abs']:.3e}  "
          f"median={tstats['median_abs']:.3e}  p99={tstats['p99_abs']:.3e}")
    print("  induced PPO ratio exp(B - A):")
    print(f"    mean|ratio-1|={rstats['mean_abs_ratio_minus_1']:.3e}  "
          f"max|ratio-1|={rstats['max_abs_ratio_minus_1']:.3e}")
    print(f"    mean logratio (approx_kl)={rstats['mean_logratio']:.3e}  "
          f"mean|logratio|={rstats['mean_abs_logratio']:.3e}")

    # Per-chunk breakdown for [B, num_chunks, action_dim] log-probs.
    if diff.dim() == 3:
        per_chunk = diff.abs().mean(dim=(0, 2))
        chunks = "  ".join(f"c{i}={v:.2e}" for i, v in enumerate(per_chunk.tolist()))
        print(f"  per-chunk mean|diff|: {chunks}")

    # Secondary: within-run rollout-vs-recompute gap (algorithmic component).
    gap_a, gap_b = _self_gap(a), _self_gap(b)
    if gap_a or gap_b:
        print("-" * 74)
        print("  rollout-vs-recompute gap within each run "
              "(recompute on that GPU vs rollout log-probs from the dump GPU):")
        if gap_a:
            print(f"    A: mean|ratio-1|={gap_a['mean_abs_ratio_minus_1']:.3e}")
        if gap_b:
            print(f"    B: mean|ratio-1|={gap_b['mean_abs_ratio_minus_1']:.3e}")
    print("-" * 74)
    print(f"  VERDICT: {verdict}  "
          f"(pass<={args.pass_threshold}, warn<={args.warn_threshold} on mean|ratio-1|)")
    if both_zero:
        print("  DEGENERATE: both runs have all-zero log-probs -> the policy is")
        print("  deterministic (noise_level=0), so the PPO log-prob ratio is")
        print("  identically 1 on any hardware and this channel cannot reveal a")
        print("  mismatch. Re-dump/-run with --noise-level 0.3 to probe stochastic")
        print("  log-prob hardware sensitivity.")
    else:
        print("  Note: compare against a same-GPU baseline run; absolute value is")
        print("  confounded by the rollout-vs-recompute algorithmic gap above.")
    print("=" * 74)

    if args.json:
        summary = {
            "verdict": verdict,
            "degenerate": both_zero,
            "logprob_diff": tstats,
            "ratio": rstats,
            "fingerprint_a": fp_a,
            "fingerprint_b": fp_b,
            "warnings": warnings,
            "self_gap_a": gap_a,
            "self_gap_b": gap_b,
        }
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[compare] wrote {args.json}")


if __name__ == "__main__":
    main()
