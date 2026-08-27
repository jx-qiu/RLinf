# Train-inference mismatch diagnostic (OpenPI / pi-0.5)

Quantify whether running pi-05 **inference (rollout) on one GPU** (e.g. RTX 4090)
and **training on another** (e.g. A100) introduces a train-inference log-prob
mismatch that corrupts the PPO importance ratio.

In RLinf embodied PPO the OpenPI configs set `algorithm.recompute_logprobs: False`,
so the PPO *old* log-prob comes from the rollout while the *current* log-prob is
recomputed by the actor via `default_forward` -> `get_log_prob_value`. At the
first inner step (rollout weights == actor weights) the ratio
`exp(logprob_train - logprob_rollout)` should be ~1; hardware differences push it
away from 1. This tool freezes one batch and replays the recompute path on each
GPU so the only variable is the hardware.

## Layout

- `dump.py` — freeze a fixed rollout batch (noise `chains`/`denoise_inds`,
  observation, tokenized prompt, rollout log-probs) into `artifact.pt`.
- `run.py` — rebuild the model and recompute log-probs on the local GPU.
- `compare.py` — diff two `run` outputs and report the induced PPO ratio.
- `common.py` — model build, synthetic obs, AMP context, hashing, fingerprint.
- `run.sh` — launch a stage in the OpenPI venv.

Run everything in the OpenPI venv (`/mnt/public/jxqiu/.venv-openpi` or
`.venv-opi-libero`). `run.sh` sets that up; or `source switch_env openpi` first.

## Presets

- `pi05_libero` — LIBERO dims (state 8, main + 1 wrist, action 7, `pi05_libero`).
- `pi05_behavior` — BEHAVIOR R1Pro (full proprio state, main + 2 wrist views
  stacked as `[B, 2, H, W, 3]`, action 23, 32 chunks, `pi05_behavior`). Mirrors
  `examples/embodiment/config/behavior_ppo_openpi_pi05.yaml`.

For maximum fidelity pass `--config path/to/logs/<run>/tensorboard/config.yaml`
instead of `--preset` (it reuses your exact `actor.model`).

## Usage

```bash
export OPENPI_VENV=/mnt/public/jxqiu/.venv-openpi
CKPT=/path/to/RLinf-Pi05-LIBERO-SFT

# 1) On any machine: freeze one batch. --bundle-weights makes the artifact
#    self-contained (~several GB) so a machine without the checkpoint can run it.
bash toolkits/train_infer_mismatch/run.sh dump \
    --preset pi05_libero --model-path "$CKPT" --bundle-weights \
    --batch-size 8 --seed 0 -o artifact.pt

# 2) On the 4090, then copy artifact.pt to the A100 box and run there too.
bash toolkits/train_infer_mismatch/run.sh run \
    --preset pi05_libero --model-path "$CKPT" --artifact artifact.pt -o out_4090.pt
bash toolkits/train_infer_mismatch/run.sh run \
    --preset pi05_libero --model-path "$CKPT" --artifact artifact.pt -o out_a100.pt

# 3) Anywhere: compare.
bash toolkits/train_infer_mismatch/run.sh compare out_4090.pt out_a100.pt --json report.json
```

## Deterministic policies (`noise_level = 0`)

Some configs (e.g. `behavior_ppo_openpi_pi05.yaml`) set `openpi.noise_level: 0.0`.
With flow-SDE this makes `x_t_std = 0`, so `get_logprob_norm` returns **exactly
zero** for every action: the policy is deterministic and the PPO log-prob ratio
is identically `exp(0-0) = 1` on any hardware. The tool detects this and reports
verdict `DEGENERATE` instead of a misleading `PASS` — hardware placement cannot
introduce a mismatch through the (inert) log-prob channel.

So that the log-prob channel is non-degenerate by default, `dump` injects
`--noise-level 0.3` (kept identical at run time — `run` inherits it from the
artifact automatically). Pass `--noise-level 0` to reproduce the raw
deterministic policy, or another value to match your config:

```bash
bash toolkits/train_infer_mismatch/run.sh dump --preset pi05_behavior \
    --model-path "$CKPT" -o artifact.pt            # noise_level 0.3 by default
bash toolkits/train_infer_mismatch/run.sh dump --preset pi05_behavior \
    --model-path "$CKPT" --noise-level 0 -o artifact.pt   # raw deterministic policy
```

Measured 4090 baseline (BEHAVIOR pi-05, `noise_level 0.3`, batch 2): same-GPU
recompute is **bit-identical** (`max_abs_diff = 0`), and the intrinsic
rollout-vs-recompute gap is `mean|ratio-1| = 4.3e-3`. Judge an A100 run against
these: a 4090-vs-A100 recompute diff on the order of the same-GPU floor means the
placement is safe.

## Interpreting the result

- The verdict is on `mean|ratio-1|` of `exp(logprob_B - logprob_A)`.
- Do **not** judge against a literal 1.0. The OpenPI rollout path differs from the
  recompute path, so a small gap exists even on identical hardware. **Establish a
  same-GPU baseline** (run stage 2 twice on two A100s, or twice on one GPU) and
  compare the 4090-vs-A100 number against it. If they match, the placement adds
  no mismatch.
- `compare` prints each run's rollout-vs-recompute gap separately (the
  algorithmic component) so you can isolate it from the hardware component.
- Keep `--tf32` and `--amp` identical on both machines. `--tf32 default` leaves
  torch defaults untouched (most faithful to training); the realized flags are
  recorded in each output's fingerprint.

## Guards

Each `run` output stores the input hash and a weight-hash fingerprint. `compare`
aborts if either differs (unless `--allow-hash-mismatch`), catching the most
common false result: accidentally different inputs or checkpoints.
