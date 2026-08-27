#!/usr/bin/env bash
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
#
# Thin wrapper that runs a stage in the OpenPI venv with the repo on PYTHONPATH.
#
# Usage:
#   OPENPI_VENV=/mnt/public/jxqiu/.venv-openpi \
#     bash toolkits/train_infer_mismatch/run.sh dump --model-path <ckpt> -o artifact.pt
#   bash toolkits/train_infer_mismatch/run.sh run --model-path <ckpt> --artifact artifact.pt -o out.pt
#   bash toolkits/train_infer_mismatch/run.sh compare out_4090.pt out_a100.pt
set -euo pipefail

STAGE="${1:?usage: run.sh <dump|run|compare> [args...]}"
shift

VENV="${OPENPI_VENV:-/mnt/public/jxqiu/.venv-openpi}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYBIN="${VENV}/bin/python"

if [[ ! -x "${PYBIN}" ]]; then
    echo "python not found at ${PYBIN}; set OPENPI_VENV to your OpenPI venv." >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"

exec "${PYBIN}" -m "toolkits.train_infer_mismatch.${STAGE}" "$@"
