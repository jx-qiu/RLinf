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
"""Train-inference mismatch diagnostic for OpenPI (pi-0 / pi-0.5).

Three stages, meant to run standalone in the OpenPI venv:

* ``dump``    - build the policy, run one rollout forward, and freeze the exact
  inputs (noise ``chains`` / ``denoise_inds`` / observation / tokenized prompt)
  plus the rollout log-probs into a portable artifact.
* ``run``     - on each machine (e.g. RTX 4090 and A100), replay the training
  recompute path (``default_forward`` -> ``get_log_prob_value``) on the frozen
  inputs and save the recomputed log-probs plus a hardware fingerprint.
* ``compare`` - diff two ``run`` outputs to quantify the PPO importance ratio
  ``exp(logprob_a - logprob_b)`` induced purely by the hardware difference.
"""

TOOL_VERSION = 1
