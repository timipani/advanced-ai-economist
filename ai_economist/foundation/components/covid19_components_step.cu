// Copyright (c) 2021, salesforce.com, inc.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// For full license text, see the LICENSE file in the repo root
// or https://opensource.org/licenses/BSD-3-Clause

extern "C" {
    // CUDA version of the components in
    // "ai_economist.foundation.components.covid19_components.py"
    __global__ void CudaControlUSStateOpenCloseStatusStep(
        int * stringency_level,
        const int kActionCooldownPeriod,
        int * action_in_cooldown_until,
        const int * kDefaultAgentActionMask,
        const int * kNoOpAgentActionMask,
        const int kNumStringencyLeve