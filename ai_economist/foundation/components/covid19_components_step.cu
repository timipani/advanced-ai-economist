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
        const int kNumStringencyLevels,
        int * actions,
        float * obs_a_stringency_policy_indicators,
        float * obs_a_action_mask,
        float * obs_p_stringency_policy_indicators,
        int * env_timestep_arr,
        const int kNumAgents,
        const int kEpisodeLength
    ) {
        const int kEnvId = blockIdx.x;
        const int kAgentId = threadIdx.x;

        // Increment time ONCE -- only 1 thread can do this.
        if (kAgentId == 0) {
            env_timestep_arr[kEnvId] += 1;
        }

        // Wait here until timestep has been updated
        __syncthreads();

        assert(env_timestep_arr[kEnvId] > 0 &&
            env_timestep_arr[kEnvId] <= kEpisodeLength);
        assert (kAgentId <= kNumAgents - 1);

        // Update the stringency levels for the US states
        if (kAgentId < (kNumAgents - 1)) {
            // Indices for time-dependent and time-independent arrays
            // Time dependent arrays have shapes
            // (num_envs, kEpisodeLength + 1, kNumAgents - 1)
            // Time independent arrays have shapes (num_envs, kNumAgents - 1)
            const int