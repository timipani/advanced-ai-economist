// Copyright (c) 2021, salesforce.com, inc.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// For full license text, see the LICENSE file in the repo root
// or https://opensource.org/licenses/BSD-3-Clause

__constant__ float kEpsilon = 1.0e-10;  // used to prevent division by 0

extern "C" {
// CUDA version of the scenario_step() in
// "ai_economist.foundation.scenarios.covid19_env.py"

    // CUDA version of the sir_step() in
    // "ai_economist.foundation.scenarios.covid19_env.py"
    __device__ void cuda_sir_step(
        float* susceptible,
        float* infected,
        float* recovered,
        float* vaccinated,
        float* deaths,
        int* num_vaccines_available_t,
        const int* kRealWorldStringencyPolicyHistory,
        const float kStatePopulation,
        const int kNumAgents,
        const int kBetaDelay,
        const float kBetaSlope,
        const float kbetaIntercept,
        int* stringency_level,
        float* beta,
        const float kGamma,
        const float kDeathRate,
        const int kEnvId,
        const int kAgentId,
        int timestep,
        const int kEpisodeLength,
        const int kArrayIdxCurrentTime,
        const int kArrayIdxPrevTime,
        const int kTimeIndependentArrayIdx
    ) {
        float susceptible_fraction_vaccinated = min(
            1.0,
            num_vaccines_available_t[kTimeIndependentArrayIdx] /
                (susceptible[kArrayIdxPrevTime] + kEpsilon));
        float vaccinated_t = min(
            static_cast<float>(num_vaccines_available_t[
                kTimeIndependentArrayIdx]),
            susceptible[kArrayIdxPrevTime]);

        // (S/N) * I in place of (S*I) / N to prevent overflow
        float neighborhood_SI_over_N = susceptible[kArrayIdxPrevTime] /
            kStatePopulation * infected[kArrayIdxPrevTime];
        int stringency_level_tmk;
        if (timestep < kBetaDelay) {
            stringency_level_tmk = kRealWorldStringencyPolicyHistory[
                (timestep - 1) * (kNumAgents - 1) + kAgentId];
        } else {
            stringency_level_tmk = stringency_level[kEnvId * (
                kEpisodeLength + 1) * (kNumAgents - 1) +
                (timestep - kBetaDelay) * (kNumAgents - 1) + k