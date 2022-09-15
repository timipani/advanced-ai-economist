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
                (timestep - kBetaDelay) * (kNumAgents - 1) + kAgentId];
        }
        beta[kTimeIndependentArrayIdx] = stringency_level_tmk *
            kBetaSlope + kbetaIntercept;

        float dS_t = -(neighborhood_SI_over_N * beta[
            kTimeIndependentArrayIdx] *
            (1 - susceptible_fraction_vaccinated) + vaccinated_t);
        float dR_t = kGamma * infected[kArrayIdxPrevTime] + vaccinated_t;
        float dI_t = - dS_t - dR_t;

        susceptible[kArrayIdxCurrentTime] = max(
            0.0,
            susceptible[kArrayIdxPrevTime] + dS_t);
        infected[kArrayIdxCurrentTime] = max(
            0.0,
            infected[kArrayIdxPrevTime] + dI_t);
        recovered[kArrayIdxCurrentTime] = max(
            0.0,
            recovered[kArrayIdxPrevTime] + dR_t);

        vaccinated[kArrayIdxCurrentTime] = vaccinated_t +
            vaccinated[kArrayIdxPrevTime];
        float recovered_but_not_vaccinated = recovered[kArrayIdxCurrentTime] -
            vaccinated[kArrayIdxCurrentTime];
        deaths[kArrayIdxCurrentTime] = recovered_but_not_vaccinated *
            kDeathRate;
    }

    // CUDA version of the softplus() in
    // "ai_economist.foundation.scenarios.covid19_env.py"
    __device__ float softplus(float x) {
        const float kBeta = 1.0;
        const float kThreshold = 20.0;
        if (kBeta * x < kThreshold) {
            return 1.0 / kBeta * log(1.0 + exp(kBeta * x));
        } else {
            return x;
        }
    }

    __device__ float signal2unemployment(
        const int kEnvId,
        const int kAgentId,
        float* signal,
        const float* kUnemploymentConvolutionalFilters,
        const float kUnemploymentBias,
        const int kNumAgents,
        const int kFilterLen,
        const int kNumFilters
    ) {
        float unemployment = 0.0;
        const int kArrayIndexOffset = kEnvId * (kNumAgents - 1) * kNumFilters *
            kFilterLen + kAgentId * kNumFilters * kFilterLen;
        for (int index = 0; index < (kFilterLen * kNumFilters); index ++) {
            unemployment += signal[kArrayIndexOffset + index] *
            kUnemploymentConvolutionalFilters[index];
        }
        return softplus(unemployment) + kUnemploymentBias;
    }

    // CUDA version of the unemployment_step() in
  