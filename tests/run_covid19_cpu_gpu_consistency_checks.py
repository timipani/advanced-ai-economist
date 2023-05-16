# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
Consistency tests for comparing the cuda (gpu) / no cuda (cpu) version
"""

import GPUtil

try:
    num_gpus_available = len(GPUtil.getAvailable())
    assert num_gpus_available > 0, "The env consistency checker needs a GPU to run!"
    print(
        f"Inside env_cpu_gpu_consistency_checker.py: "
        f"{num_gpus_available} GPUs are available."
    )
    from warp_drive.utils.env_registrar import EnvironmentRegistrar
    from warp_drive.env_cpu_gpu_consistency_checker import EnvironmentCPUvsGPU
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "The env consistency checker requires the 'WarpDrive' package, please run "
        "'pip install rl-warp-drive' first."
    ) from None
except ValueError:
    raise ValueError("The env consistency checker needs a GPU to run!") from None

import os

from ai_economist.foundation.env_wrapper import FoundationEnvWrapper
from ai_economist.foundation.scenarios.covid19.covid19_env import (
    CovidAndEconomyEnvironment,
)

env_registrar = EnvironmentRegistrar()
this_file_dir = os.path.dirname(os.path.abspath(__file__))
env_registrar.add_cuda_env_src_path(
    CovidAndEconomyEnvironment.name,
    os.path.join(this_file_dir, "../ai_economist/foundation/scenarios/covid19/covid19_build.cu")
)
env_configs = {
    "test1": {
        "collate_agent_step_and_reset_data": True,
        "components": [
            {"ControlUSStateOpenCloseStatus": {"action_cooldown_period": 28}},
            {
                "FederalGovernmentSubsidy": {
                    "num_subsidy_levels": 20,
                    "subsidy_interval": 90,
                    "max_annual_subsidy_per_person": 20000,
                }
            },
            {
                "VaccinationCampaign": {
             