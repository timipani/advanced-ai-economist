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
        "'pip i