# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import json
import os
import sys
from hashlib import sha512

import lz4.frame
from Crypto.PublicKey import RSA

from ai_economist.foundation.base.base_env import BaseEnvironment


def save_episode_log(game_object, filepath, compression_level=16):
    """Save a lz4 compressed version of the dense log stored
    in the provided game object"""
    assert isinstance(game_object, BaseEnvironment)
    compression_level = int(compression_level)
    if compression_level < 0:
        compression_level = 0
    elif compression_level > 16:
        compression_level = 16

    with lz4.frame.open(
        filepath, mode="wb", compression_level=compression_level
    ) as log_file:
        log_bytes = bytes(
            json.dumps(
                game_object.previous_episode_dense_log, ensure_ascii=False
            ).encode("utf-8")
        )
        log_file.write(log_bytes)


def load_episode_log(filepath):
    """Load the dense log saved at provided filepath"""
    with lz4.frame.open(filepath, mode="rb") as log_file:
        log_bytes = log_file.read()
    return json.loads(log_bytes)


def verify_activation_code():
    """
    Validate the user's activation code.
    If the activation code is valid, also save it in a text file for future reference.
    If the activation code is invalid, simply exit the program
    """
    path_to_activation_code_dir = os.path.dirname(os.path.abspath(__file__))

    def validate_activation_code(code, msg=b"covid19 code activation"):
        filepath = os.path.abspath(
            os.path.join(
                path_to_activation_code_dir,
                "scenarios/covid19/key_to_check_activation_code_against",
            )
        )
        with open(filepath, "r") as fp:
            key_pair = RSA.import_key(fp.read())

        hashed_msg = int.from_bytes(sha512(msg).digest(), byteorder="big")
        signature = pow(hashed_msg, key_pair.d, key_pair.n)
        try:
            exp_from_code = int(code, 16)
            hashed_msg_from_signature = pow(signature, exp_from_code, key_pair.n)

            return has