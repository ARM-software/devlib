#
#    Copyright 2024 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Module for testing targets."""

import os
import shutil
import tempfile
from pprint import pp
import pytest

from devlib import LocalLinuxTarget
from devlib.utils.misc import load_struct_from_yaml


def build_targets():
    """Read targets from a YAML formatted config file"""

    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'target_configs.yaml')

    target_configs = load_struct_from_yaml(config_file)
    if target_configs is None:
        raise ValueError(f'{config_file} looks empty!')

    targets = []

    if target_configs.get('LocalLinuxTarget') is not None:
        print('> LocalLinux targets:')
        for entry in target_configs['LocalLinuxTarget'].values():
            pp(entry)
            ll_target = LocalLinuxTarget(connection_settings=entry['connection_settings'])
            targets.append(ll_target)

    return targets


@pytest.mark.parametrize("target", build_targets())
def test_read_multiline_values(target):
    """
    Test Target.read_tree_values_flat()

    :param target: Type of target per :class:`Target` based classes.
    :type target: Target
    """

    data = {
        'test1': '1',
        'test2': '2\n\n',
        'test3': '3\n\n4\n\n',
    }

    tempdir = tempfile.mkdtemp(prefix='devlib-test-')
    for key, value in data.items():
        path = os.path.join(tempdir, key)
        with open(path, 'w', encoding='utf-8') as wfh:
            wfh.write(value)

    raw_result = target.read_tree_values_flat(tempdir)
    result = {os.path.basename(k): v for k, v in raw_result.items()}

    shutil.rmtree(tempdir)

    assert {k: v.strip() for k, v in data.items()} == result
