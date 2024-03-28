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

"""
Module for testing targets.

Sample run with log level is set to DEBUG (see
https://docs.pytest.org/en/7.1.x/how-to/logging.html#live-logs for logging details):

$ python -m pytest --log-cli-level DEBUG test_target.py
"""

import logging
import os
import pytest

from devlib import AndroidTarget, ChromeOsTarget, LinuxTarget, LocalLinuxTarget
from devlib._target_runner import NOPTargetRunner, QEMUTargetRunner
from devlib.utils.android import AdbConnection
from devlib.utils.misc import load_struct_from_yaml


logger = logging.getLogger('test_target')


def build_target_runners():
    """Read targets from a YAML formatted config file and create runners for them"""

    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'target_configs.yaml')

    target_configs = load_struct_from_yaml(config_file)
    if target_configs is None:
        raise ValueError(f'{config_file} looks empty!')

    target_runners = []

    if target_configs.get('AndroidTarget') is not None:
        logger.info('> Android targets:')
        for entry in target_configs['AndroidTarget'].values():
            logger.info('%s', repr(entry))
            a_target = AndroidTarget(
                connect=False,
                connection_settings=entry['connection_settings'],
                conn_cls=lambda **kwargs: AdbConnection(adb_as_root=True, **kwargs),
            )
            a_target.connect(timeout=entry.get('timeout', 60))
            target_runners.append(NOPTargetRunner(a_target))

    if target_configs.get('LinuxTarget') is not None:
        logger.info('> Linux targets:')
        for entry in target_configs['LinuxTarget'].values():
            logger.info('%s', repr(entry))
            l_target = LinuxTarget(connection_settings=entry['connection_settings'])
            target_runners.append(NOPTargetRunner(l_target))

    if target_configs.get('ChromeOsTarget') is not None:
        logger.info('> ChromeOS targets:')
        for entry in target_configs['ChromeOsTarget'].values():
            logger.info('%s', repr(entry))
            c_target = ChromeOsTarget(
                connection_settings=entry['connection_settings'],
                working_directory='/tmp/devlib-target',
            )
            target_runners.append(NOPTargetRunner(c_target))

    if target_configs.get('LocalLinuxTarget') is not None:
        logger.info('> LocalLinux targets:')
        for entry in target_configs['LocalLinuxTarget'].values():
            logger.info('%s', repr(entry))
            ll_target = LocalLinuxTarget(connection_settings=entry['connection_settings'])
            target_runners.append(NOPTargetRunner(ll_target))

    if target_configs.get('QEMUTargetRunner') is not None:
        logger.info('> QEMU target runners:')
        for entry in target_configs['QEMUTargetRunner'].values():
            logger.info('%s', repr(entry))

            qemu_runner = QEMUTargetRunner(
                qemu_settings=entry.get('qemu_settings'),
                connection_settings=entry.get('connection_settings'),
            )

            if entry.get('ChromeOsTarget') is not None:
                # Leave termination of QEMU runner to ChromeOS target.
                target_runners.append(NOPTargetRunner(qemu_runner.target))

                logger.info('>> ChromeOS target: %s', repr(entry["ChromeOsTarget"]))
                qemu_runner.target = ChromeOsTarget(
                    connection_settings={
                        **entry['ChromeOsTarget']['connection_settings'],
                        **qemu_runner.target.connection_settings,
                    },
                    working_directory='/tmp/devlib-target',
                )

            target_runners.append(qemu_runner)

    return target_runners


@pytest.mark.parametrize("target_runner", build_target_runners())
def test_read_multiline_values(target_runner):
    """
    Test Target.read_tree_values_flat()

    :param target_runner: TargetRunner object to test.
    :type target_runner: TargetRunner
    """

    data = {
        'test1': '1',
        'test2': '2\n\n',
        'test3': '3\n\n4\n\n',
    }

    target = target_runner.target

    logger.info('target=%s os=%s hostname=%s',
                target.__class__.__name__, target.os, target.hostname)

    with target.make_temp() as tempdir:
        logger.debug('Created %s.', tempdir)

        for key, value in data.items():
            path = os.path.join(tempdir, key)
            logger.debug('Writing %s to %s...', repr(value), path)
            target.write_value(path, value, verify=False,
                               as_root=target.conn.connected_as_root)

        logger.debug('Reading values from target...')
        raw_result = target.read_tree_values_flat(tempdir)
        result = {os.path.basename(k): v for k, v in raw_result.items()}

    logger.debug('Removing %s...', target.working_directory)
    target.remove(target.working_directory)

    target_runner.terminate()

    assert {k: v.strip() for k, v in data.items()} == result
