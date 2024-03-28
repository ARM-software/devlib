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
Target runner and related classes are implemented here.
"""

import logging
import os
import time
from platform import machine

from devlib.exception import (TargetStableError, HostError)
from devlib.target import LinuxTarget
from devlib.utils.misc import get_subprocess, which
from devlib.utils.ssh import SshConnection


class TargetRunner:
    """
    A generic class for interacting with targets runners.

    It mainly aims to provide framework support for QEMU like target runners
    (e.g., :class:`QEMUTargetRunner`).

    :param target: Specifies type of target per :class:`Target` based classes.
    :type target: Target

    :param runner_cmd: The command to start runner process (e.g.,
        ``qemu-system-aarch64 -kernel Image -append "console=ttyAMA0" ...``).
        Defaults to None.
    :type runner_cmd: str, optional

    :param connect: Specifies if :class:`TargetRunner` should try to connect
        target after launching it, defaults to True.
    :type connect: bool, optional

    :param boot_timeout: Timeout for target's being ready for SSH access in
        seconds, defaults to 60.
    :type boot_timeout: int, optional

    :raises HostError: if it cannot execute runner command successfully.

    :raises TargetStableError: if Target is inaccessible.
    """

    def __init__(self,
                 target,
                 runner_cmd=None,
                 connect=True,
                 boot_timeout=60):
        self.boot_timeout = boot_timeout
        self.target = target
        self.runner_process = None

        self.logger = logging.getLogger(self.__class__.__name__)

        if runner_cmd is None:
            return

        self.logger.info('runner_cmd: %s', runner_cmd)

        try:
            self.runner_process = get_subprocess(list(runner_cmd.split()))
        except Exception as ex:
            raise HostError(f'Error while running "{runner_cmd}": {ex}') from ex

        if connect:
            self.wait_boot_complete()

    def __enter__(self):
        """
        Complementary method for contextmanager.

        :return: Self object.
        :rtype: TargetRunner
        """

        return self

    def __exit__(self, *_):
        """
        Exit routine for contextmanager.

        Ensure ``TargetRunner.runner_process`` is terminated on exit.
        """

        self.terminate()

    def wait_boot_complete(self):
        """
        Wait for target OS to finish boot up and become accessible over SSH in at most
        ``TargetRunner.boot_timeout`` seconds.

        :raises TargetStableError: In case of timeout.
        """

        start_time = time.time()
        elapsed = 0
        while self.boot_timeout >= elapsed:
            try:
                self.target.connect(timeout=self.boot_timeout - elapsed)
                self.logger.debug('Target is ready.')
                return
            # pylint: disable=broad-except
            except Exception as ex:
                self.logger.info('Cannot connect target: %s', ex)

            time.sleep(1)
            elapsed = time.time() - start_time

        self.terminate()
        raise TargetStableError(f'Target is inaccessible for {self.boot_timeout} seconds!')

    def terminate(self):
        """
        Terminate ``TargetRunner.runner_process``.
        """

        if self.runner_process is None:
            return

        self.logger.debug('Killing target runner...')
        self.runner_process.kill()


class NOPTargetRunner(TargetRunner):
    """
    Class for implementing a target runner which does nothing except providing .target attribute.

    :class:`NOPTargetRunner` is a subclass of :class:`TargetRunner` which is implemented only for
    testing purposes.

    :param target: Specifies type of target per :class:`Target` based classes.
    :type target: Target
    """

    def __init__(self, target):
        super().__init__(target=target)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def wait_boot_complete(self):
        pass

    def terminate(self):
        pass


class QEMUTargetRunner(TargetRunner):
    """
    Class for interacting with QEMU runners.

    :class:`QEMUTargetRunner` is a subclass of :class:`TargetRunner` which performs necessary
    groundwork for launching a guest OS on QEMU.

    :param qemu_settings: A dictionary which has QEMU related parameters. The full list
        of QEMU parameters is below:
        * ``kernel_image``: This is the location of kernel image (e.g., ``Image``) which
            will be used as target's kernel.

        * ``arch``: Architecture type. Defaults to ``aarch64``.

        * ``cpu_types``: List of CPU ids for QEMU. The list only contains ``cortex-a72`` by
            default. This parameter is valid for Arm architectures only.

        * ``initrd_image``: This points to the location of initrd image (e.g.,
            ``rootfs.cpio.xz``) which will be used as target's root filesystem if kernel
            does not include one already.

        * ``mem_size``: Size of guest memory in MiB.

        * ``num_cores``: Number of CPU cores. Guest will have ``2`` cores by default.

        * ``num_threads``: Number of CPU threads. Set to ``2`` by defaults.

        * ``cmdline``: Kernel command line parameter. It only specifies console device in
            default (i.e., ``console=ttyAMA0``) which is valid for Arm architectures.
            May be changed to ``ttyS0`` for x86 platforms.

        * ``enable_kvm``: Specifies if KVM will be used as accelerator in QEMU or not.
            Enabled by default if host architecture matches with target's for improving
            QEMU performance.
    :type qemu_settings: Dict

    :param connection_settings: the dictionary to store connection settings
        of ``Target.connection_settings``, defaults to None.
    :type connection_settings: Dict, optional

    :param make_target: Lambda function for creating :class:`Target` based object.
    :type make_target: func, optional

    :Variable positional arguments: Forwarded to :class:`TargetRunner`.

    :raises FileNotFoundError: if QEMU executable, kernel or initrd image cannot be found.
    """

    def __init__(self,
                 qemu_settings,
                 connection_settings=None,
                 make_target=LinuxTarget,
                 **args):
        self.connection_settings = {
            'host': '127.0.0.1',
            'port': 8022,
            'username': 'root',
            'password': 'root',
            'strict_host_check': False,
        }

        self.connection_settings = {**self.connection_settings, **(connection_settings or {})}

        qemu_args = {
            'arch': 'aarch64',
            'cpu_type': 'cortex-a72',
            'mem_size': 512,
            'num_cores': 2,
            'num_threads': 2,
            'cmdline': 'console=ttyAMA0',
            'enable_kvm': True,
        }

        qemu_args = {**qemu_args, **qemu_settings}

        qemu_executable = f'qemu-system-{qemu_args["arch"]}'
        qemu_path = which(qemu_executable)
        if qemu_path is None:
            raise FileNotFoundError(f'Cannot find {qemu_executable} executable!')

        if qemu_args.get("kernel_image"):
            if not os.path.exists(qemu_args["kernel_image"]):
                raise FileNotFoundError(f'{qemu_args["kernel_image"]} does not exist!')
        else:
            raise KeyError('qemu_settings must have kernel_image!')

        qemu_cmd = f'''{qemu_path} -kernel {qemu_args["kernel_image"]} \
-append "{qemu_args["cmdline"]}" -m {qemu_args["mem_size"]} -smp \
cores={qemu_args["num_cores"]},threads={qemu_args["num_threads"]} \
-netdev user,id=net0,hostfwd=tcp::{self.connection_settings["port"]}-:22 \
-device virtio-net-pci,netdev=net0 --nographic'''

        if qemu_args.get("initrd_image"):
            if not os.path.exists(qemu_args["initrd_image"]):
                raise FileNotFoundError(f'{qemu_args["initrd_image"]} does not exist!')

            qemu_cmd += f' -initrd {qemu_args["initrd_image"]}'

        if qemu_args['arch'].startswith('x86') and machine().startswith('x86'):
            if qemu_args["enable_kvm"]:
                qemu_cmd += ' --enable-kvm'
        else:
            qemu_cmd += f' -machine virt -cpu {qemu_args["cpu_type"]}'

        target = make_target(connect=False,
                             conn_cls=SshConnection,
                             connection_settings=self.connection_settings)

        super().__init__(target=target,
                         runner_cmd=qemu_cmd,
                         **args)
