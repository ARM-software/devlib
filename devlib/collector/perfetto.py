#    Copyright 2023-2025 ARM Limited
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

import os
import subprocess
from shlex import quote

from devlib.host import PACKAGE_BIN_DIRECTORY
from devlib.collector import (CollectorBase, CollectorOutput,
                              CollectorOutputEntry)
from devlib.exception import TargetStableError, HostError
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from devlib.target import Target, BackgroundCommand

OUTPUT_PERFETTO_TRACE = 'devlib-trace.perfetto-trace'


class PerfettoCollector(CollectorBase):
    """
    Perfetto is a production-grade open-source stack for performance instrumentation
    and trace analysis developed by Google. It offers services and libraries for
    recording system-level and app-level traces, native + java heap profiling,
    a library for analyzing traces using SQL and a web-based UI to visualize and
    explore multi-GB traces.

    This collector takes a path to a perfetto config file saved on disk and passes
    it directly to the tool.

    On Android platfroms Perfetto is included in the framework starting with Android 9.
    On Android 8 and below, follow the Linux instructions below to build and include
    the standalone tracebox binary.

    On Linux platforms, either traced (Perfetto tracing daemon) needs to be running
    in the background or the tracebox binary needs to be built from source and placed
    in the Package Bin directory. The build instructions can be found here:

    It is also possible to force using the prebuilt tracebox binary on platforms which
    already have traced running using the force_tracebox collector parameter.

    https://perfetto.dev/docs/contributing/build-instructions

    After building the 'tracebox' binary should be copied to devlib/bin/<arch>/.

    For more information consult the official documentation:
    https://perfetto.dev/docs/

    :param target: The devlib Target.
    :param config: Path to a Perfetto text config (proto) if any.
    :param force_tracebox: If True, force usage of tracebox instead of native Perfetto.
    """

    def __init__(self, target: 'Target', config: Optional[str] = None, force_tracebox: bool = False):
        super().__init__(target)
        self.bg_cmd: Optional['BackgroundCommand'] = None
        self.config = config
        self.target_binary: str = 'perfetto'
        target_output_path: Optional[str] = self.target.working_directory

        install_tracebox: bool = force_tracebox or (target.os in ['linux', 'android'] and not target.is_running('traced'))

        # Install Perfetto through tracebox
        if install_tracebox:
            self.target_binary = 'tracebox'
            if not self.target.get_installed(self.target_binary):
                host_executable: str = os.path.join(PACKAGE_BIN_DIRECTORY,
                                                    self.target.abi, self.target_binary)
                if not os.path.exists(host_executable):
                    raise HostError("{} not found on the host".format(self.target_binary))
                self.target.install(host_executable)
        # Use Android's built-in Perfetto
        elif target.os == 'android':
            os_version: str = target.os_version['release']
            if int(os_version) >= 9:
                # Android requires built-in Perfetto to write to this directory
                target_output_path = '/data/misc/perfetto-traces'
                # Android 9 and 10 require traced to be enabled manually
                if int(os_version) <= 10:
                    target.execute('setprop persist.traced.enable 1')

        self.target_output_file = target.path.join(target_output_path, OUTPUT_PERFETTO_TRACE)

    def start(self) -> None:
        """
        Start Perfetto tracing by feeding the config to the perfetto (or tracebox) binary.

        :raises TargetStableError: If perfetto/tracebox cannot be started on the target.
        """
        cmd: str = "{} cat {} | {} --txt -c - -o {}".format(
            quote(self.target.busybox or ''), quote(self.config or ''), quote(self.target_binary), quote(self.target_output_file)
        )
        # start tracing
        if self.bg_cmd is None:
            self.bg_cmd = self.target.background(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            raise TargetStableError('Perfetto collector is not re-entrant')

    def stop(self) -> None:
        """
        Stop Perfetto tracing and finalize the trace file.
        """
        if self.bg_cmd:
            # stop tracing
            self.bg_cmd.cancel()
            self.bg_cmd = None

    def set_output(self, output_path: str) -> None:
        """
        Specify where the trace file will be pulled on the host.

        :param output_path: The file path or directory on the host.
        """
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, os.path.basename(self.target_output_file))
        self.output_path = output_path

    def get_data(self) -> CollectorOutput:
        """
        Pull the trace file from the target and return a :class:`CollectorOutput`.

        :raises RuntimeError: If :attr:`output_path` is unset or if no trace file exists.
        :return: A collector output referencing the Perfetto trace file.
        """
        if self.output_path is None:
            raise RuntimeError("Output path was not set.")
        if not self.target.file_exists(self.target_output_file):
            raise RuntimeError("Output file not found on the device")
        self.target.pull(self.target_output_file, self.output_path)
        output = CollectorOutput()
        if not os.path.isfile(self.output_path):
            self.logger.warning('Perfetto trace not pulled from device.')
        else:
            output.append(CollectorOutputEntry(self.output_path, 'file'))
        return output
