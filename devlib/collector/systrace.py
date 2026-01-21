#    Copyright 2024-2025 ARM Limited
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

from devlib.collector import (CollectorBase, CollectorOutput,
                              CollectorOutputEntry)
from devlib.exception import TargetStableError, HostError
import devlib.utils.android
from devlib.utils.misc import memoized
from typing import TYPE_CHECKING, List, Optional, Union, TextIO
from subprocess import Popen
if TYPE_CHECKING:
    from devlib.target import AndroidTarget

DEFAULT_CATEGORIES: List[str] = [
    'gfx',
    'view',
    'sched',
    'freq',
    'idle'
]


class SystraceCollector(CollectorBase):
    """
    A trace collector based on Systrace

    For more details, see https://developer.android.com/studio/command-line/systrace

    :param target: Devlib target

    :param outdir: Working directory to use on the host

    :param categories: Systrace categories to trace. See `available_categories`

    :param buffer_size: Buffer size in kb

    :param strict: Raise an exception if any of the requested categories
        are not available
    """

    @property
    @memoized
    def available_categories(self) -> List[str]:
        """
        list of available categories
        """
        lines: List[str] = subprocess.check_output(
            [self.systrace_binary or '', '-l'], universal_newlines=True
        ).splitlines()

        return [line.split()[0] for line in lines if line]

    def __init__(self, target: 'AndroidTarget',
                 categories: Optional[str] = None,
                 buffer_size: Optional[int] = None,
                 strict: bool = False):

        super(SystraceCollector, self).__init__(target)

        self.categories: Union[str, List[str]] = categories or DEFAULT_CATEGORIES
        self.buffer_size = buffer_size
        self.output_path: Optional[str] = None

        self._systrace_process: Optional[Popen] = None
        self._outfile_fh: Optional[TextIO] = None

        # Try to find a systrace binary
        self.systrace_binary: Optional[str] = None

        platform_tools: str = devlib.utils.android.platform_tools  # type: ignore
        systrace_binary_path: str = os.path.join(platform_tools, 'systrace', 'systrace.py')
        if not os.path.isfile(systrace_binary_path):
            raise HostError('Could not find any systrace binary under {}'.format(platform_tools))

        self.systrace_binary = systrace_binary_path

        # Filter the requested categories
        for category in self.categories:
            if category not in self.available_categories:
                message: str = 'Category [{}] not available for tracing'.format(category)
                if strict:
                    raise TargetStableError(message)
                self.logger.warning(message)

        self.categories = list(set(self.categories) & set(self.available_categories))
        if not self.categories:
            raise TargetStableError('None of the requested categories are available')

    def __del__(self):
        self.reset()

    def _build_cmd(self) -> None:
        """
        build command
        """
        self._outfile_fh = open(self.output_path or '', 'w')

        # pylint: disable=attribute-defined-outside-init
        self.systrace_cmd: str = 'python2 -u {} -o {} -e {}'.format(
            self.systrace_binary,
            self._outfile_fh.name,
            self.target.adb_name
        )

        if self.buffer_size:
            self.systrace_cmd += ' -b {}'.format(self.buffer_size)

        self.systrace_cmd += ' {}'.format(' '.join(self.categories))

    def reset(self) -> None:
        if self._systrace_process:
            self.stop()

    def start(self) -> None:
        """
        Start systrace, typically running a systrace command in the background.
        """
        if self._systrace_process:
            raise RuntimeError("Tracing is already underway, call stop() first")
        if self.output_path is None:
            raise RuntimeError("Output path was not set.")

        self.reset()

        self._build_cmd()

        self._systrace_process = subprocess.Popen(
            self.systrace_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            shell=True,
            universal_newlines=True
        )
        if self._systrace_process.stdout:
            self._systrace_process.stdout.read(1)

    def stop(self) -> None:
        """
        Stop systrace and finalize the trace file.
        """
        if not self._systrace_process:
            raise RuntimeError("No tracing to stop, call start() first")

        # Systrace expects <enter> to stop
        self._systrace_process.communicate('\n')
        self._systrace_process = None

        if self._outfile_fh:
            self._outfile_fh.close()
            self._outfile_fh = None

    def set_output(self, output_path: str) -> None:
        self.output_path = output_path

    def get_data(self) -> CollectorOutput:
        """
        Pull the trace HTML (or raw data) from the target and return a
        :class:`CollectorOutput`.

        :return: A collector output referencing the systrace file.
        """
        if self._systrace_process:
            raise RuntimeError("Tracing is underway, call stop() first")
        if self.output_path is None:
            raise RuntimeError("No data collected.")
        return CollectorOutput([CollectorOutputEntry(self.output_path, 'file')])
