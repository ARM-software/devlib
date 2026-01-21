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

from devlib.collector import (CollectorBase, CollectorOutput,
                              CollectorOutputEntry)
from devlib.utils.android import LogcatMonitor
from typing import (cast, TYPE_CHECKING, List, Optional,
                    Union)
from io import TextIOWrapper
from tempfile import _TemporaryFileWrapper
if TYPE_CHECKING:
    from devlib.target import AndroidTarget, Target


class LogcatCollector(CollectorBase):
    """
    A collector that retrieves logs via `adb logcat` from an Android target.

    :param target: The devlib Target (must be Android).
    :param regexps: A list of regular expressions to filter log lines (optional).
    :param logcat_format: The desired logcat output format (optional).
    """
    def __init__(self, target: 'Target', regexps: Optional[List[str]] = None,
                 logcat_format: Optional[str] = None):
        super(LogcatCollector, self).__init__(target)
        self.regexps = regexps
        self.logcat_format = logcat_format
        self.output_path: Optional[str] = None
        self._collecting: bool = False
        self._prev_log: Optional[Union[TextIOWrapper, _TemporaryFileWrapper[str]]] = None
        self._monitor: Optional[LogcatMonitor] = None

    def reset(self) -> None:
        """
        Clear Collector data but do not interrupt collection
        """
        if not self._monitor:
            return

        if self._collecting:
            self._monitor.clear_log()
        elif self._prev_log:
            os.remove(cast(str, self._prev_log))
            self._prev_log = None

    def start(self) -> None:
        """
        Start capturing logcat output. Raises RuntimeError if no output path is set.
        """
        if self.output_path is None:
            raise RuntimeError("Output path was not set.")
        self._monitor = LogcatMonitor(cast('AndroidTarget', self.target), self.regexps, logcat_format=self.logcat_format)
        if self._prev_log:
            # Append new data collection to previous collection
            self._monitor.start(cast(str, self._prev_log))
        else:
            self._monitor.start(self.output_path)

        self._collecting = True

    def stop(self) -> None:
        """
        Stop collecting logcat lines
        """
        if not self._collecting:
            raise RuntimeError('Logcat monitor not running, nothing to stop')
        if self._monitor:
            self._monitor.stop()
        self._collecting = False
        self._prev_log = self._monitor.logfile if self._monitor else None

    def set_output(self, output_path: str) -> None:
        self.output_path = output_path

    def get_data(self) -> CollectorOutput:
        """
        Return a :class:`CollectorOutput` for the captured logcat data.

        :raises RuntimeError: If :attr:`output_path` is unset.
        :return: A collector output referencing the logcat file.
        """
        if self.output_path is None:
            raise RuntimeError("No data collected.")
        return CollectorOutput([CollectorOutputEntry(self.output_path, 'file')])
