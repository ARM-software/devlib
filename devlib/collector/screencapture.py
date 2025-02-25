#    Copyright 2018-2025ARM Limited
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

import logging
import os
import sys
import threading
import time

from devlib.collector import (CollectorBase, CollectorOutput,
                              CollectorOutputEntry)
from devlib.exception import WorkerThreadError
from devlib.utils.misc import get_logger
from typing import TYPE_CHECKING, Optional, cast
if TYPE_CHECKING:
    from devlib.target import Target


class ScreenCapturePoller(threading.Thread):
    """
    A background thread that periodically captures screenshots from the target.

    :param target: The devlib Target.
    :param period: Interval in seconds between captures. If None, the logic may differ.
    :param timeout: Maximum time to wait for the poller thread to stop.
    """
    def __init__(self, target: 'Target', period: Optional[float], timeout: int = 30):
        super(ScreenCapturePoller, self).__init__()
        self.target = target
        self.logger: logging.Logger = get_logger('screencapture')
        self.period = period
        self.timeout = timeout
        self.stop_signal = threading.Event()
        self.lock = threading.Lock()
        self.last_poll: float = 0
        self.daemon: bool = True
        self.exc: Optional[Exception] = None
        self.output_path: Optional[str] = None

    def set_output(self, output_path: str) -> None:
        self.output_path = output_path

    def run(self) -> None:
        """
        Continuously capture screenshots at the specified interval until stopped.
        """
        self.logger.debug('Starting screen capture polling')
        try:
            if self.output_path is None:
                raise RuntimeError("Output path was not set.")
            while True:
                if self.stop_signal.is_set():
                    break
                with self.lock:
                    current_time = time.time()
                    if (current_time - self.last_poll) >= cast(float, self.period):
                        self.poll()
                time.sleep(0.5)
        except Exception:  # pylint: disable=W0703
            self.exc = WorkerThreadError(self.name, sys.exc_info())

    def stop(self) -> None:
        """
        Signal the thread to stop and wait for it to exit, up to :attr:`timeout`.
        """
        self.logger.debug('Stopping screen capture polling')
        self.stop_signal.set()
        self.join(self.timeout)
        if self.is_alive():
            self.logger.error('Could not join screen capture poller thread.')
        if self.exc:
            raise self.exc  # pylint: disable=E0702

    def poll(self) -> None:
        self.last_poll = time.time()
        self.target.capture_screen(os.path.join(self.output_path or '', "screencap_{ts}.png"))


class ScreenCaptureCollector(CollectorBase):
    """
    A collector that periodically captures screenshots from a target device.

    :param target: The devlib Target.
    :param period: Interval in seconds between captures.
    """
    def __init__(self, target: 'Target', period: Optional[float] = None):
        super(ScreenCaptureCollector, self).__init__(target)
        self._collecting: bool = False
        self.output_path: Optional[str] = None
        self.period = period
        self.target = target

    def set_output(self, output_path: str) -> None:
        self.output_path = output_path

    def reset(self) -> None:
        self._poller = ScreenCapturePoller(self.target, self.period)

    def get_data(self) -> CollectorOutput:
        """
        Return a :class:`CollectorOutput` referencing the directory of captured screenshots.

        :return: A collector output referencing the screenshot directory.
        """
        if self.output_path is None:
            raise RuntimeError("No data collected.")
        return CollectorOutput([CollectorOutputEntry(self.output_path, 'directory')])

    def start(self) -> None:
        """
        Start the screen capture poller thread.

        :raises RuntimeError: If :attr:`output_path` is unset.
        """
        if self.output_path is None:
            raise RuntimeError("Output path was not set.")
        self._poller.set_output(self.output_path)
        self._poller.start()
        self._collecting = True

    def stop(self) -> None:
        """
        Stop the screen capture poller thread.
        """
        if not self._collecting:
            raise RuntimeError('Screen capture collector is not running, nothing to stop')

        self._poller.stop()
        self._collecting = False
