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

from pexpect.exceptions import TIMEOUT

from devlib.collector import (CollectorBase, CollectorOutput,
                              CollectorOutputEntry)
from devlib.utils.serial_port import get_connection
from typing import TextIO, cast, TYPE_CHECKING, Optional
from pexpect import fdpexpect
from serial import Serial
from io import BufferedWriter
if TYPE_CHECKING:
    from devlib.target import Target


class SerialTraceCollector(CollectorBase):
    """
    A collector that reads serial output and saves it to a file.

    :param target: The devlib Target.
    :param serial_port: The serial port to open.
    :param baudrate: The baud rate (bits per second).
    :param timeout: A timeout for serial reads, in seconds.
    """
    @property
    def collecting(self) -> bool:
        return self._collecting

    def __init__(self, target: 'Target', serial_port: int,
                 baudrate: int, timeout: int = 20):
        super(SerialTraceCollector, self).__init__(target)
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.timeout = timeout
        self.output_path: Optional[str] = None

        self._serial_target: Optional[fdpexpect.fdspawn] = None
        self._conn: Optional[Serial] = None
        self._outfile_fh: Optional[BufferedWriter] = None
        self._collecting: bool = False

    def reset(self) -> None:
        if self._collecting:
            raise RuntimeError("reset was called whilst collecting")

        if self._outfile_fh:
            self._outfile_fh.close()
            self._outfile_fh = None

    def start(self) -> None:
        """
        Open the serial connection and write all data to :attr:`output_path`.

        :raises RuntimeError: If already collecting or :attr:`output_path` is unset.
        """
        if self._collecting:
            raise RuntimeError("start was called whilst collecting")
        if self.output_path is None:
            raise RuntimeError("Output path was not set.")

        self._outfile_fh = open(self.output_path, 'wb')
        start_marker: str = "-------- Starting serial logging --------\n"
        self._outfile_fh.write(start_marker.encode('utf-8'))

        self._serial_target, self._conn = get_connection(port=self.serial_port,
                                                         baudrate=self.baudrate,
                                                         timeout=self.timeout,
                                                         logfile=cast(TextIO, self._outfile_fh),
                                                         init_dtr=False)
        self._collecting = True

    def stop(self) -> None:
        """
        Close the serial connection and finalize the log file.

        :raises RuntimeError: If not currently collecting.
        """
        if not self._collecting:
            raise RuntimeError("stop was called whilst not collecting")

        # We expect the below to fail, but we need to get pexpect to
        # do something so that it interacts with the serial device,
        # and hence updates the logfile.
        try:
            if self._serial_target:
                self._serial_target.expect(".", timeout=1)
        except TIMEOUT:
            pass
        if self._serial_target:
            self._serial_target.close()
        del self._conn

        stop_marker: str = "-------- Stopping serial logging --------\n"
        if self._outfile_fh:
            self._outfile_fh.write(stop_marker.encode('utf-8'))
            self._outfile_fh.flush()
            self._outfile_fh.close()
            self._outfile_fh = None

        self._collecting = False

    def set_output(self, output_path: str) -> None:
        self.output_path = output_path

    def get_data(self) -> CollectorOutput:
        """
        Return a :class:`CollectorOutput` referencing the saved serial log file.

        :raises RuntimeError: If :attr:`output_path` is unset.
        :return: A collector output referencing the serial log file.
        """
        if self._collecting:
            raise RuntimeError("get_data was called whilst collecting")
        if self.output_path is None:
            raise RuntimeError("No data collected.")
        return CollectorOutput([CollectorOutputEntry(self.output_path, 'file')])
