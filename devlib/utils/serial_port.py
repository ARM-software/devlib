#    Copyright 2013-2025 ARM Limited
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

import time
from contextlib import contextmanager
from logging import Logger

import serial

# pylint: disable=ungrouped-imports
try:
    from pexpect import fdpexpect
# pexpect < 4.0.0 does not have fdpexpect module
except ImportError:
    import fdpexpect    # type:ignore


# Adding pexpect exceptions into this module's namespace
from pexpect import EOF, TIMEOUT  # NOQA pylint: disable=W0611

from devlib.exception import HostError

from typing import Optional, TextIO, Union, Tuple
from collections.abc import Generator


class SerialLogger(Logger):

    write = Logger.debug

    def flush(self):
        pass


def pulse_dtr(conn: serial.Serial, state: bool = True, duration: float = 0.1) -> None:
    """Set the DTR line of the specified serial connection to the specified state
    for the specified duration (note: the initial state of the line is *not* checked."""
    conn.dtr = state
    time.sleep(duration)
    conn.dtr = not state


# pylint: disable=keyword-arg-before-vararg
def get_connection(timeout: int, init_dtr: Optional[bool] = None,
                   logcls=SerialLogger,
                   logfile: Optional[TextIO] = None, *args, **kwargs) -> Tuple[fdpexpect.fdspawn,
                                                                               serial.Serial]:
    """
    get the serial connection
    """
    if init_dtr is not None:
        kwargs['dsrdtr'] = True
    try:
        conn = serial.Serial(*args, **kwargs)
    except serial.SerialException as e:
        raise HostError(str(e))
    if init_dtr is not None:
        conn.dtr = init_dtr
    conn.nonblocking()
    conn.reset_output_buffer()
    target: fdpexpect.fdspawn = fdpexpect.fdspawn(conn.fileno(), timeout=timeout, logfile=logfile)
    target.logfile_read = logcls('read')
    target.logfile_send = logcls('send')

    # Monkey-patching sendline to introduce a short delay after
    # chacters are sent to the serial. If two sendline s are issued
    # one after another the second one might start putting characters
    # into the serial device before the first one has finished, causing
    # corruption. The delay prevents that.
    tsln = target.sendline

    def sendline(s: Union[str, bytes]) -> int:
        ret: int = tsln(s)
        time.sleep(0.1)
        return ret

    target.sendline = sendline
    return target, conn


def write_characters(conn: fdpexpect.fdspawn, line: str, delay: float = 0.05) -> None:
    """Write a single line out to serial charcter-by-character. This will ensure that nothing will
    be dropped for longer lines."""
    line = line.rstrip('\r\n')
    for c in line:
        conn.send(c)
        time.sleep(delay)
    conn.sendline('')


# pylint: disable=keyword-arg-before-vararg
@contextmanager
def open_serial_connection(timeout: int, get_conn: bool = False,
                           init_dtr: Optional[bool] = None,
                           logcls=SerialLogger, *args, **kwargs) -> Generator[Union[Tuple[fdpexpect.fdspawn, serial.Serial],
                                                                                    fdpexpect.fdspawn], None, None]:
    """
    Opens a serial connection to a device.

    :param timeout: timeout for the fdpexpect spawn object.
    :param conn: ``bool`` that specfies whether the underlying connection
                 object should be yielded as well.
    :param init_dtr: specifies the initial DTR state stat should be set.

    All arguments are passed into the __init__ of serial.Serial. See
    pyserial documentation for details:

        http://pyserial.sourceforge.net/pyserial_api.html#serial.Serial

    :returns: a pexpect spawn object connected to the device.
              See: http://pexpect.sourceforge.net/pexpect.html

    """
    target, conn = get_connection(timeout, init_dtr,
                                  logcls, *args, **kwargs)

    if get_conn:
        target_and_conn: Union[Tuple[fdpexpect.fdspawn, serial.Serial], fdpexpect.fdspawn] = (target, conn)
    else:
        target_and_conn = target

    try:
        yield target_and_conn
    finally:
        target.close()  # Closes the file descriptor used by the conn.
