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

import re
from itertools import takewhile
from datetime import timedelta
import logging

from devlib.collector import (CollectorBase, CollectorOutput,
                              CollectorOutputEntry)
from devlib.exception import TargetStableError
from devlib.utils.misc import memoized, get_logger

from typing import (Optional, Tuple, List,
                    Union, Any, TYPE_CHECKING)
from collections.abc import Generator
if TYPE_CHECKING:
    from devlib.target import Target

_LOGGER: logging.Logger = get_logger('dmesg')


class KernelLogEntry(object):
    """
    Entry of the kernel ring buffer.

    :param facility: facility the entry comes from

    :param level: log level

    :param timestamp: Timestamp of the entry

    :param msg: Content of the entry

    :param line_nr: Line number at which this entry appeared in the ``dmesg``
        output. Note that this is not guaranteed to be unique across collectors, as
        the buffer can be cleared. The timestamp is the only reliable index.
    """

    _TIMESTAMP_MSG_REGEX = re.compile(r'\[(.*?)\] (.*)')
    _RAW_LEVEL_REGEX = re.compile(r'<([0-9]+)>(.*)')
    _PRETTY_LEVEL_REGEX = re.compile(r'\s*([a-z]+)\s*:([a-z]+)\s*:\s*(.*)')

    def __init__(self, facility: Optional[str], level: str,
                 timestamp: timedelta, msg: str, line_nr: int = 0):
        self.facility = facility
        self.level = level
        self.timestamp = timestamp
        self.msg = msg
        self.line_nr = line_nr

    @classmethod
    def from_str(cls, line: str, line_nr: int = 0) -> 'KernelLogEntry':
        """
        Parses a "dmesg --decode" output line, formatted as following:
        kern  :err   : [3618282.310743] nouveau 0000:01:00.0: systemd-logind[988]: nv50cal_space: -16

        Or the more basic output given by "dmesg -r":
        <3>[3618282.310743] nouveau 0000:01:00.0: systemd-logind[988]: nv50cal_space: -16

        :param line: A string from dmesg.
        :param line_nr: The line number in the overall log.
        :raises ValueError: If the line format is invalid.
        :return: A constructed :class:`KernelLogEntry`.

        """

        def parse_raw_level(line: str) -> Tuple[str, Union[str, Any]]:
            match = cls._RAW_LEVEL_REGEX.match(line)
            if not match:
                raise ValueError(f'dmesg entry format not recognized: {line}')
            level, remainder = match.groups()
            levels = DmesgCollector.LOG_LEVELS
            # BusyBox dmesg can output numbers that need to wrap around
            level = levels[int(level) % len(levels)]
            return level, remainder

        def parse_pretty_level(line: str) -> Tuple[str, str, str]:
            match = cls._PRETTY_LEVEL_REGEX.match(line)
            if not match:
                raise ValueError(f'dmesg entry pretty format not recognized: {line}')
            facility, level, remainder = match.groups()
            return facility, level, remainder

        def parse_timestamp_msg(line: str) -> Tuple[timedelta, str]:
            match = cls._TIMESTAMP_MSG_REGEX.match(line)
            if not match:
                raise ValueError(f'dmesg entry timestamp format not recognized: {line}')
            timestamp, msg = match.groups()
            timestamp = timedelta(seconds=float(timestamp.strip()))
            return timestamp, msg

        line = line.strip()

        # If we can parse the raw prio directly, that is a basic line
        try:
            level, remainder = parse_raw_level(line)
            facility: Optional[str] = None
        except ValueError:
            facility, level, remainder = parse_pretty_level(line)

        timestamp, msg = parse_timestamp_msg(remainder)

        return cls(
            facility=facility,
            level=level,
            timestamp=timestamp,
            msg=msg.strip(),
            line_nr=line_nr,
        )

    @classmethod
    def from_dmesg_output(cls, dmesg_out: str, error: Optional[str] = None) -> Generator['KernelLogEntry', None, None]:
        """
        Return a generator of :class:`KernelLogEntry` for each line of the
        output of dmesg command.

        :param dmesg_out: The dmesg output to parse.

        :param error: If ``"raise"`` or ``None``, an exception will be raised
            if a parsing error occurs. If ``"warn"``, it will be logged at
            WARNING level. If ``"ignore"``, it will be ignored. If a callable
            is passed, the exception will be passed to it.
        :return: A generator of parsed :class:`KernelLogEntry` objects.

        .. note:: The same restrictions on the dmesg output format as for
            :meth:`from_str` apply.
        """
        for i, line in enumerate(dmesg_out.splitlines()):
            if line.strip():
                try:
                    yield cls.from_str(line, line_nr=i)
                except Exception as e:
                    if error in (None, 'raise'):
                        raise e
                    elif error == 'warn':
                        _LOGGER.warn(f'error while parsing line "{line!r}": {e}')
                    elif error == 'ignore':
                        pass
                    elif callable(error):
                        error(e)
                    else:
                        raise ValueError(f'Unknown error handling strategy: {error}')

    def __str__(self):
        facility = self.facility + ': ' if self.facility else ''
        return '{facility}{level}: [{timestamp}] {msg}'.format(
            facility=facility,
            level=self.level,
            timestamp=self.timestamp.total_seconds(),
            msg=self.msg,
        )


class DmesgCollector(CollectorBase):
    """
    Dmesg output collector.

    :param target: The devlib Target (must be rooted).

    :param level: Minimum log level to enable. All levels that are more
        critical will be collected as well.

    :param facility: Facility to record, see dmesg --help for the list.

    :param empty_buffer: If ``True``, the kernel dmesg ring buffer will be
        emptied before starting. Note that this will break nesting of collectors,
        so it's not recommended unless it's really necessary.

    :param parse_error: A string to be appended to error lines if parse fails.
    .. warning:: If BusyBox dmesg is used, facility and level will be ignored,
        and the parsed entries will also lack that information.
    """

    # taken from "dmesg --help"
    # This list needs to be ordered by priority
    LOG_LEVELS: List[str] = [
        "emerg",        # system is unusable
        "alert",        # action must be taken immediately
        "crit",         # critical conditions
        "err",          # error conditions
        "warn",         # warning conditions
        "notice",       # normal but significant condition
        "info",         # informational
        "debug",        # debug-level messages
    ]

    def __init__(self, target: 'Target', level: str = LOG_LEVELS[-1],
                 facility: str = 'kern', empty_buffer: bool = False,
                 parse_error: Optional[str] = None):
        super(DmesgCollector, self).__init__(target)

        if not target.is_rooted:
            raise TargetStableError('Cannot collect dmesg on non-rooted target')

        self.output_path: Optional[str] = None

        if level not in self.LOG_LEVELS:
            raise ValueError('level needs to be one of: {}'.format(
                ', '.join(self.LOG_LEVELS)
            ))
        self.level = level

        # Check if we have a dmesg from a recent util-linux build, rather than
        # e.g. busybox's dmesg or the one shipped on some Android versions
        # (toybox).  Note: BusyBox dmesg does not support -h, but will still
        # print the help with an exit code of 1
        help_: str = self.target.execute('dmesg -h', check_exit_code=False)
        self.basic_dmesg: bool = not all(
            opt in help_
            for opt in ('--facility', '--force-prefix', '--decode', '--level')
        )

        self.facility = facility
        try:
            needs_root: bool = target.read_sysctl('kernel.dmesg_restrict')
        except ValueError:
            needs_root = True
        else:
            needs_root = bool(int(needs_root))
        self.needs_root = needs_root

        self._begin_timestamp: Optional[timedelta] = None
        self.empty_buffer: bool = empty_buffer
        self._dmesg_out: Optional[str] = None
        self._parse_error: Optional[str] = parse_error

    @property
    def dmesg_out(self) -> Optional[str]:
        """
        Get the dmesg output
        """
        out: Optional[str] = self._dmesg_out
        if out is None:
            return None
        else:
            try:
                entry: KernelLogEntry = self.entries[0]
            except IndexError:
                return ''
            else:
                i: int = entry.line_nr
                return '\n'.join(out.splitlines()[i:])

    @property
    def entries(self) -> List[KernelLogEntry]:
        """
        Get the entries as a list of class:KernelLogEntry
        """
        if self._dmesg_out is None:
            raise ValueError('dmesg is None')

        return self._get_entries(
            self._dmesg_out,
            self._begin_timestamp,
            error=self._parse_error,
        )

    @memoized
    def _get_entries(self, dmesg_out: str, timestamp: Optional[timedelta],
                     error: Optional[str]) -> List[KernelLogEntry]:
        entry_ = KernelLogEntry.from_dmesg_output(dmesg_out, error=error)
        entries = list(entry_)
        if timestamp is None:
            return entries
        else:
            try:
                first: KernelLogEntry = entries[0]
            except IndexError:
                pass
            else:
                if first.timestamp > timestamp:
                    msg = 'The dmesg ring buffer has ran out of memory or has been cleared and some entries have been lost'
                    raise ValueError(msg)

            return [
                entry
                for entry in entries
                # Only select entries that are more recent than the one at last
                # reset()
                if entry.timestamp > timestamp
            ]

    def _get_output(self) -> None:
        """
        Get the dmesg collector output into _dmesg_out attribute
        """
        levels_list: List[str] = list(takewhile(
            lambda level: level != self.level,
            self.LOG_LEVELS
        ))
        levels_list.append(self.level)
        if self.basic_dmesg:
            cmd = 'dmesg -r'
        else:
            cmd = 'dmesg --facility={facility} --force-prefix --decode --level={levels}'.format(
                levels=','.join(levels_list),
                facility=self.facility,
            )

        self._dmesg_out = self.target.execute(cmd, as_root=self.needs_root)

    def reset(self) -> None:
        self._dmesg_out = None

    def start(self) -> None:
        """
        Start collecting dmesg logs.
        :raises TargetStableError: If the target is not rooted.
        """
        # If the buffer is emptied on start(), it does not matter as we will
        # not end up with entries dating from before start()
        if self.empty_buffer:
            # Empty the dmesg ring buffer. This requires root in all cases
            self.target.execute('dmesg -c', as_root=True)
        else:
            self._get_output()
            try:
                entry = self.entries[-1]
            except IndexError:
                pass
            else:
                self._begin_timestamp = entry.timestamp

    def stop(self) -> None:
        self._get_output()

    def set_output(self, output_path: str) -> None:
        self.output_path = output_path

    def get_data(self) -> CollectorOutput:
        if self.output_path is None:
            raise RuntimeError("Output path was not set.")
        with open(self.output_path, 'wt') as f:
            f.write((self.dmesg_out or '') + '\n')
        return CollectorOutput([CollectorOutputEntry(self.output_path, 'file')])
