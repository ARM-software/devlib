#    Copyright 2018-2019 ARM Limited
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

# pylint: disable=missing-docstring

import collections
import os
import sys

from devlib.utils.cli import Command
from devlib.host import PACKAGE_BIN_DIRECTORY
from devlib.trace import TraceCollector

if sys.version_info >= (3, 0):
    from shlex import quote
else:
    from pipes import quote


class PerfCommandDict(collections.OrderedDict):

    def __init__(self, yaml_dict):
        super().__init__()
        self._stat_command_labels = set()
        if isinstance(yaml_dict, self.__class__):
            for key, val in yaml_dict.items():
                self[key] = val
            return
        yaml_dict_copy = yaml_dict.copy()
        for label, parameters in yaml_dict_copy.items():
            self[label] = Command(kwflags_join=',',
                                  kwflags_sep='=',
                                  end_of_options='--',
                                  **parameters)
            if 'stat'in parameters['command']:
                self._stat_command_labels.add(label)

    def stat_commands(self):
        return {label: self[label] for label in self._stat_command_labels}

    def as_strings(self):
        return {label: str(cmd) for label, cmd in self.items()}


class PerfCollector(TraceCollector):
    """Perf is a Linux profiling tool based on performance counters.

    Performance counters are typically CPU hardware registers (found in the
    Performance Monitoring Unit) that count hardware events such as
    instructions executed, cache-misses suffered, or branches mispredicted.
    Because each ``event`` corresponds to a hardware counter, the maximum
    number of events that can be tracked is imposed by the available hardware.

    By extension, performance counters, in the context of ``perf``, also refer
    to so-called "software counters" representing events that can be tracked by
    the OS kernel (e.g. context switches). As these are software events, the
    counters are kept in RAM and the hardware virtually imposes no limit on the
    number that can be used.

    This collector calls ``perf`` ``commands`` to capture a run of a workload.
    The ``pre_commands`` and ``post_commands`` are provided to suit those
    ``perf`` commands that don't actually capture data (``list``, ``config``,
    ``report``, ...).

    ``pre_commands``, ``commands`` and ``post_commands`` are instances of
    :class:`PerfCommandDict`.
    """
    def __init__(self, target, force_install=False, pre_commands=None,
                 commands=None, post_commands=None):
        # pylint: disable=too-many-arguments
        super(PerfCollector, self).__init__(target)
        self.pre_commands = pre_commands or PerfCommandDict({})
        self.commands = commands or PerfCommandDict({})
        self.post_commands = post_commands or PerfCommandDict({})

        self.binary = self.target.get_installed('perf')
        if force_install or not self.binary:
            host_binary = os.path.join(PACKAGE_BIN_DIRECTORY,
                                       self.target.abi, 'perf')
            self.binary = self.target.install(host_binary)

        self.kill_sleep = False

    def reset(self):
        super(PerfCollector, self).reset()
        self.target.remove(self.working_directory())
        self.target.killall('perf', as_root=self.target.is_rooted)

    def start(self):
        super(PerfCollector, self).start()
        for label, command in self.pre_commands.items():
            self.execute(str(command), label)
        for label, command in self.commands.items():
            self.kick_off(str(command), label)
            if 'sleep' in str(command):
                self.kill_sleep = True

    def stop(self):
        super(PerfCollector, self).stop()
        self.target.killall('perf', signal='SIGINT',
                            as_root=self.target.is_rooted)
        if self.kill_sleep:
            self.target.killall('sleep', as_root=self.target.is_rooted)
        for label, command in self.post_commands.items():
            self.execute(str(command), label)

    def kick_off(self, command, label=None):
        directory = quote(self.working_directory(label or 'default'))
        return self.target.kick_off('mkdir -p {0} && cd {0} && {1} {2}'
                                    .format(directory, self.binary, command),
                                    as_root=self.target.is_rooted)

    def execute(self, command, label=None):
        directory = quote(self.working_directory(label or 'default'))
        return self.target.execute('mkdir -p {0} && cd {0} && {1} {2}'
                                   .format(directory, self.binary, command),
                                   as_root=self.target.is_rooted)

    def working_directory(self, label=None):
        wdir = self.target.path.join(self.target.working_directory,
                                     'instrument', 'perf')
        return wdir if label is None else self.target.path.join(wdir, label)

    def get_traces(self, host_outdir):
        self.target.pull(self.working_directory(), host_outdir,
                         as_root=self.target.is_rooted)

    def get_trace(self, outfile):
        raise NotImplementedError
