#    Copyright 2018-2025 ARM Limited
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
import re
import time
from past.builtins import zip

from devlib.host import PACKAGE_BIN_DIRECTORY
from devlib.collector import (CollectorBase, CollectorOutput,
                              CollectorOutputEntry)
from devlib.utils.misc import ensure_file_directory_exists as _f
from typing import (cast, List, Dict, TYPE_CHECKING, Optional,
                    Union)
from signal import Signals
if TYPE_CHECKING:
    from devlib.target import Target


PERF_STAT_COMMAND_TEMPLATE: str = '{binary} {command} {options} {events} {sleep_cmd} > {outfile} 2>&1 '
PERF_REPORT_COMMAND_TEMPLATE: str = '{binary} report {options} -i {datafile} > {outfile} 2>&1 '
PERF_REPORT_SAMPLE_COMMAND_TEMPLATE: str = '{binary} report-sample {options} -i {datafile} > {outfile} '
PERF_RECORD_COMMAND_TEMPLATE: str = '{binary} record {options} {events} -o {outfile}'

PERF_DEFAULT_EVENTS: List[str] = [
    'cpu-migrations',
    'context-switches',
]

SIMPLEPERF_DEFAULT_EVENTS: List[str] = [
    'raw-cpu-cycles',
    'raw-l1-dcache',
    'raw-l1-dcache-refill',
    'raw-br-mis-pred',
    'raw-instruction-retired',
]

DEFAULT_EVENTS: Dict[str, List[str]] = {'perf': PERF_DEFAULT_EVENTS, 'simpleperf': SIMPLEPERF_DEFAULT_EVENTS}


class PerfCollector(CollectorBase):
    """
    Perf is a Linux profiling with performance counters.
    Simpleperf is an Android profiling tool with performance counters.

    It is highly recomended to use perf_type = simpleperf when using this instrument
    on android devices, since it recognises android symbols in record mode and is much more stable
    when reporting record .data files. For more information see simpleperf documentation at:
    https://android.googlesource.com/platform/system/extras/+/master/simpleperf/doc/README.md

    Performance counters are CPU hardware registers that count hardware events
    such as instructions executed, cache-misses suffered, or branches
    mispredicted. They form a basis for profiling applications to trace dynamic
    control flow and identify hotspots.

    Perf accepts options and events. If no option is given the default '-a' is
    used. For events, the default events are migrations and cs for perf and raw-cpu-cycles,
    raw-l1-dcache, raw-l1-dcache-refill, raw-instructions-retired. They both can
    be specified in the config file.

    Events must be provided as a list that contains them and they will look like
    this ::

        perf_events = ['migrations', 'cs']

    Events can be obtained by typing the following in the command line on the
    device ::

        perf list
        simpleperf list

    Whereas options, they can be provided as a single string as following ::

        perf_options = '-a -i'

    Options can be obtained by running the following in the command line ::

        man perf-stat

    :param target: The devlib Target (rooted if on Android).
    :param perf_type: Either 'perf' or 'simpleperf'.
    :param command: The perf command to run (e.g. 'stat' or 'record').
    :param events: A list of events to collect. Defaults to built-in sets.
    :param optionstring: Extra CLI options (a string or list of strings).
    :param report_options: Additional options for ``perf report``.
    :param run_report_sample: If True, run the ``report-sample`` subcommand.
    :param report_sample_options: Additional options for ``report-sample``.
    :param labels: Unique labels for each command or option set.
    :param force_install: If True, reinstall perf even if it's already on the target.
    :param validate_events: If True, verify that requested events are available.
    """

    def __init__(self,
                 target: 'Target',
                 perf_type: str = 'perf',
                 command: str = 'stat',
                 events: Optional[List[str]] = None,
                 optionstring: Optional[Union[str, List[str]]] = None,
                 report_options: Optional[str] = None,
                 run_report_sample: bool = False,
                 report_sample_options: Optional[str] = None,
                 labels: Optional[List[str]] = None,
                 force_install: bool = False,
                 validate_events: bool = True):
        super(PerfCollector, self).__init__(target)
        self.force_install = force_install
        self.labels = labels
        self.report_options = report_options
        self.run_report_sample = run_report_sample
        self.report_sample_options = report_sample_options
        self.output_path: Optional[str] = None
        self.validate_events = validate_events

        # Validate parameters
        if isinstance(optionstring, list):
            self.optionstrings: List[str] = optionstring
        else:
            self.optionstrings = [optionstring] if optionstring else []
        if perf_type in ['perf', 'simpleperf']:
            self.perf_type: str = perf_type
        else:
            raise ValueError('Invalid perf type: {}, must be perf or simpleperf'.format(perf_type))
        if not events:
            self.events: List[str] = DEFAULT_EVENTS[self.perf_type]
        else:
            self.events = events
        if isinstance(self.events, str):
            self.events = [self.events]
        if not self.labels:
            self.labels = ['perf_{}'.format(i) for i in range(len(self.optionstrings))]
        if len(self.labels) != len(self.optionstrings):
            raise ValueError('The number of labels must match the number of optstrings provided for perf.')
        if command in ['stat', 'record']:
            self.command = command
        else:
            raise ValueError('Unsupported perf command, must be stat or record')
        if report_options and (command != 'record'):
            raise ValueError('report_options specified, but command is not record')
        if report_sample_options and (command != 'record'):
            raise ValueError('report_sample_options specified, but command is not record')

        self.binary: str = self.target.get_installed(self.perf_type)
        if self.force_install or not self.binary:
            self.binary = self._deploy_perf()

        if self.validate_events:
            self._validate_events(self.events)

        self.commands: List[str] = self._build_commands()

    def reset(self) -> None:
        self.target.killall(self.perf_type, as_root=self.target.is_rooted)
        self.target.remove(self.target.get_workpath('TemporaryFile*'))
        if self.labels:
            for label in self.labels:
                filepath = self._get_target_file(label, 'data')
                self.target.remove(filepath)
                filepath = self._get_target_file(label, 'rpt')
                self.target.remove(filepath)
                filepath = self._get_target_file(label, 'rptsamples')
                self.target.remove(filepath)

    def start(self) -> None:
        """
        Start the perf command(s) in the background on the target.
        """
        for command in self.commands:
            self.target.background(command, as_root=self.target.is_rooted)

    def stop(self) -> None:
        """
        Send SIGINT to terminate the perf tool, finalizing any data files.
        """
        self.target.killall(self.perf_type, signal=cast(Signals, 'SIGINT'),
                            as_root=self.target.is_rooted)
        if self.perf_type == "perf" and self.command == "stat":
            # perf doesn't transmit the signal to its sleep call so handled here:
            self.target.killall('sleep', as_root=self.target.is_rooted)
            # NB: we hope that no other "important" sleep is on-going

    def set_output(self, output_path: str) -> None:
        """
        Define where perf data or reports will be stored on the host.

        :param output_path: A directory or file path for storing perf results.
        """
        self.output_path = output_path

    def get_data(self) -> CollectorOutput:
        """
        Pull the perf data from the target to the host and optionally generate
        textual reports.

        :raises RuntimeError: If :attr:`output_path` is unset.
        :return: A collector output referencing the saved perf files.
        """
        if self.output_path is None:
            raise RuntimeError("Output path was not set.")

        output = CollectorOutput()
        if self.labels is None:
            raise RuntimeError("labels not set")
        for label in self.labels:
            if self.command == 'record':
                self._wait_for_data_file_write(label, self.output_path)
                path: str = self._pull_target_file_to_host(label, 'rpt', self.output_path)
                output.append(CollectorOutputEntry(path, 'file'))
                if self.run_report_sample:
                    report_samples_path = self._pull_target_file_to_host(label, 'rptsamples', self.output_path)
                    output.append(CollectorOutputEntry(report_samples_path, 'file'))
            else:
                path = self._pull_target_file_to_host(label, 'out', self.output_path)
                output.append(CollectorOutputEntry(path, 'file'))
        return output

    def _deploy_perf(self) -> str:
        """
        install perf on target
        """
        host_executable: str = os.path.join(PACKAGE_BIN_DIRECTORY,
                                            self.target.abi, self.perf_type)
        return self.target.install(host_executable)

    def _get_target_file(self, label: str, extension: str) -> Optional[str]:
        """
        get file path on target
        """
        return self.target.get_workpath('{}.{}'.format(label, extension))

    def _build_commands(self) -> List[str]:
        """
        build perf commands
        """
        commands: List[str] = []
        for opts, label in zip(self.optionstrings, self.labels):
            if self.command == 'stat':
                commands.append(self._build_perf_stat_command(opts, self.events, label))
            else:
                commands.append(self._build_perf_record_command(opts, label))
        return commands

    def _build_perf_stat_command(self, options: str, events: List[str], label) -> str:
        """
        Construct a perf stat command string.

        :param options: Additional perf stat options.
        :param events: The list of events to measure.
        :param label: A label to identify this command/run.
        :return: A command string suitable for running on the target.
        """
        event_string: str = ' '.join(['-e {}'.format(e) for e in events])
        sleep_cmd: str = 'sleep 1000' if self.perf_type == 'perf' else ''
        command: str = PERF_STAT_COMMAND_TEMPLATE.format(binary=self.binary,
                                                         command=self.command,
                                                         options=options or '',
                                                         events=event_string,
                                                         sleep_cmd=sleep_cmd,
                                                         outfile=self._get_target_file(label, 'out'))
        return command

    def _build_perf_report_command(self, report_options: Optional[str], label: str) -> str:
        """
        Construct a perf stat command string.

        :param options: Additional perf stat options.
        :param events: The list of events to measure.
        :param label: A label to identify this command/run.
        :return: A command string suitable for running on the target.
        """
        command = PERF_REPORT_COMMAND_TEMPLATE.format(binary=self.binary,
                                                      options=report_options or '',
                                                      datafile=self._get_target_file(label, 'data'),
                                                      outfile=self._get_target_file(label, 'rpt'))
        return command

    def _build_perf_report_sample_command(self, label: str) -> str:
        """
        build perf report sample command
        """
        command = PERF_REPORT_SAMPLE_COMMAND_TEMPLATE.format(binary=self.binary,
                                                             options=self.report_sample_options or '',
                                                             datafile=self._get_target_file(label, 'data'),
                                                             outfile=self._get_target_file(label, 'rptsamples'))
        return command

    def _build_perf_record_command(self, options: Optional[str], label: str) -> str:
        """
        build perf record command
        """
        event_string: str = ' '.join(['-e {}'.format(e) for e in self.events])
        command: str = PERF_RECORD_COMMAND_TEMPLATE.format(binary=self.binary,
                                                           options=options or '',
                                                           events=event_string,
                                                           outfile=self._get_target_file(label, 'data'))
        return command

    def _pull_target_file_to_host(self, label: str, extension: str, output_path: str) -> str:
        """
        pull a file from target to host
        """
        target_file: Optional[str] = self._get_target_file(label, extension)
        host_relpath: str = os.path.basename(target_file or '')
        host_file: str = _f(os.path.join(output_path, host_relpath))
        self.target.pull(target_file, host_file)
        return host_file

    def _wait_for_data_file_write(self, label: str, output_path: str) -> None:
        """
        wait for file write operation by perf
        """
        data_file_finished_writing: bool = False
        max_tries: int = 80
        current_tries: int = 0
        while not data_file_finished_writing:
            files = self.target.execute('cd {} && ls'.format(self.target.get_workpath('')))
            # Perf stores data in tempory files whilst writing to data output file. Check if they have been removed.
            if 'TemporaryFile' in files and current_tries <= max_tries:
                time.sleep(0.25)
                current_tries += 1
            else:
                if current_tries >= max_tries:
                    self.logger.warning('''writing {}.data file took longer than expected,
                                        file may not have written correctly'''.format(label))
                data_file_finished_writing = True
        report_command: str = self._build_perf_report_command(self.report_options, label)
        self.target.execute(report_command)
        if self.run_report_sample:
            report_sample_command = self._build_perf_report_sample_command(label)
            self.target.execute(report_sample_command)

    def _validate_events(self, events: List[str]) -> None:
        """
        validate events against available perf events on target
        """
        available_events_string: str = self.target.execute('{} list | {} cat'.format(self.perf_type, self.target.busybox))
        available_events: List[str] = available_events_string.splitlines()
        for available_event in available_events:
            if available_event == '':
                continue
            if 'OR' in available_event:
                available_events.append(available_event.split('OR')[1])
            available_events[available_events.index(available_event)] = available_event.split()[0].strip()
        # Raw hex event codes can also be passed in that do not appear on perf/simpleperf list, prefixed with 'r'
        raw_event_code_regex = re.compile(r"^r(0x|0X)?[A-Fa-f0-9]+$")
        for event in events:
            if event in available_events or re.match(raw_event_code_regex, event):
                continue
            else:
                raise ValueError('Event: {} is not in available event list for {}'.format(event, self.perf_type))
