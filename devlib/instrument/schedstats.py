#    Copyright 2017 ARM Limited
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

from collections import OrderedDict
import logging
import re

from devlib.instrument import (Instrument, INSTANTANEOUS,
                               Measurement, MeasurementType)
from devlib.exception import TargetError

# Each entry in schedstats has a space-separated list of fields. DOMAIN_MEASURES
# and CPU_MEASURES are the fields for domain entries and CPU entries
# resepectively.
#
# See kernel/sched/stat.c and Documentation/scheduler/sched-stats.txt
#
# The names used here are based on the identifiers in the scheduler code.

# Some domain fields are repeated for each idle type
DOMAIN_MEASURES = []
for idle_type in ['CPU_IDLE', 'CPU_NOT_IDLE', 'CPU_NEWLY_IDLE']:
    for lb_measure in [
            'lb_count',
            'lb_balanced',
            'lb_failed',
            'lb_imbalance',
            'lb_gained',
            'lb_hot_gained',
            'lb_nobusyq',
            'lb_nobusyg']:
        DOMAIN_MEASURES.append('{}:{}'.format(lb_measure, idle_type))

DOMAIN_MEASURES += [
    'alb_count',
    'alb_failed',
    'alb_pushed',
    'sbe_count',
    'sbe_balanced',
    'sbe_pushed',
    'sbf_count',
    'sbf_balanced',
    'sbf_pushed',
    'ttwu_wake_remote',
    'ttwu_move_affine',
    'ttwu_move_balance'
]

CPU_MEASURES = [
    'yld_count',
    'legacy_always_zero',
    'schedule_count',
    'sched_goidle',
    'ttwu_count',
    'ttwu_local',
    'rq_cpu_time',
    'run_delay',
    'pcount'
]

class SchedstatsInstrument(Instrument):
    """
    An instrument for parsing Linux's schedstats

    Creates a *site* for each CPU and each sched_domain (i.e. for each line of
    /proc/schedstat), and a *channel* for each item in the schedstats file. For
    example a *site* named "cpu0" will be created for the scheduler stats on
    CPU0 and a *site* named "cpu0domain0" will be created for the scheduler
    stats on CPU0's first-level scheduling domain.

    For example:

     - If :method:`reset` is called with ``sites=['cpu0']`` then all
       stats will be collected for CPU0's runqueue, with a channel for each
       statistic.

     - If :method:`reset` is called with ``kinds=['alb_pushed']`` then the count
       of migrations successfully triggered by active_load_balance will be
       collected for each sched domain, with a channel for each domain.

    The measurements are named according to corresponding identifiers in the
    kernel scheduler code. The names for ``sched_domain.lb_*`` stats, which are
    recorded per ``cpu_idle_type`` are suffixed with a ':' followed by the idle
    type, for example ``'lb_balanced:CPU_NEWLY_IDLE'``.

    Only supports schedstats version 15.

    Only supports the CPU and domain data in /proc/schedstat, not the per-task
    data under /proc/<pid>/schedstat.
    """

    mode = INSTANTANEOUS

    sysctl_path = '/proc/sys/kernel/sched_schedstats'
    schedstat_path = '/proc/schedstat'

    def __init__(self, *args, **kwargs):
        super(SchedstatsInstrument, self).__init__(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Check schedstats is present in the kernel and the format version
        # matches what we can parse.
        try:
            lines = self.target.read_value(self.schedstat_path).splitlines()
        except TargetError:
            if not self.target.file_exists(self.schedstat_path):
                raise TargetError('schedstats not supported by target. '
                                  'Ensure CONFIG_SCHEDSTATS is enabled.')
            raise

        match = re.search(r'version ([0-9]+)', lines[0])
        if not match or match.group(1) != '15':
            raise TargetError(
                'Unsupported schedstat version "{}" - only version 15 is supported'
                .format(lines[0]))

        self._ensure_schedstats_enabled()

        # Take a sample of the schedstat file to figure out which channels to
        # create.
        # We'll create a site for each CPU and a site for each sched_domain.
        for site, measures in self._get_sample().iteritems():
            if site.startswith('cpu'):
                measurement_category = 'schedstat_cpu'
            else:
                measurement_category = 'schedstat_domain'

            for measurement_name in measures.keys():
                measurement_type = MeasurementType(
                    measurement_name, '', measurement_category)
                self.add_channel(site=site,
                                 measure=measurement_type)

    def _ensure_schedstats_enabled(self):
        # On 4.6+ kernels, schedstats needs to be enabled via kernel cmdline or
        # sysctl. If not, we'll just get a load of zeroes.
        self.old_sysctl_value = None
        if self.target.kernel_version.parts >= (4, 6):
            if self.target.file_exists(self.sysctl_path):
                self.old_sysctl_value = self.target.read_int(self.sysctl_path)
                self.target.write_value(self.sysctl_path, 1)
            else:
                try:
                    cmdline = self.target.read_value('/proc/cmdline')
                except TargetError:
                    raise TargetError(
                        "Couldn't verify that schedstats is enabled. "
                        "Enabling CONFIG_PROC_SYSCTL will probably help")
                if "schedstats=enable" not in cmdline:
                    raise TargetError(
                        "schedstats is compiled into the kernel but not enabled at runtime. "
                        "Enable CONFIG_PROC_SYSCTL or add schedstats=enable to the cmdline.")
    def teardown(self):
        if self.old_sysctl_value is not None:
            self.target.write_value(self.sysctl_path, self.old_sysctl_value)

    def _get_sample(self):
        lines = self.target.read_value(self.schedstat_path).splitlines()
        ret = OrderedDict()

        # Example /proc/schedstat contents:
        #
        # version 15
        # timestamp <timestamp>
        # cpu0 <cpu fields>
        # domain0 <domain fields>
        # domain1 <domain fields>
        # cpu1 <cpu_fields>
        # domain0 <domain fields>
        # domain1 <domain fields>

        curr_cpu = None
        for line in lines[2:]:
            tokens = line.split()
            if tokens[0].startswith('cpu'):
                curr_cpu = tokens[0]
                site = curr_cpu
                measures = CPU_MEASURES
                tokens = tokens[1:]
            elif tokens[0].startswith('domain'):
                if not curr_cpu:
                    raise TargetError(
                        'Failed to parse schedstats, found domain before CPU')
                # We'll name the site for the domain like "cpu0domain0"
                site = curr_cpu + tokens[0]
                measures = DOMAIN_MEASURES
                tokens = tokens[2:]
            elif tokens[0] == 'eas':
                # This line is added by EAS features. We don't yet parse it as
                # it might not be stable.
                continue
            else:
                self.logger.warning(
                    'Unrecognised schedstats line: "%s', line)
                continue

            values = [int(t) for t in tokens]
            if len(values) != len(measures):
                raise TargetError(
                    'Unexpected length for schedstat line "%s"', line)
            ret[site] = OrderedDict(zip(measures, values))

        return ret

    def take_measurement(self):
        ret = []
        sample = self._get_sample()

        for channel in self.active_channels:
            value = sample[channel.site][channel.kind]
            ret.append(Measurement(value, channel))

        return ret
