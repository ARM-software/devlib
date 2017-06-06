#    Copyright 2014-2015 ARM Limited
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
from devlib.module import Module
from devlib.exception import TargetError
from devlib.utils.misc import memoized


# a dict of governor name and a list of it tunables that can't be read
WRITE_ONLY_TUNABLES = {
    'interactive': ['boostpulse']
}


class CpufreqModule(Module):

    name = 'cpufreq'

    @staticmethod
    def probe(target):

        # x86 with Intel P-State driver
        if target.abi == 'x86_64':
            path = '/sys/devices/system/cpu/intel_pstate'
            if target.file_exists(path):
                return True

        # Generic CPUFreq support (single policy)
        path = '/sys/devices/system/cpu/cpufreq'
        if target.file_exists(path):
            return True

        # Generic CPUFreq support (per CPU policy)
        path = '/sys/devices/system/cpu/cpu0/cpufreq'
        return target.file_exists(path)

    def __init__(self, target):
        super(CpufreqModule, self).__init__(target)
        self._governor_tunables = {}

    @memoized
    def list_governors(self, cpu):
        """Returns a list of governors supported by the cpu."""
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        sysfile = '/sys/devices/system/cpu/{}/cpufreq/scaling_available_governors'.format(cpu)
        output = self.target.read_value(sysfile)
        return output.strip().split()

    def get_governor(self, cpu):
        """Returns the governor currently set for the specified CPU."""
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        sysfile = '/sys/devices/system/cpu/{}/cpufreq/scaling_governor'.format(cpu)
        return self.target.read_value(sysfile)

    def set_governor(self, cpu, governor, **kwargs):
        """
        Set the governor for the specified CPU.
        See https://www.kernel.org/doc/Documentation/cpu-freq/governors.txt

        :param cpu: The CPU for which the governor is to be set. This must be
                    the full name as it appears in sysfs, e.g. "cpu0".
        :param governor: The name of the governor to be used. This must be
                         supported by the specific device.

        Additional keyword arguments can be used to specify governor tunables for
        governors that support them.

        :note: On big.LITTLE all cores in a cluster must be using the same governor.
               Setting the governor on any core in a cluster will also set it on all
               other cores in that cluster.

        :raises: TargetError if governor is not supported by the CPU, or if,
                 for some reason, the governor could not be set.

        """
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        supported = self.list_governors(cpu)
        if governor not in supported:
            raise TargetError('Governor {} not supported for cpu {}'.format(governor, cpu))
        sysfile = '/sys/devices/system/cpu/{}/cpufreq/scaling_governor'.format(cpu)
        self.target.write_value(sysfile, governor)
        self.set_governor_tunables(cpu, governor, **kwargs)

    def list_governor_tunables(self, cpu):
        """Returns a list of tunables available for the governor on the specified CPU."""
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        governor = self.get_governor(cpu)
        if governor not in self._governor_tunables:
            try:
                tunables_path = '/sys/devices/system/cpu/{}/cpufreq/{}'.format(cpu, governor)
                self._governor_tunables[governor] = self.target.list_directory(tunables_path)
            except TargetError:  # probably an older kernel
                try:
                    tunables_path = '/sys/devices/system/cpu/cpufreq/{}'.format(governor)
                    self._governor_tunables[governor] = self.target.list_directory(tunables_path)
                except TargetError:  # governor does not support tunables
                    self._governor_tunables[governor] = []
        return self._governor_tunables[governor]

    def get_governor_tunables(self, cpu):
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        governor = self.get_governor(cpu)
        tunables = {}
        for tunable in self.list_governor_tunables(cpu):
            if tunable not in WRITE_ONLY_TUNABLES.get(governor, []):
                try:
                    path = '/sys/devices/system/cpu/{}/cpufreq/{}/{}'.format(cpu, governor, tunable)
                    tunables[tunable] = self.target.read_value(path)
                except TargetError:  # May be an older kernel
                    path = '/sys/devices/system/cpu/cpufreq/{}/{}'.format(governor, tunable)
                    tunables[tunable] = self.target.read_value(path)
        return tunables

    def set_governor_tunables(self, cpu, governor=None, **kwargs):
        """
        Set tunables for the specified governor. Tunables should be specified as
        keyword arguments. Which tunables and values are valid depends on the
        governor.

        :param cpu: The cpu for which the governor will be set. ``int`` or
                    full cpu name as it appears in sysfs, e.g. ``cpu0``.
        :param governor: The name of the governor. Must be all lower case.

        The rest should be keyword parameters mapping tunable name onto the value to
        be set for it.

        :raises: TargetError if governor specified is not a valid governor name, or if
                 a tunable specified is not valid for the governor, or if could not set
                 tunable.

        """
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        if governor is None:
            governor = self.get_governor(cpu)
        valid_tunables = self.list_governor_tunables(cpu)
        for tunable, value in kwargs.iteritems():
            if tunable in valid_tunables:
                path = '/sys/devices/system/cpu/{}/cpufreq/{}/{}'.format(cpu, governor, tunable)
                try:
                    self.target.write_value(path, value)
                except TargetError:
                    if self.target.file_exists(path):
                        # File exists but we did something wrong
                        raise
                    # Expected file doesn't exist, try older sysfs layout.
                    path = '/sys/devices/system/cpu/cpufreq/{}/{}'.format(governor, tunable)
                    self.target.write_value(path, value)
            else:
                message = 'Unexpected tunable {} for governor {} on {}.\n'.format(tunable, governor, cpu)
                message += 'Available tunables are: {}'.format(valid_tunables)
                raise TargetError(message)

    @memoized
    def list_frequencies(self, cpu):
        """Returns a list of frequencies supported by the cpu or an empty list
        if not could be found."""
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        try:
            cmd = 'cat /sys/devices/system/cpu/{}/cpufreq/scaling_available_frequencies'.format(cpu)
            output = self.target.execute(cmd)
            available_frequencies = map(int, output.strip().split())  # pylint: disable=E1103
        except TargetError:
            # On some devices scaling_frequencies  is not generated.
            # http://adrynalyne-teachtofish.blogspot.co.uk/2011/11/how-to-enable-scalingavailablefrequenci.html
            # Fall back to parsing stats/time_in_state
            cmd = 'cat /sys/devices/system/cpu/{}/cpufreq/stats/time_in_state'.format(cpu)
            out_iter = iter(self.target.execute(cmd).strip().split())
            available_frequencies = map(int, reversed([f for f, _ in zip(out_iter, out_iter)]))
        return available_frequencies

    def get_min_frequency(self, cpu):
        """
        Returns the min frequency currently set for the specified CPU.

        Warning, this method does not check if the cpu is online or not. It will
        try to read the minimum frequency and the following exception will be
        raised ::

        :raises: TargetError if for some reason the frequency could not be read.

        """
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        sysfile = '/sys/devices/system/cpu/{}/cpufreq/scaling_min_freq'.format(cpu)
        return self.target.read_int(sysfile)

    def set_min_frequency(self, cpu, frequency, exact=True):
        """
        Set's the minimum value for CPU frequency. Actual frequency will
        depend on the Governor used and may vary during execution. The value should be
        either an int or a string representing an integer. The Value must also be
        supported by the device. The available frequencies can be obtained by calling
        get_frequencies() or examining

        /sys/devices/system/cpu/cpuX/cpufreq/scaling_frequencies

        on the device.

        :raises: TargetError if the frequency is not supported by the CPU, or if, for
                 some reason, frequency could not be set.
        :raises: ValueError if ``frequency`` is not an integer.

        """
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        available_frequencies = self.list_frequencies(cpu)
        try:
            value = int(frequency)
            if exact and available_frequencies and value not in available_frequencies:
                raise TargetError('Can\'t set {} frequency to {}\nmust be in {}'.format(cpu,
                                                                                        value,
                                                                                        available_frequencies))
            sysfile = '/sys/devices/system/cpu/{}/cpufreq/scaling_min_freq'.format(cpu)
            self.target.write_value(sysfile, value)
        except ValueError:
            raise ValueError('Frequency must be an integer; got: "{}"'.format(frequency))

    def get_frequency(self, cpu):
        """
        Returns the current frequency currently set for the specified CPU.

        Warning, this method does not check if the cpu is online or not. It will
        try to read the current frequency and the following exception will be
        raised ::

        :raises: TargetError if for some reason the frequency could not be read.

        """
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        sysfile = '/sys/devices/system/cpu/{}/cpufreq/scaling_cur_freq'.format(cpu)
        return self.target.read_int(sysfile)

    def set_frequency(self, cpu, frequency, exact=True):
        """
        Set's the minimum value for CPU frequency. Actual frequency will
        depend on the Governor used and may vary during execution. The value should be
        either an int or a string representing an integer.

        If ``exact`` flag is set (the default), the Value must also be supported by
        the device. The available frequencies can be obtained by calling
        get_frequencies() or examining

        /sys/devices/system/cpu/cpuX/cpufreq/scaling_frequencies

        on the device (if it exists).

        :raises: TargetError if the frequency is not supported by the CPU, or if, for
                 some reason, frequency could not be set.
        :raises: ValueError if ``frequency`` is not an integer.

        """
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        try:
            value = int(frequency)
            if exact:
                available_frequencies = self.list_frequencies(cpu)
                if available_frequencies and value not in available_frequencies:
                    raise TargetError('Can\'t set {} frequency to {}\nmust be in {}'.format(cpu,
                                                                                            value,
                                                                                            available_frequencies))
            if self.get_governor(cpu) != 'userspace':
                raise TargetError('Can\'t set {} frequency; governor must be "userspace"'.format(cpu))
            sysfile = '/sys/devices/system/cpu/{}/cpufreq/scaling_setspeed'.format(cpu)
            self.target.write_value(sysfile, value, verify=False)
        except ValueError:
            raise ValueError('Frequency must be an integer; got: "{}"'.format(frequency))

    def get_max_frequency(self, cpu):
        """
        Returns the max frequency currently set for the specified CPU.

        Warning, this method does not check if the cpu is online or not. It will
        try to read the maximum frequency and the following exception will be
        raised ::

        :raises: TargetError if for some reason the frequency could not be read.
        """
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        sysfile = '/sys/devices/system/cpu/{}/cpufreq/scaling_max_freq'.format(cpu)
        return self.target.read_int(sysfile)

    def set_max_frequency(self, cpu, frequency, exact=True):
        """
        Set's the minimum value for CPU frequency. Actual frequency will
        depend on the Governor used and may vary during execution. The value should be
        either an int or a string representing an integer. The Value must also be
        supported by the device. The available frequencies can be obtained by calling
        get_frequencies() or examining

        /sys/devices/system/cpu/cpuX/cpufreq/scaling_frequencies

        on the device.

        :raises: TargetError if the frequency is not supported by the CPU, or if, for
                 some reason, frequency could not be set.
        :raises: ValueError if ``frequency`` is not an integer.

        """
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        available_frequencies = self.list_frequencies(cpu)
        try:
            value = int(frequency)
            if exact and available_frequencies and value not in available_frequencies:
                raise TargetError('Can\'t set {} frequency to {}\nmust be in {}'.format(cpu,
                                                                                        value,
                                                                                        available_frequencies))
            sysfile = '/sys/devices/system/cpu/{}/cpufreq/scaling_max_freq'.format(cpu)
            self.target.write_value(sysfile, value)
        except ValueError:
            raise ValueError('Frequency must be an integer; got: "{}"'.format(frequency))

    def set_governor_for_cpus(self, cpus, governor, **kwargs):
        """
        Set the governor for the specified list of CPUs.
        See https://www.kernel.org/doc/Documentation/cpu-freq/governors.txt

        :param cpus: The list of CPU for which the governor is to be set.
        """
        for cpu in cpus:
            self.set_governor(cpu, governor, **kwargs)

    def set_frequency_for_cpus(self, cpus, freq, exact=False):
        """
        Set the frequency for the specified list of CPUs.
        See https://www.kernel.org/doc/Documentation/cpu-freq/governors.txt

        :param cpus: The list of CPU for which the frequency has to be set.
        """
        for cpu in cpus:
            self.set_frequency(cpu, freq, exact)

    def set_all_frequencies(self, freq):
        """
        Set the specified (minimum) frequency for all the (online) CPUs
        """
        return self.target._execute_util(
                'cpufreq_set_all_frequencies {}'.format(freq),
                as_root=True)

    def get_all_frequencies(self):
        """
        Get the current frequency for all the (online) CPUs
        """
        output = self.target._execute_util(
                'cpufreq_get_all_frequencies', as_root=True)
        frequencies = {}
        for x in output.splitlines():
            kv = x.split(' ')
            if kv[0] == '':
                break
            frequencies[kv[0]] = kv[1]
        return frequencies

    def set_all_governors(self, governor):
        """
        Set the specified governor for all the (online) CPUs
        """
        try:
            return self.target._execute_util(
                'cpufreq_set_all_governors {}'.format(governor),
                as_root=True)
        except TargetError as e:
            if ("echo: I/O error" in str(e) or
                "write error: Invalid argument" in str(e)):

                cpus_unsupported = [c for c in self.target.list_online_cpus()
                                    if governor not in self.list_governors(c)]
                raise TargetError("Governor {} unsupported for CPUs {}".format(
                    governor, cpus_unsupported))
            else:
                raise

    def get_all_governors(self):
        """
        Get the current governor for all the (online) CPUs
        """
        output = self.target._execute_util(
                'cpufreq_get_all_governors', as_root=True)
        governors = {}
        for x in output.splitlines():
            kv = x.split(' ')
            if kv[0] == '':
                break
            governors[kv[0]] = kv[1]
        return governors

    def trace_frequencies(self):
        """
        Report current frequencies on trace file
        """
        return self.target._execute_util('cpufreq_trace_all_frequencies', as_root=True)

    @memoized
    def get_domain_cpus(self, cpu):
        """
        Get the CPUs that share a frequency domain with the given CPU
        """
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)

        sysfile = '/sys/devices/system/cpu/{}/cpufreq/affected_cpus'.format(cpu)

        return [int(c) for c in self.target.read_value(sysfile).split()]

    def iter_domains(self):
        """
        Iterate over the frequency domains in the system
        """
        cpus = set(range(self.target.number_of_cpus))
        while cpus:
            cpu = iter(cpus).next()
            domain = self.target.cpufreq.get_domain_cpus(cpu)
            yield domain
            cpus = cpus.difference(domain)
