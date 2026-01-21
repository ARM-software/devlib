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
from devlib.module import Module
from devlib.exception import TargetStableError
from devlib.utils.misc import memoized
from typing import TYPE_CHECKING, List, Union, Dict
if TYPE_CHECKING:
    from devlib.target import Target


class DevfreqModule(Module):
    """
    The devfreq framework in Linux is used for dynamic voltage and frequency scaling (DVFS) of various devices.
    """
    name = 'devfreq'

    @staticmethod
    def probe(target: 'Target') -> bool:
        path: str = '/sys/class/devfreq/'
        if not target.file_exists(path):
            return False

        # Check that at least one policy is implemented
        if not target.list_directory(path):
            return False

        return True

    @memoized
    def list_devices(self) -> List[str]:
        """Returns a list of devfreq devices supported by the target platform."""
        sysfile = '/sys/class/devfreq/'
        return self.target.list_directory(sysfile)

    @memoized
    def list_governors(self, device: str) -> List[str]:
        """Returns a list of governors supported by the device."""
        sysfile: str = '/sys/class/devfreq/{}/available_governors'.format(device)
        output: str = self.target.read_value(sysfile)
        return output.strip().split()

    def get_governor(self, device: Union[str, int]) -> str:
        """Returns the governor currently set for the specified device."""
        if isinstance(device, int):
            device = 'device{}'.format(device)
        sysfile = '/sys/class/devfreq/{}/governor'.format(device)
        return self.target.read_value(sysfile)

    def set_governor(self, device: str, governor: str) -> None:
        """
        Set the governor for the specified device.

        :param device: The device for which the governor is to be set. This must be
                    the full name as it appears in sysfs, e.g. "e82c0000.mali".
        :param governor: The name of the governor to be used. This must be
                         supported by the specific device.

        Additional keyword arguments can be used to specify governor tunables for
        governors that support them.

        :raises: TargetStableError if governor is not supported by the device, or if,
                 for some reason, the governor could not be set.

        """
        supported: List[str] = self.list_governors(device)
        if governor not in supported:
            raise TargetStableError('Governor {} not supported for device {}'.format(governor, device))
        sysfile: str = '/sys/class/devfreq/{}/governor'.format(device)
        self.target.write_value(sysfile, governor)

    @memoized
    def list_frequencies(self, device: str) -> List[int]:
        """
        Returns a list of frequencies supported by the device or an empty list
        if could not be found.
        """
        cmd: str = 'cat /sys/class/devfreq/{}/available_frequencies'.format(device)
        output: str = self.target.execute(cmd)
        available_frequencies: List[int] = [int(freq) for freq in output.strip().split()]

        return available_frequencies

    def get_min_frequency(self, device: str) -> int:
        """
        Returns the min frequency currently set for the specified device.

        Warning, this method does not check if the device is present or not. It
        will try to read the minimum frequency and the following exception will
        be raised ::

        :raises: TargetStableError if for some reason the frequency could not be read.

        """
        sysfile = '/sys/class/devfreq/{}/min_freq'.format(device)
        return self.target.read_int(sysfile)

    def set_min_frequency(self, device: str, frequency: Union[int, str], exact: bool = True) -> None:
        """
        Sets the minimum value for device frequency. Actual frequency will
        depend on the thermal governor used and may vary during execution. The
        value should be either an int or a string representing an integer. The
        Value must also be supported by the device. The available frequencies
        can be obtained by calling list_frequencies() or examining

        /sys/class/devfreq/<device_name>/available_frequencies

        on the device.

        :raises: TargetStableError if the frequency is not supported by the device, or if, for
                 some reason, frequency could not be set.
        :raises: ValueError if ``frequency`` is not an integer.

        """
        available_frequencies: List[int] = self.list_frequencies(device)
        try:
            value = int(frequency)
            if exact and available_frequencies and value not in available_frequencies:
                raise TargetStableError('Can\'t set {} frequency to {}\nmust be in {}'.format(device,
                                                                                              value,
                                                                                              available_frequencies))
            sysfile: str = '/sys/class/devfreq/{}/min_freq'.format(device)
            self.target.write_value(sysfile, value)
        except ValueError:
            raise ValueError('Frequency must be an integer; got: "{}"'.format(frequency))

    def get_frequency(self, device: str) -> int:
        """
        Returns the current frequency currently set for the specified device.

        Warning, this method does not check if the device is present or not. It
        will try to read the current frequency and the following exception will
        be raised ::

        :raises: TargetStableError if for some reason the frequency could not be read.

        """
        sysfile: str = '/sys/class/devfreq/{}/cur_freq'.format(device)
        return self.target.read_int(sysfile)

    def get_max_frequency(self, device: str) -> int:
        """
        Returns the max frequency currently set for the specified device.

        Warning, this method does not check if the device is online or not. It will
        try to read the maximum frequency and the following exception will be
        raised ::

        :raises: TargetStableError if for some reason the frequency could not be read.
        """
        sysfile: str = '/sys/class/devfreq/{}/max_freq'.format(device)
        return self.target.read_int(sysfile)

    def set_max_frequency(self, device: str, frequency: Union[int, str], exact: bool = True) -> None:
        """
        Sets the maximum value for device frequency. Actual frequency will
        depend on the Governor used and may vary during execution. The value
        should be either an int or a string representing an integer. The Value
        must also be supported by the device. The available frequencies can be
        obtained by calling get_frequencies() or examining

        /sys/class/devfreq/<device_name>/available_frequencies

        on the device.

        :raises: TargetStableError if the frequency is not supported by the device, or
                 if, for some reason, frequency could not be set.
        :raises: ValueError if ``frequency`` is not an integer.

        """
        available_frequencies: List[int] = self.list_frequencies(device)
        try:
            value = int(frequency)
        except ValueError:
            raise ValueError('Frequency must be an integer; got: "{}"'.format(frequency))

        if exact and value not in available_frequencies:
            raise TargetStableError('Can\'t set {} frequency to {}\nmust be in {}'.format(device,
                                                                                          value,
                                                                                          available_frequencies))
        sysfile: str = '/sys/class/devfreq/{}/max_freq'.format(device)
        self.target.write_value(sysfile, value)

    def set_governor_for_devices(self, devices: List[str], governor: str) -> None:
        """
        Set the governor for the specified list of devices.

        :param devices: The list of device for which the governor is to be set.
        """
        for device in devices:
            self.set_governor(device, governor)

    def set_all_governors(self, governor: str) -> None:
        """
        Set the specified governor for all the (available) devices
        """
        try:
            return self.target._execute_util(  # pylint: disable=protected-access
                'devfreq_set_all_governors {}'.format(governor), as_root=True)
        except TargetStableError as e:
            if ("echo: I/O error" in str(e) or
               "write error: Invalid argument" in str(e)):

                devs_unsupported: List[str] = [d for d in self.list_devices()
                                               if governor not in self.list_governors(d)]
                raise TargetStableError("Governor {} unsupported for devices {}".format(
                    governor, devs_unsupported))
            else:
                raise

    def get_all_governors(self) -> Dict[str, str]:
        """
        Get the current governor for all the (online) CPUs
        """
        output: str = self.target._execute_util(  # pylint: disable=protected-access
            'devfreq_get_all_governors', as_root=True)
        governors: Dict[str, str] = {}
        for x in output.splitlines():
            kv = x.split(' ')
            if kv[0] == '':
                break
            governors[kv[0]] = kv[1]
        return governors

    def set_frequency_for_devices(self, devices: List[str], freq: Union[int, str], exact: bool = False) -> None:
        """
        Set the frequency for the specified list of devices.

        :param devices: The list of device for which the frequency has to be set.
        """
        for device in devices:
            self.set_max_frequency(device, freq, exact)
            self.set_min_frequency(device, freq, exact)

    def set_all_frequencies(self, freq: Union[int, str]) -> None:
        """
        Set the specified (minimum) frequency for all the (available) devices
        """
        return self.target._execute_util(  # pylint: disable=protected-access
            'devfreq_set_all_frequencies {}'.format(freq),
            as_root=True)

    def get_all_frequencies(self) -> Dict[str, str]:
        """
        Get the current frequency for all the (available) devices
        """
        output: str = self.target._execute_util(  # pylint: disable=protected-access
            'devfreq_get_all_frequencies', as_root=True)
        frequencies: Dict[str, str] = {}
        for x in output.splitlines():
            kv = x.split(' ')
            if kv[0] == '':
                break
            frequencies[kv[0]] = kv[1]
        return frequencies
