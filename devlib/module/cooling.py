#    Copyright 2014-2025 ARM Limited
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
from devlib.utils.serial_port import open_serial_connection
from typing import TYPE_CHECKING, cast
from pexpect import fdpexpect
if TYPE_CHECKING:
    from devlib.target import Target


class MbedFanActiveCoolingModule(Module):
    """
    Module to control active cooling using fan
    """
    name: str = 'mbed-fan'
    timeout: int = 30

    @staticmethod
    def probe(target: 'Target') -> bool:
        return True

    def __init__(self, target: 'Target', port: str = '/dev/ttyACM0', baud: int = 115200, fan_pin: int = 0):
        super(MbedFanActiveCoolingModule, self).__init__(target)
        self.port = port
        self.baud = baud
        self.fan_pin = fan_pin

    def start(self) -> None:
        """
        send motor start to fan
        """
        with open_serial_connection(timeout=self.timeout,
                                    port=self.port,
                                    baudrate=self.baud) as target:
            # pylint: disable=no-member
            cast(fdpexpect.fdspawn, target).sendline('motor_{}_1'.format(self.fan_pin))

    def stop(self) -> None:
        """
        send motor stop to fan
        """
        with open_serial_connection(timeout=self.timeout,
                                    port=self.port,
                                    baudrate=self.baud) as target:
            # pylint: disable=no-member
            cast(fdpexpect.fdspawn, target).sendline('motor_{}_0'.format(self.fan_pin))


class OdroidXU3ctiveCoolingModule(Module):

    name: str = 'odroidxu3-fan'

    @staticmethod
    def probe(target: 'Target') -> bool:
        return target.file_exists('/sys/devices/odroid_fan.15/fan_mode')

    def start(self) -> None:
        """
        start fan
        """
        self.target.write_value('/sys/devices/odroid_fan.15/fan_mode', 0, verify=False)
        self.target.write_value('/sys/devices/odroid_fan.15/pwm_duty', 255, verify=False)

    def stop(self):
        """
        stop fan
        """
        self.target.write_value('/sys/devices/odroid_fan.15/fan_mode', 0, verify=False)
        self.target.write_value('/sys/devices/odroid_fan.15/pwm_duty', 1, verify=False)
