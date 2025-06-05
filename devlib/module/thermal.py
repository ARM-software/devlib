#    Copyright 2015-2025 ARM Limited
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

import re
import logging
import devlib.utils.asyn as asyn

from devlib.utils.misc import get_logger
from devlib.module import Module
from devlib.exception import TargetStableCalledProcessError
from typing import (TYPE_CHECKING, Dict, Optional,
                    Tuple, List)
if TYPE_CHECKING:
    from devlib.target import Target


class TripPoint(object):
    """
    Trip points are predefined temperature thresholds within a thermal zone. When the temperature reaches these points,
    specific actions are triggered to manage the system's thermal state. There are typically three types of trip points:

    Active Trip Points: Trigger active cooling mechanisms like fans when the temperature exceeds a certain threshold.
    Passive Trip Points: Initiate passive cooling strategies, such as reducing the processor's clock speed, to lower the temperature.
    Critical Trip Points: Indicate a critical temperature level that requires immediate action, such as shutting down the system to prevent damage
    """
    def __init__(self, zone: 'ThermalZone', _id: str):
        self._id = _id
        self.zone = zone
        self.temp_node: str = 'trip_point_' + _id + '_temp'
        self.type_node: str = 'trip_point_' + _id + '_type'

    @property
    def target(self) -> 'Target':
        """
        target of the trip point
        """
        return self.zone.target

    @asyn.asyncf
    async def get_temperature(self) -> int:
        """Returns the currently configured temperature of the trip point"""
        temp_file: str = self.target.path.join(self.zone.path, self.temp_node)
        return await self.target.read_int.asyn(temp_file)

    @asyn.asyncf
    async def set_temperature(self, temperature: int) -> None:
        """
        set temperature threshold for the trip point
        """
        temp_file: str = self.target.path.join(self.zone.path, self.temp_node)
        await self.target.write_value.asyn(temp_file, temperature)

    @asyn.asyncf
    async def get_type(self) -> str:
        """Returns the type of trip point"""
        type_file: str = self.target.path.join(self.zone.path, self.type_node)
        return await self.target.read_value.asyn(type_file)


class ThermalZone(object):
    """
    A thermal zone is a logical collection of interfaces to temperature sensors, trip points,
    thermal property information, and thermal controls. These zones help manage the temperature
    of various components within a system, such as CPUs, GPUs, and other hardware.
    """
    def __init__(self, target: 'Target', root: str, _id: str):
        self.target = target
        self.name = 'thermal_zone' + _id
        self.path = target.path.join(root, self.name)
        self.trip_points: Dict[int, TripPoint] = {}
        self.type: str = self.target.read_value(self.target.path.join(self.path, 'type'))

        for entry in self.target.list_directory(self.path, as_root=target.is_rooted):
            re_match = re.match('^trip_point_([0-9]+)_temp', entry)
            if re_match is not None:
                self._add_trip_point(re_match.group(1))

    def _add_trip_point(self, _id: str) -> None:
        """
        add a trip point to the thermal zone
        """
        self.trip_points[int(_id)] = TripPoint(self, _id)

    @asyn.asyncf
    async def is_enabled(self) -> bool:
        """Returns a boolean representing the 'mode' of the thermal zone"""
        value: str = await self.target.read_value.asyn(self.target.path.join(self.path, 'mode'))
        return value == 'enabled'

    @asyn.asyncf
    async def set_enabled(self, enabled: bool = True) -> None:
        """
        enable or disable the thermal zone
        """
        value = 'enabled' if enabled else 'disabled'
        await self.target.write_value.asyn(self.target.path.join(self.path, 'mode'), value)

    @asyn.asyncf
    async def get_temperature(self) -> int:
        """Returns the temperature of the thermal zone"""
        sysfs_temperature_file = self.target.path.join(self.path, 'temp')
        return await self.target.read_int.asyn(sysfs_temperature_file)

    @asyn.asyncf
    async def get_policy(self) -> str:
        """Returns the policy of the thermal zone"""
        temp_file = self.target.path.join(self.path, 'policy')
        return await self.target.read_value.asyn(temp_file)

    @asyn.asyncf
    async def set_policy(self, policy: str) -> None:
        """
        Sets the policy of the thermal zone

        :params policy: Thermal governor name
        """
        await self.target.write_value.asyn(self.target.path.join(self.path, 'policy'), policy)

    @asyn.asyncf
    async def get_offset(self) -> int:
        """Returns the temperature offset of the thermal zone"""
        offset_file: str = self.target.path.join(self.path, 'offset')
        return await self.target.read_value.asyn(offset_file)

    @asyn.asyncf
    async def set_offset(self, offset: int) -> None:
        """
        Sets the temperature offset in milli-degrees of the thermal zone

        :params offset: Temperature offset in milli-degrees
        """
        await self.target.write_value.asyn(self.target.path.join(self.path, 'offset'), offset)

    @asyn.asyncf
    async def set_emul_temp(self, offset: int) -> None:
        """
        Sets the emulated temperature in milli-degrees of the thermal zone

        :params offset: Emulated temperature in milli-degrees
        """
        await self.target.write_value.asyn(self.target.path.join(self.path, 'emul_temp'), offset)

    @asyn.asyncf
    async def get_available_policies(self) -> str:
        """Returns the policies available for the thermal zone"""
        temp_file: str = self.target.path.join(self.path, 'available_policies')
        return await self.target.read_value.asyn(temp_file)


class ThermalModule(Module):
    """
    The /sys/class/thermal directory in Linux provides a sysfs interface for thermal management.
    This directory contains subdirectories and files that represent thermal zones and cooling devices,
    allowing users and applications to monitor and manage system temperatures.
    """
    name = 'thermal'
    thermal_root = '/sys/class/thermal'

    @staticmethod
    def probe(target: 'Target') -> bool:
        if target.file_exists(ThermalModule.thermal_root):
            return True
        return False

    def __init__(self, target: 'Target'):
        super(ThermalModule, self).__init__(target)

        self.logger: logging.Logger = get_logger(self.name)
        self.logger.debug('Initialized [%s] module', self.name)

        self.zones: Dict[int, ThermalZone] = {}
        self.cdevs: List = []

        for entry in target.list_directory(self.thermal_root):
            re_match = re.match('^(thermal_zone|cooling_device)([0-9]+)', entry)
            if not re_match:
                self.logger.warning('unknown thermal entry: %s', entry)
                continue

            if re_match.group(1) == 'thermal_zone':
                self._add_thermal_zone(re_match.group(2))
            elif re_match.group(1) == 'cooling_device':
                # TODO
                pass

    def _add_thermal_zone(self, _id: str) -> None:
        self.zones[int(_id)] = ThermalZone(self.target, self.thermal_root, _id)

    def disable_all_zones(self) -> None:
        """Disables all the thermal zones in the target"""
        for zone in self.zones.values():
            zone.set_enabled(False)

    @asyn.asyncf
    async def get_all_temperatures(self, error: str = 'raise') -> Dict[str, int]:
        """
        Returns dictionary with current reading of all thermal zones.

        :params error: Sensor read error handling (raise or ignore)

        :returns: a dictionary in the form: {tz_type:temperature}
        """

        async def get_temperature_noexcep(item: Tuple[str, ThermalZone]) -> Optional[int]:
            tzid, tz = item
            try:
                temperature: int = await tz.get_temperature.asyn()
            except TargetStableCalledProcessError as e:
                if error == 'raise':
                    raise e
                elif error == 'ignore':
                    self.logger.warning(f'Skipping thermal_zone_id={tzid} thermal_zone_type={tz.type} error="{e}"')
                    return None
                else:
                    raise ValueError(f'Unknown error parameter value: {error}')
            return temperature

        tz_temps = await self.target.async_manager.map_concurrently(get_temperature_noexcep, self.zones.items())

        return {tz.type: temperature for (tzid, tz), temperature in tz_temps.items() if temperature is not None}
