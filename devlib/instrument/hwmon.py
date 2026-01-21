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
#
import re

from devlib.instrument import Instrument, Measurement, INSTANTANEOUS
from devlib.exception import TargetStableError
from typing import (Dict, Tuple, Callable, Union, TYPE_CHECKING,
                    cast, List)
from devlib.module.hwmon import HwmonModule, HwmonSensor
if TYPE_CHECKING:
    from devlib.target import Target


class HwmonInstrument(Instrument):

    name: str = 'hwmon'
    mode: int = INSTANTANEOUS

    # sensor kind --> (meaure, standard unit conversion)
    measure_map: Dict[str, Tuple[str, Callable[[Union[int, float]], float]]] = {
        'temp': ('temperature', lambda x: x / 1000),
        'in': ('voltage', lambda x: x / 1000),
        'curr': ('current', lambda x: x / 1000),
        'power': ('power', lambda x: x / 1000000),
        'energy': ('energy', lambda x: x / 1000000),
    }

    def __init__(self, target: 'Target'):
        if not hasattr(target, 'hwmon'):
            raise TargetStableError('Target does not support HWMON')
        super(HwmonInstrument, self).__init__(target)

        self.logger.debug('Discovering available HWMON sensors...')
        for ts in cast(HwmonModule, self.target.hwmon).sensors:
            try:
                ts.get_file('input')
                measure_map = self.measure_map.get(ts.kind)
                if measure_map:
                    measure: str = measure_map[0]
                if measure:
                    self.logger.debug('\tAdding sensor {}'.format(ts.name))
                    self.add_channel(_guess_site(ts), measure, sensor=ts)
                else:
                    self.logger.debug('\tSkipping sensor {} (unknown kind "{}")'.format(ts.name, ts.kind))
            except ValueError:
                message = 'Skipping sensor {} because it does not have an input file'
                self.logger.debug(message.format(ts.name))
                continue

    def take_measurement(self) -> List[Measurement]:
        result: List[Measurement] = []
        for chan in self.active_channels:
            convert = self.measure_map[chan.sensor.kind][1]  # type: ignore
            value = convert(chan.sensor.get('input'))  # type: ignore
            result.append(Measurement(value, chan))
        return result


def _guess_site(sensor: HwmonSensor):
    """
    HWMON does not specify a standard for labeling its sensors, or for
    device/item split (the implication is that each hwmon device a separate chip
    with possibly several sensors on it, but not everyone adheres to that, e.g.,
    with some mobile devices splitting a chip's sensors across multiple hwmon
    devices.  This function processes name/label of the senors to attempt to
    identify the best "candidate" for the site to which the sensor belongs.
    """
    if sensor.name == sensor.label:
        # If no label has been specified for the sensor (in which case, it
        # defaults to the sensor's name), assume that the "site" of the sensor
        # is identified by the HWMON device
        text: str = sensor.device.name
    else:
        # If a label has been specified, assume multiple sensors controlled by
        # the same device and the label identifies the site.
        text = sensor.label
    # strip out sensor kind suffix, if any, as that does not indicate a site
    for kind in ['volt', 'in', 'curr', 'power', 'energy',
                 'temp', 'voltage', 'temperature', 'current']:
        if kind in text.lower():
            regex = re.compile(r'_*{}\d*_*'.format(kind), re.I)
            text = regex.sub('', text)
    return text.strip()
