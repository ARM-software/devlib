#    Copyright 2021 ARM Limited
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
from __future__ import division

import logging
import re
import sys
import tempfile
import threading
import time

from devlib.instrument import Instrument, MeasurementsCsv, CONTINUOUS
from devlib.exception import TargetStableError, TargetNotRespondingError, WorkerThreadError
from devlib.utils.csvutil import csvreader, csvwriter


logger = logging.getLogger('BattryMonitor')


BATTERY_STATS_REGEX = re.compile(r'level: (?P<level>\d+).*scale: (?P<scale>\d+).*voltage: (?P<voltage>\d+)', flags=re.DOTALL)


# List of SysFS node for current taken from Ampere Application
# https://forum.xda-developers.com/t/app-4-0-3-ampere-current-meter.3040329/post-59086006
CURRENT_SYSFS_NODES = [
    '/sys/class/power_supply/ab8500_fg/current_now',
    '/sys/class/power_supply/android-battery/current_now',
    '/sys/class/power_supply/battery/batt_attr_text',
    '/sys/class/power_supply/battery/batt_chg_current',
    '/sys/class/power_supply/battery/batt_current',
    '/sys/class/power_supply/battery/batt_current_adc',
    '/sys/class/power_supply/battery/batt_current_now',
    '/sys/class/power_supply/battery/BatteryAverageCurrent',
    '/sys/class/power_supply/battery/charger_current',
    '/sys/class/power_supply/battery/current_avg',
    '/sys/class/power_supply/battery/current_max',
    '/sys/class/power_supply/battery/current_now',
    '/sys/class/power_supply/Battery/current_now',
    '/sys/class/power_supply/battery/smem_text',
    '/sys/class/power_supply/bq27520/current_now',
    '/sys/class/power_supply/da9052-bat/current_avg',
    '/sys/class/power_supply/ds2784-fuelgauge/current_now',
    '/sys/class/power_supply/max17042-0/current_now',
    '/sys/class/power_supply/max170xx_battery/current_now',
    '/sys/devices/platform/battery/power_supply/battery/BatteryAverageCurrent',
    '/sys/devices/platform/cpcap_battery/power_supply/usb/current_now',
    '/sys/devices/platform/ds2784-battery/getcurrent',
    '/sys/devices/platform/i2c-adapter/i2c-0/0-0036/power_supply/battery/current_now',
    '/sys/devices/platform/i2c-adapter/i2c-0/0-0036/power_supply/ds2746-battery/current_now',
    '/sys/devices/platform/msm-charger/power_supply/battery_gauge/current_now',
    '/sys/devices/platform/mt6320-battery/power_supply/battery/BatteryAverageCurrent',
    '/sys/devices/platform/mt6329-battery/FG_Battery_CurrentConsumption',
    '/sys/EcControl/BatCurrent',
]


class BatteryMonitorInstrument(Instrument):

    name = 'battey_monitor'
    mode = CONTINUOUS

    def __init__(self, target, period=2, current_scale=1e6,
                 voltage_scale=1e3, current_node=None,):

        if not target.is_rooted:
            self.logger.warn('Target is not rooted, current readings are likely to fail.')
        super(BatteryMonitorInstrument, self).__init__(target)

        self.period = period
        self.target = target

        self.logger.debug('Discovering available current sysfs node..')
        self.current_node, inverse = self._discover_current_node(current_node)

        # sensor kind --> unit conversion
        self.value_convert = {
            'voltage': lambda x: int(x) / voltage_scale,
            'current': lambda x: (-int(x) if inverse else int(x)) / current_scale,
            'power': lambda x: -int(x) / (voltage_scale * current_scale),
            'percent': lambda x: float(x),
            'time': lambda x: x,
        }

        self.add_channel('battery', 'voltage')
        self.add_channel('battery', 'current')
        self.add_channel('battery', 'power')
        self.add_channel('battery', 'percent')
        self.add_channel('timestamp', 'time')

    def reset(self, sites=None, kinds=None, channels=None):
        super(BatteryMonitorInstrument, self).reset(sites, kinds, channels)
        self.raw_data_file = tempfile.mkstemp('.csv')[1]
        self.collector = BatteryStatsCollector(self.target,
                                               self.period,
                                               self.current_node,
                                               self.raw_data_file)

    def start(self):
        if not self.collector:
            raise RuntimeError('Must call "reset" before "start"')
        self.collector.start()

    def stop(self):
        self.collector.stop()

    def _discover_current_node(self, current_node):
        paths = [current_node] if current_node else CURRENT_SYSFS_NODES
        for path in paths:
            try:
                reading = self.target.read_int(path)
            except TargetStableError:
                continue
            if reading:
                self.logger.debug('Found current sysfs node at: {}'.format(path))
                # Return if the value reported is negative or positive, assuming
                # device is currently discharging
                return path, reading < 0

        raise RuntimeError('Failed to detect valid reading from known current nodes.')


    def get_data(self, outfile):
        all_channels = self.list_channels()
        channels_labels = [c.label for c in all_channels]
        active_channels = [c.label for c in self.active_channels]
        active_indexes = [channels_labels.index(ac) for ac in active_channels]

        with csvreader(self.raw_data_file, skipinitialspace=True) as reader:
            with csvwriter(outfile) as writer:
                writer.writerow(active_channels)
                for row in reader:
                    output_row = [self.value_convert[all_channels[i].kind](row[i]) for i in active_indexes]
                    writer.writerow(output_row)

        return MeasurementsCsv(outfile, self.active_channels, 1/self.period)


class BatteryStatsCollector(threading.Thread):

    def __init__(self, target, period, current_node, raw_data_file):
        super(BatteryStatsCollector, self).__init__()
        self.target = target
        self.period = period
        self.current_node = current_node
        self.raw_data_file = raw_data_file
        self.stop_signal = threading.Event()
        self.measurements = []
        self.exc = None

    def run(self):
        logger.debug('Battery stats collection started.')
        try:
            self.stop_signal.clear()
            logger.debug('Using temp file: {}'.format(self.raw_data_file))
            wfh = open(self.raw_data_file, 'wb')
            try:
                while not self.stop_signal.is_set():
                    self.collect_stats(wfh)
                    time.sleep(self.period)
            finally:
                wfh.close()
        except (TargetNotRespondingError, TimeoutError):
            raise
        except Exception as e:
            logger.warning('Exception on collector thread: {}({})'.format(e.__class__.__name__, e))
            self.exc = WorkerThreadError(self.name, sys.exc_info())
        logger.debug('Battery stats collection stopped.')

    def collect_stats(self, wfh):
        voltage, batt_pct = self._get_battery_stats()
        current = self._measure_current()
        power = voltage * current
        results = ','.join(map(str, [voltage, current, power, batt_pct, time.time()]))
        wfh.write('{}\n'.format(results).encode('utf-8'))

    def _get_battery_stats(self):
        output = self.target.execute('dumpsys battery')
        match = BATTERY_STATS_REGEX.search(output)
        voltage = int(match.group('voltage'))
        batt_pct = (int(match.group('level'))*100)/int(match.group('scale'))
        return voltage, batt_pct

    def _measure_current(self):
        return self.target.read_int(self.current_node)

    def stop(self):
        self.stop_signal.set()
        self.join()
        if self.exc:
            raise self.exc  # pylint: disable=E0702
