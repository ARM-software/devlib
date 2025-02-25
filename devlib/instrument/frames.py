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

from devlib.instrument import (Instrument, CONTINUOUS,
                               MeasurementsCsv, MeasurementType,
                               InstrumentChannel)
from devlib.utils.rendering import (GfxinfoFrameCollector,
                                    SurfaceFlingerFrameCollector,
                                    SurfaceFlingerFrame,
                                    read_gfxinfo_columns,
                                    FrameCollector)
from typing import (TYPE_CHECKING, Optional, Type,
                    OrderedDict, Any, List)
if TYPE_CHECKING:
    from devlib.target import Target


class FramesInstrument(Instrument):

    mode = CONTINUOUS
    collector_cls: Optional[Type[FrameCollector]] = None

    def __init__(self, target: 'Target', collector_target: Any,
                 period: int = 2, keep_raw: bool = True):
        super(FramesInstrument, self).__init__(target)
        self.collector_target = collector_target
        self.period = period
        self.keep_raw = keep_raw
        self.sample_rate_hz: float = 1 / self.period
        self.collector: Optional[FrameCollector] = None
        self.header: Optional[List[str]] = None
        self._need_reset: bool = True
        self._raw_file: Optional[str] = None
        self._init_channels()

    def reset(self, sites: Optional[List[str]] = None,
              kinds: Optional[List[str]] = None,
              channels: Optional[OrderedDict[str, InstrumentChannel]] = None) -> None:
        super(FramesInstrument, self).reset(sites, kinds, channels)
        if self.collector_cls:
            # pylint: disable=not-callable
            self.collector = self.collector_cls(self.target, self.period,
                                                self.collector_target, self.header)  # type: ignore
        self._need_reset = False
        self._raw_file = None

    def start(self) -> None:
        if self._need_reset:
            self.reset()
        if self.collector:
            self.collector.start()

    def stop(self) -> None:
        if self.collector:
            self.collector.stop()
        self._need_reset = True

    def get_data(self, outfile: str) -> MeasurementsCsv:
        if self.keep_raw:
            self._raw_file = outfile + '.raw'
        if self.collector:
            self.collector.process_frames(self._raw_file)
        active_sites: List[str] = [chan.label for chan in self.active_channels]
        if self.collector:
            self.collector.write_frames(outfile, columns=active_sites)
        return MeasurementsCsv(outfile, self.active_channels, self.sample_rate_hz)

    def get_raw(self) -> List[str]:
        return [self._raw_file] if self._raw_file else []

    def _init_channels(self) -> None:
        raise NotImplementedError()

    def teardown(self) -> None:
        if not self.keep_raw:
            if os.path.isfile(self._raw_file or ''):
                os.remove(self._raw_file or '')


class GfxInfoFramesInstrument(FramesInstrument):

    mode: int = CONTINUOUS
    collector_cls = GfxinfoFrameCollector

    def _init_channels(self) -> None:
        columns: List[str] = read_gfxinfo_columns(self.target)
        for entry in columns:
            if entry == 'Flags':
                self.add_channel('Flags', MeasurementType('flags', 'flags'))
            else:
                self.add_channel(entry, 'time_ns')
        self.header = [chan.label for chan in self.channels.values()]


class SurfaceFlingerFramesInstrument(FramesInstrument):

    mode: int = CONTINUOUS
    collector_cls = SurfaceFlingerFrameCollector

    def _init_channels(self) -> None:
        for field in SurfaceFlingerFrame._fields:
            # remove the "_time" from filed names to avoid duplication
            self.add_channel(field[:-5], 'time_us')
        self.header = [chan.label for chan in self.channels.values()]
