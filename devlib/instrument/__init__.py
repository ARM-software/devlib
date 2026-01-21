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
import collections
from abc import abstractmethod
from devlib.utils.csvutil import csvreader
from devlib.utils.types import numeric
from devlib.utils.types import identifier
from devlib.utils.misc import get_logger
from typing import (Dict, Optional, List, OrderedDict,
                    TYPE_CHECKING, Union, Callable,
                    Any, Tuple)
from collections.abc import Generator

if TYPE_CHECKING:
    from devlib.target import Target

# Channel modes describe what sort of measurement the instrument supports.
# Values must be powers of 2
INSTANTANEOUS = 1
CONTINUOUS = 2

MEASUREMENT_TYPES: Dict[str, 'MeasurementType'] = {}  # populated further down


class MeasurementType(object):
    """
    In order to make instruments easer to use, and to make it easier to swap them
    out when necessary (e.g. change method of collecting power), a number of
    standard measurement types are defined. This way, for example, power will
    always be reported as "power" in Watts, and never as "pwr" in milliWatts.
    Currently defined measurement types are


    +-------------+-------------+---------------+
    | Name        | Units       | Category      |
    +=============+=============+===============+
    | count       | count       |               |
    +-------------+-------------+---------------+
    | percent     | percent     |               |
    +-------------+-------------+---------------+
    | time        | seconds     |  time         |
    +-------------+-------------+---------------+
    | time_us     | microseconds|  time         |
    +-------------+-------------+---------------+
    | time_ms     | milliseconds|  time         |
    +-------------+-------------+---------------+
    | time_ns     | nanoseconds |  time         |
    +-------------+-------------+---------------+
    | temperature | degrees     |  thermal      |
    +-------------+-------------+---------------+
    | power       | watts       | power/energy  |
    +-------------+-------------+---------------+
    | voltage     | volts       | power/energy  |
    +-------------+-------------+---------------+
    | current     | amps        | power/energy  |
    +-------------+-------------+---------------+
    | energy      | joules      | power/energy  |
    +-------------+-------------+---------------+
    | tx          | bytes       | data transfer |
    +-------------+-------------+---------------+
    | rx          | bytes       | data transfer |
    +-------------+-------------+---------------+
    | tx/rx       | bytes       | data transfer |
    +-------------+-------------+---------------+
    | fps         | fps         |  ui render    |
    +-------------+-------------+---------------+
    | frames      | frames      |  ui render    |
    +-------------+-------------+---------------+

    """
    def __init__(self, name: str, units: Optional[str],
                 category: Optional[str] = None, conversions: Optional[Dict[str, Callable]] = None):
        self.name = name
        self.units = units
        self.category = category
        self.conversions: Dict[str, Callable] = {}
        if conversions is not None:
            for key, value in conversions.items():
                if not callable(value):
                    msg = 'Converter must be callable; got {} "{}"'
                    raise ValueError(msg.format(type(value), value))
                self.conversions[key] = value

    def convert(self, value: str, to: Union[str, 'MeasurementType']) -> Union[str, 'MeasurementType']:
        if isinstance(to, str) and to in MEASUREMENT_TYPES:
            to = MEASUREMENT_TYPES[to]
        if not isinstance(to, MeasurementType):
            msg: str = 'Unexpected conversion target: "{}"'
            raise ValueError(msg.format(to))
        if to.name == self.name:
            return value
        if to.name not in self.conversions:
            msg = 'No conversion from {} to {} available'
            raise ValueError(msg.format(self.name, to.name))
        return self.conversions[to.name](value)

    def __lt__(self, other):
        if isinstance(other, MeasurementType):
            return self.name < other.name
        return self.name < other

    def __le__(self, other):
        if isinstance(other, MeasurementType):
            return self.name <= other.name
        return self.name <= other

    def __eq__(self, other):
        if isinstance(other, MeasurementType):
            return self.name == other.name
        return self.name == other

    def __ne__(self, other):
        if isinstance(other, MeasurementType):
            return self.name != other.name
        return self.name != other

    def __gt__(self, other):
        if isinstance(other, MeasurementType):
            return self.name > other.name
        return self.name > other

    def __ge__(self, other):
        if isinstance(other, MeasurementType):
            return self.name >= other.name
        return self.name >= other

    def __str__(self):
        return self.name

    def __repr__(self):
        if self.category:
            text = 'MeasurementType({}, {}, {})'
            return text.format(self.name, self.units, self.category)
        else:
            text = 'MeasurementType({}, {})'
            return text.format(self.name, self.units)


# Standard measures. In order to make sure that downstream data processing is not tied
# to particular insturments (e.g. a particular method of mearuing power), instruments
# must, where possible, resport their measurments formatted as on of the standard types
# defined here.
_measurement_types: List[MeasurementType] = [
    # For whatever reason, the type of measurement could not be established.
    MeasurementType('unknown', None),

    # Generic measurements
    MeasurementType('count', 'count'),
    MeasurementType('percent', 'percent'),

    # Time measurement. While there is typically a single "canonical" unit
    # used for each type of measurmenent, time may be measured to a wide variety
    # of events occuring at a wide range of scales. Forcing everying into a
    # single scale will lead to inefficient and awkward to work with result tables.
    # Coversion functions between the formats are specified, so that downstream
    # processors that expect all times time be at a particular scale can automatically
    # covert without being familar with individual instruments.
    MeasurementType('time', 'seconds', 'time',
                    conversions={
                            'time_us': lambda x: x * 1e6,
                            'time_ms': lambda x: x * 1e3,
                            'time_ns': lambda x: x * 1e9,
                    }
                    ),
    MeasurementType('time_us', 'microseconds', 'time',
                    conversions={
                        'time': lambda x: x / 1e6,
                        'time_ms': lambda x: x / 1e3,
                        'time_ns': lambda x: x * 1e3,
                    }
                    ),
    MeasurementType('time_ms', 'milliseconds', 'time',
                    conversions={
                        'time': lambda x: x / 1e3,
                        'time_us': lambda x: x * 1e3,
                        'time_ns': lambda x: x * 1e6,
                    }
                    ),
    MeasurementType('time_ns', 'nanoseconds', 'time',
                    conversions={
                        'time': lambda x: x / 1e9,
                        'time_ms': lambda x: x / 1e6,
                        'time_us': lambda x: x / 1e3,
                    }
                    ),

    # Measurements related to thermals.
    MeasurementType('temperature', 'degrees', 'thermal'),

    # Measurements related to power end energy consumption.
    MeasurementType('power', 'watts', 'power/energy'),
    MeasurementType('voltage', 'volts', 'power/energy'),
    MeasurementType('current', 'amps', 'power/energy'),
    MeasurementType('energy', 'joules', 'power/energy'),

    # Measurments realted to data transfer, e.g. neworking,
    # memory, or backing storage.
    MeasurementType('tx', 'bytes', 'data transfer'),
    MeasurementType('rx', 'bytes', 'data transfer'),
    MeasurementType('tx/rx', 'bytes', 'data transfer'),

    MeasurementType('fps', 'fps', 'ui render'),
    MeasurementType('frames', 'frames', 'ui render'),
]
for m in _measurement_types:
    MEASUREMENT_TYPES[m.name] = m


class Measurement(object):

    __slots__ = ['value', 'channel']

    @property
    def name(self) -> str:
        """
        name of the measurement
        """
        return '{}_{}'.format(self.channel.site, self.channel.kind)

    @property
    def units(self) -> Optional[str]:
        """
        Units in which measurement will be reported.
        """
        return self.channel.units

    def __init__(self, value: Union[int, float], channel: 'InstrumentChannel'):
        self.value = value
        self.channel = channel

    def __lt__(self, other):
        if hasattr(other, 'value'):
            return self.value < other.value
        return self.value < other

    def __eq__(self, other):
        if hasattr(other, 'value'):
            return self.value == other.value
        return self.value == other

    def __le__(self, other):
        if hasattr(other, 'value'):
            return self.value <= other.value
        return self.value <= other

    def __ne__(self, other):
        if hasattr(other, 'value'):
            return self.value != other.value
        return self.value != other

    def __gt__(self, other):
        if hasattr(other, 'value'):
            return self.value > other.value
        return self.value > other

    def __ge__(self, other):
        if hasattr(other, 'value'):
            return self.value >= other.value
        return self.value >= other

    def __str__(self):
        if self.units:
            return '{}: {} {}'.format(self.name, self.value, self.units)
        else:
            return '{}: {}'.format(self.name, self.value)

    __repr__ = __str__


class MeasurementsCsv(object):

    def __init__(self, path, channels: Optional[List['InstrumentChannel']] = None,
                 sample_rate_hz: Optional[float] = None):
        self.path = path
        self.channels = channels
        self.sample_rate_hz = sample_rate_hz
        if self.channels is None:
            self._load_channels()
        headings = [chan.label for chan in self.channels] if self.channels else []

        self.data_tuple = collections.namedtuple('csv_entry',    # type:ignore
                                                 map(identifier, headings))

    def measurements(self) -> List[List['Measurement']]:
        return list(self.iter_measurements())

    def iter_measurements(self) -> Generator[List['Measurement'], None, None]:
        for row in self._iter_rows():
            values = map(numeric, row)
            if self.channels:
                yield [Measurement(v, c) for (v, c) in zip(values, self.channels)]

    def values(self) -> List:
        return list(self.iter_values())

    def iter_values(self) -> Generator[Tuple[Any], None, None]:
        for row in self._iter_rows():
            values = list(map(numeric, row))
            yield self.data_tuple(*values)

    def _load_channels(self) -> None:
        header: List[str] = []
        with csvreader(self.path) as reader:
            header = next(reader)

        self.channels = []
        for entry in header:
            for mt in MEASUREMENT_TYPES:
                suffix: str = '_{}'.format(mt)
                if entry.endswith(suffix):
                    site: Optional[str] = entry[:-len(suffix)]
                    measure: str = mt
                    break
            else:
                if entry in MEASUREMENT_TYPES:
                    site = None
                    measure = entry
                else:
                    site = entry
                    measure = 'unknown'
            if site:
                chan = InstrumentChannel(site, measure)
                self.channels.append(chan)

    # pylint: disable=stop-iteration-return
    def _iter_rows(self) -> Generator[List[str], None, None]:
        with csvreader(self.path) as reader:
            next(reader)  # headings
            for row in reader:
                yield row


class InstrumentChannel(object):
    """
    An :class:`InstrumentChannel` describes a single type of measurement that may
    be collected by an :class:`~devlib.instrument.Instrument`. A channel is
    primarily defined by a ``site`` and a ``measurement_type``.

    A ``site`` indicates where  on the target a measurement is collected from
    (e.g. a voltage rail or location of a sensor).

    A ``measurement_type`` is an instance of :class:`MeasurmentType` that
    describes what sort of measurement this is (power, temperature, etc). Each
    measurement type has a standard unit it is reported in, regardless of an
    instrument used to collect it.

    A channel (i.e. site/measurement_type combination) is unique per instrument,
    however there may be more than one channel associated with one site (e.g. for
    both voltage and power).

    It should not be assumed that any site/measurement_type combination is valid.
    The list of available channels can queried with
    :func:`Instrument.list_channels()`.

    .. attribute:: InstrumentChannel.site

    The name of the "site" from which the measurements are collected (e.g. voltage
    rail, sensor, etc).

    """
    @property
    def label(self) -> str:
        """
        Returns a label uniquely identifying the channel.

        This label is used to tag measurements and is constructed by
        combining the channel's site and kind using the format:
        '<site>_<kind>'.

        If the site is not defined (i.e., None), only the kind is returned.

        Returns:
            str: A string label for the channel.

        Example:
            If site = "cluster0" and kind = "power", the label will be "cluster0_power".
            If site = None and kind = "temperature", the label will be "temperature".
        """
        if self.site is not None:
            return '{}_{}'.format(self.site, self.kind)
        return self.kind

    name = label

    @property
    def kind(self) -> str:
        """
        A string indicating the type of measurement that will be collected. This is
        the ``name`` of the :class:`MeasurmentType` associated with this channel.
        """
        return self.measurement_type.name

    @property
    def units(self) -> Optional[str]:
        """
        Units in which measurement will be reported. this is determined by the
        underlying :class:`MeasurmentType`.
        """
        return self.measurement_type.units

    def __init__(self, site: str, measurement_type: Union[str, MeasurementType], **attrs):
        self.site = site
        if isinstance(measurement_type, MeasurementType):
            self.measurement_type = measurement_type
        else:
            try:
                self.measurement_type = MEASUREMENT_TYPES[measurement_type]
            except KeyError:
                raise ValueError('Unknown measurement type:  {}'.format(measurement_type))
        for atname, atvalue in attrs.items():
            setattr(self, atname, atvalue)

    def __str__(self):
        if self.name == self.label:
            return 'CHAN({})'.format(self.label)
        else:
            return 'CHAN({}, {})'.format(self.name, self.label)

    __repr__ = __str__


class Instrument(object):
    """
    The ``Instrument`` API provide a consistent way of collecting measurements from
    a target. Measurements are collected via an instance of a class derived from
    :class:`~devlib.instrument.Instrument`. An ``Instrument`` allows collection of
    measurement from one or more channels. An ``Instrument`` may support
    ``INSTANTANEOUS`` or ``CONTINUOUS`` collection, or both.

    .. attribute:: Instrument.mode

    A bit mask that indicates collection modes that are supported by this
    instrument. Possible values are:

    :INSTANTANEOUS: The instrument supports taking a single sample via
                    ``take_measurement()``.
    :CONTINUOUS: The instrument supports collecting measurements over a
                    period of time via ``start()``, ``stop()``, ``get_data()``,
            and (optionally) ``get_raw`` methods.

    .. note:: It's possible for one instrument to support more than a single
                mode.

    .. attribute:: Instrument.active_channels

    Channels that have been activated via ``reset()``. Measurements will only be
    collected for these channels.
    .. attribute:: Instrument.sample_rate_hz

   Sample rate of the instrument in Hz. Assumed to be the same for all channels.

   .. note:: This attribute is only provided by
             :class:`~devlib.instrument.Instrument` s that
             support ``CONTINUOUS`` measurement.
    """
    mode: int = 0

    def __init__(self, target: 'Target'):
        self.target = target
        self.logger = get_logger(self.__class__.__name__)
        self.channels: OrderedDict[str, InstrumentChannel] = collections.OrderedDict()
        self.active_channels: List[InstrumentChannel] = []
        self.sample_rate_hz: Optional[float] = None

    # channel management

    def list_channels(self) -> List[InstrumentChannel]:
        """
        Returns a list of :class:`InstrumentChannel` instances that describe what
        this instrument can measure on the current target. A channel is a combination
        of a ``kind`` of measurement (power, temperature, etc) and a ``site`` that
        indicates where on the target the measurement will be collected from.
        """
        return list(self.channels.values())

    def get_channels(self, measure: Union[str, MeasurementType]):
        """
        Returns channels for a particular ``measure`` type. A ``measure`` can be
        either a string (e.g. ``"power"``) or a :class:`MeasurmentType` instance.
        """
        if isinstance(measure, MeasurementType):
            if hasattr(measure, 'name'):
                measure = measure.name
        return [c for c in self.list_channels() if c.kind == measure]

    def add_channel(self, site: str, measure: Union[str, MeasurementType], **attrs) -> None:
        """
        add channel to channels dict
        """
        chan = InstrumentChannel(site, measure, **attrs)
        self.channels[chan.label] = chan

    # initialization and teardown

    def setup(self, *args, **kwargs) -> None:
        """
        This will set up the instrument on the target. Parameters this method takes
        are particular to subclasses (see documentation for specific instruments
        below).  What actions are performed by this method are also
        instrument-specific.  Usually these will be things like  installing
        executables, starting services, deploying assets, etc. Typically, this method
        needs to be invoked at most once per reboot of the target (unless
        ``teardown()`` has been called), but see documentation for the instrument
        you're interested in.
        """
        pass

    def teardown(self) -> None:
        """
        Performs any required clean up of the instrument. This usually includes
        removing temporary and raw files (if ``keep_raw`` is set to ``False`` on relevant
        instruments), stopping services etc.
        """
        pass

    def reset(self, sites: Optional[List[str]] = None,
              kinds: Optional[List[str]] = None,
              channels: Optional[OrderedDict[str, InstrumentChannel]] = None) -> None:
        """
        This is used to configure an instrument for collection. This must be invoked
        before ``start()`` is called to begin collection. This methods sets the
        ``active_channels`` attribute of the ``Instrument``.

        If ``channels`` is provided, it is a list of names of channels to enable and
        ``sites`` and ``kinds`` must both be ``None``.

        Otherwise, if one of ``sites`` or ``kinds`` is provided, all channels
        matching the given sites or kinds are enabled. If both are provided then all
        channels of the given kinds at the given sites are enabled.

        If none of ``sites``, ``kinds`` or ``channels`` are provided then all
        available channels are enabled.
        """
        if channels is not None:
            if sites is not None or kinds is not None:
                raise ValueError('sites and kinds should not be set if channels is set')

            try:
                self.active_channels = [self.channels[ch] for ch in channels]
            except KeyError as e:
                msg: str = 'Unexpected channel "{}"; must be in {}'
                raise ValueError(msg.format(e, self.channels.keys()))
        elif sites is None and kinds is None:
            self.active_channels = sorted(self.channels.values(), key=lambda x: x.label)
        else:
            if isinstance(sites, str):
                sites = [sites]
            if isinstance(kinds, str):
                kinds = [kinds]

            wanted = lambda ch: ((kinds is None or ch.kind in kinds) and
                                 (sites is None or ch.site in sites))
            self.active_channels = list(filter(wanted, self.channels.values()))

    # instantaneous
    @abstractmethod
    def take_measurement(self) -> List[Measurement]:
        """
        Take a single measurement from ``active_channels``. Returns a list of
        :class:`Measurement` objects (one for each active channel).

        .. note:: This method is only implemented by
                    :class:`~devlib.instrument.Instrument's that
                    support ``INSTANTANEOUS`` measurement.
        """
        pass

    # continuous

    def start(self) -> None:
        """
        Starts collecting measurements from ``active_channels``.

        .. note:: This method is only implemented by
                    :class:`~devlib.instrument.Instrument` s that
                    support ``CONTINUOUS`` measurement.
        """
        pass

    def stop(self) -> None:
        """
        Stops collecting measurements from ``active_channels``. Must be called after
        :func:`start()`.

        .. note:: This method is only implemented by
                    :class:`~devlib.instrument.Instrument` s that
                    support ``CONTINUOUS`` measurement.
        """
        pass

    @abstractmethod
    def get_data(self, outfile: str) -> MeasurementsCsv:
        """
        Write collected data into ``outfile``. Must be called after :func:`stop()`.
        Data will be written in CSV format with a column for each channel and a row
        for each sample. Column heading will be channel, labels in the form
        ``<site>_<kind>`` (see :class:`InstrumentChannel`). The order of the columns
        will be the same as the order of channels in ``Instrument.active_channels``.

        If reporting timestamps, one channel must have a ``site`` named
        ``"timestamp"`` and a ``kind`` of a :class:`MeasurmentType` of an appropriate
        time unit which will be used, if appropriate, during any post processing.

        .. note:: Currently supported time units are seconds, milliseconds and
                    microseconds, other units can also be used if an appropriate
                    conversion is provided.

        This returns a :class:`MeasurementCsv` instance associated with the outfile
        that can be used to stream :class:`Measurement` s lists (similar to what is
        returned by ``take_measurement()``.

        .. note:: This method is only implemented by
                    :class:`~devlib.instrument.Instrument` s that
                    support ``CONTINUOUS`` measurement.
        """
        pass

    def get_raw(self) -> List[str]:
        """
        Returns a list of paths to files containing raw output from the underlying
        source(s) that is used to produce the data CSV. If no raw output is
        generated or saved, an empty list will be returned. The format of the
        contents of the raw files is entirely source-dependent.

        .. note:: This method is not guaranteed to return valid filepaths after the
                    :meth:`teardown` method has been invoked as the raw files may have
                    been deleted. Please ensure that copies are created manually
                    prior to calling :meth:`teardown` if the files are to be retained.
        """
        return []
