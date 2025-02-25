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

import logging

from devlib.utils.types import caseless_string
from devlib.utils.misc import get_logger
from typing import TYPE_CHECKING, Optional, List
if TYPE_CHECKING:
    from devlib.target import Target


class CollectorBase(object):
    """
    The Collector API provide a consistent way of collecting arbitrary data from
    a target. Data is collected via an instance of a class derived from :class:`CollectorBase`.

    :param target: The devlib Target from which data will be collected.
    """
    def __init__(self, target: 'Target'):
        self.target = target
        self.logger: logging.Logger = get_logger(self.__class__.__name__)
        self.output_path: Optional[str] = None

    def reset(self) -> None:
        """
        This can be used to configure a collector for collection. This must be invoked
        before :meth:`start()` is called to begin collection.
        """
        pass

    def start(self) -> None:
        """
        Starts collecting from the target.
        """
        pass

    def stop(self):
        """
        Stops collecting from target. Must be called after
        :func:`start()`.
        """
        pass

    def set_output(self, output_path: str) -> None:
        """
        Configure the output path for the particular collector. This will be either
        a directory or file path which will be used when storing the data. Please see
        the individual Collector documentation for more information.

        :param output_path: The path (file or directory) to which data will be saved.
        """
        self.output_path = output_path

    def get_data(self) -> 'CollectorOutput':
        """
        The collected data will be return via the previously specified output_path.
        This method will return a :class:`CollectorOutput` object which is a subclassed
        list object containing individual ``CollectorOutputEntry`` objects with details
        about the individual output entry.

        :raises RuntimeError: If ``output_path`` has not been set.
        """
        return CollectorOutput()

    def __enter__(self):
        self.reset()
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


class CollectorOutputEntry(object):
    """
    This object is designed to allow for the output of a collector to be processed
    generically. The object will behave as a regular string containing the path to
    underlying output path and can be used directly in ``os.path`` operations.

    :param path: The file path of the collected output data.
    :param path_kind: The type of output. Must be one of ``file`` or ``directory``.
    """
    path_kinds: List[str] = ['file', 'directory']

    def __init__(self, path: str, path_kind: str):
        self.path = path  # path for the corresponding output item

        path_kind = caseless_string(path_kind)
        if path_kind not in self.path_kinds:
            msg = '{} is not a valid path_kind [{}]'
            raise ValueError(msg.format(path_kind, ' '.join(self.path_kinds)))
        self.path_kind = path_kind  # file or directory

    def __str__(self):
        return self.path

    def __repr__(self):
        return '<{} ({})>'.format(self.path, self.path_kind)

    def __fspath__(self):
        """Allow using with os.path operations"""
        return self.path


class CollectorOutput(list):
    pass
