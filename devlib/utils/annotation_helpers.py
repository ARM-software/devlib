#    Copyright 2025 ARM Limited
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

"""
Helpers to annotate the code

"""
import sys
from typing import Union, Sequence, Optional
from typing_extensions import NotRequired, LiteralString, TYPE_CHECKING, TypedDict
if TYPE_CHECKING:
    from _typeshed import StrPath, BytesPath
    from devlib.platform import Platform
    from devlib.utils.android import AdbConnection
    from devlib.utils.ssh import SshConnection
    from devlib.host import LocalConnection
    from devlib.connection import PopenBackgroundCommand, AdbBackgroundCommand, ParamikoBackgroundCommand
else:
    StrPath = str
    BytesPath = bytes


import os
if sys.version_info >= (3, 9):
    SubprocessCommand = Union[
        str, bytes, os.PathLike[str], os.PathLike[bytes],
        Sequence[Union[str, bytes, os.PathLike[str], os.PathLike[bytes]]]]
else:
    SubprocessCommand = Union[str, bytes, os.PathLike,
                              Sequence[Union[str, bytes, os.PathLike]]]

BackgroundCommand = Union['AdbBackgroundCommand', 'ParamikoBackgroundCommand', 'PopenBackgroundCommand']

SupportedConnections = Union['LocalConnection', 'AdbConnection', 'SshConnection']


class SshUserConnectionSettings(TypedDict, total=False):
    username: NotRequired[str]
    password: NotRequired[str]
    keyfile: NotRequired[Optional[Union[LiteralString, StrPath, BytesPath]]]
    host: NotRequired[str]
    port: NotRequired[int]
    timeout: NotRequired[float]
    platform: NotRequired['Platform']
    sudo_cmd: NotRequired[str]
    strict_host_check: NotRequired[bool]
    use_scp: NotRequired[bool]
    poll_transfers: NotRequired[bool]
    start_transfer_poll_delay: NotRequired[int]
    total_transfer_timeout: NotRequired[int]
    transfer_poll_period: NotRequired[int]


class AdbUserConnectionSettings(SshUserConnectionSettings):
    device: NotRequired[str]
    adb_server: NotRequired[str]
    adb_port: NotRequired[int]


UserConnectionSettings = Union[SshUserConnectionSettings, AdbUserConnectionSettings]
