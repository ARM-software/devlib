#    Copyright 2024-2025 ARM Limited
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
Target module for devlib.

This module defines the Target class and supporting functionality.
"""
import atexit
import asyncio
import io
import base64
import functools
import gzip
import glob
import os
from operator import itemgetter
import re
import sys
import time
import logging
import posixpath
import subprocess
import tarfile
import tempfile
import threading
import uuid
import xml.dom.minidom
import copy
import inspect
import itertools
from collections import namedtuple, defaultdict
from numbers import Number
from shlex import quote
from weakref import WeakMethod
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping  # type: ignore

from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from devlib.host import LocalConnection, PACKAGE_BIN_DIRECTORY
from devlib.module import get_module, Module, HardRestModule, BootModule
from devlib.platform import Platform
from devlib.exception import (DevlibTransientError, TargetStableError,
                              TargetNotRespondingError, TimeoutError,
                              TargetTransientError, KernelConfigKeyError,
                              TargetError, HostError, TargetCalledProcessError,
                              DevlibError)
from devlib.utils.ssh import SshConnection
from devlib.utils.android import (AdbConnection, AndroidProperties,
                                  LogcatMonitor, adb_command, INTENT_FLAGS)
from devlib.utils.misc import (memoized, isiterable, convert_new_lines,
                               groupby_value, commonprefix, ABI_MAP, get_cpu_name,
                               ranges_to_list, batch_contextmanager, tls_property,
                               _BoundTLSProperty, nullcontext, safe_extract, get_logger)
from devlib.utils.types import (integer, boolean, bitmask, identifier,
                                caseless_string, bytes_regex)
import devlib.utils.asyn as asyn
from devlib.utils.annotation_helpers import (SshUserConnectionSettings, UserConnectionSettings,
                                             AdbUserConnectionSettings, SupportedConnections,
                                             SubprocessCommand, BackgroundCommand)
from typing import (List, Set, Dict, Union, Optional, Callable, TypeVar,
                    Any, cast, TYPE_CHECKING, Type, Pattern,
                    Tuple, Iterator, AsyncContextManager, Iterable,
                    Mapping as Maptype, ClassVar)
from collections.abc import AsyncGenerator
from types import ModuleType
from typing_extensions import Literal
import signal
if TYPE_CHECKING:
    from devlib.connection import ConnectionBase
    from devlib.utils.misc import InitCheckpointMeta
    from devlib.utils.asyn import AsyncManager, _AsyncPolymorphicFunction
    from asyncio import AbstractEventLoop
    from contextlib import _GeneratorContextManager
    from xml.dom.minidom import Document


FSTAB_ENTRY_REGEX = re.compile(r'(\S+) on (.+) type (\S+) \((\S+)\)')
ANDROID_SCREEN_STATE_REGEX = re.compile('(?:mPowerState|mScreenOn|mWakefulness|Display Power: state)=([0-9]+|true|false|ON|OFF|DOZE|Dozing|Asleep|Awake)',
                                        re.IGNORECASE)
ANDROID_SCREEN_RESOLUTION_REGEX = re.compile(r'cur=(?P<width>\d+)x(?P<height>\d+)')
ANDROID_SCREEN_ROTATION_REGEX = re.compile(r'orientation=(?P<rotation>[0-3])')
DEFAULT_SHELL_PROMPT = re.compile(r'^.*(shell|root|juno)@?.*:[/~]\S* *[#$] ',
                                  re.MULTILINE)
KVERSION_REGEX = re.compile(
    r'(?P<version>\d+)(\.(?P<major>\d+)(\.(?P<minor>\d+))?(-rc(?P<rc>\d+))?)?(-android(?P<android_version>[0-9]+))?(-(?P<commits>\d+)-g(?P<sha1>[0-9a-fA-F]{7,}))?(-ab(?P<gki_abi>[0-9]+))?'
)

GOOGLE_DNS_SERVER_ADDRESS = '8.8.8.8'


installed_package_info = namedtuple('installed_package_info', 'apk_path package')

T = TypeVar('T', bound=Callable[..., Any])


# FIXME - need to annotate to indicate the self argument needs to have a conn object of ConnectionBase type.
def call_conn(f: T) -> T:
    """
    Decorator to be used on all :class:`devlib.target.Target` methods that
    directly use a method of ``self.conn``.

    This ensures that if a call to any of the decorated method occurs while
    executing, a new connection will be created in order to avoid possible
    deadlocks. This can happen if e.g. a target's method is called from
    ``__del__``, which could be executed by the garbage collector, interrupting
    another call to a method of the connection instance.

    :param f: Method to decorate.

    :returns: The wrapped method that automatically creates and releases
              a new connection if reentered.

    .. note:: This decorator could be applied directly to all methods with a
        metaclass or ``__init_subclass__`` but it could create issues when
        passing target methods as callbacks to connections' methods.
    """

    @functools.wraps(f)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        conn = self.conn
        reentered = conn.is_in_use
        disconnect = False
        try:
            # If the connection was already in use we need to use a different
            # instance to avoid reentrancy deadlocks. This can happen even in
            # single threaded code via __del__ implementations that can be
            # called at any point.
            if reentered:
                # Shallow copy so we can use another connection instance
                _self = copy.copy(self)
                new_conn = _self.get_connection()
                assert conn is not new_conn
                _self.conn = new_conn
                disconnect = True
            else:
                _self = self
            return f(_self, *args, **kwargs)
        finally:
            if disconnect:
                _self.disconnect()
            elif not reentered:
                # Return the connection to the pool, so if we end up exiting
                # the thread the connection can then be reused by another
                # thread.
                del self.conn
                with self._lock:
                    self._unused_conns.add(conn)

    return cast(T, wrapper)


class Target(object):
    """
    An abstract base class defining the interface for a devlib target device.

    :param connection_settings: Connection parameters for the target
        (e.g., SSH, ADB) in a dictionary.
    :param platform: A platform object describing architecture, ABI, kernel,
        etc. If ``None``, platform info may be inferred or left unspecified.
    :param working_directory: A writable directory on the target for devlib's
        temporary files or scripts. If ``None``, a default path is used.
    :param executables_directory: A directory on the target for storing
        executables installed by devlib. If ``None``, a default path may be used.
    :param connect: If ``True``, attempt to connect to the device immediately,
        else call :meth:`connect` manually.
    :param modules: Dict mapping module names to their parameters. Additional
        devlib modules to load on initialization.
    :param load_default_modules: If ``True``, load the modules specified in
        :attr:`default_modules`.
    :param shell_prompt: Compiled regex matching the target’s shell prompt.
    :param conn_cls: A reference to the Connection class to be used.
    :param is_container: If ``True``, indicates the target is a container
        rather than a physical or virtual machine.
    :param max_async: Number of asynchronous operations supported. Affects the
        creation of parallel connections.

    :raises: Various :class:`devlib.exception` types if connection fails.

    .. note::
       Subclasses must implement device-specific methods (e.g., for Android vs. Linux or
       specialized boards). The default implementation here may be incomplete.
    """
    path: ClassVar[ModuleType]
    os: Optional[str] = None
    system_id: Optional[str] = None

    default_modules: List[Type[Module]] = []

    def __init__(self,
                 connection_settings: Optional[UserConnectionSettings] = None,
                 platform: Optional[Platform] = None,
                 working_directory: Optional[str] = None,
                 executables_directory: Optional[str] = None,
                 connect: bool = True,
                 modules: Optional[Dict[str, Dict[str, Type[Module]]]] = None,
                 load_default_modules: bool = True,
                 shell_prompt: Pattern[str] = DEFAULT_SHELL_PROMPT,
                 conn_cls: Optional['InitCheckpointMeta'] = None,
                 is_container: bool = False,
                 max_async: int = 50,
                 tmp_directory: Optional[str] = None,
                 ):
        """
        Initialize a new Target instance and optionally connect to it.
        """
        self._lock = threading.RLock()
        self._async_pool: Optional[ThreadPoolExecutor] = None
        self._async_pool_size: Optional[int] = None
        self._unused_conns: Set[ConnectionBase] = set()

        self._is_rooted: Optional[bool] = None
        self.connection_settings: UserConnectionSettings = connection_settings or {}
        # Set self.platform: either it's given directly (by platform argument)
        # or it's given in the connection_settings argument
        # If neither, create default Platform()
        if platform is None:
            self.platform = self.connection_settings.get('platform', Platform())
        else:
            self.platform = platform
        # Check if the user hasn't given two different platforms
        if connection_settings and ('platform' in self.connection_settings) and ('platform' in connection_settings):
            if connection_settings['platform'] is not platform:
                raise TargetStableError('Platform specified in connection_settings '
                                        '({}) differs from that directly passed '
                                        '({})!)'
                                        .format(connection_settings['platform'],
                                                self.platform))
        self.connection_settings['platform'] = self.platform
        self.working_directory = working_directory
        self.executables_directory = executables_directory
        self.tmp_directory = tmp_directory
        self.load_default_modules = load_default_modules
        self.shell_prompt: Pattern[bytes] = bytes_regex(shell_prompt)
        self.conn_cls = conn_cls
        self.is_container = is_container
        self.logger = get_logger(self.__class__.__name__)
        self._installed_binaries: Dict[str, str] = {}
        self._installed_modules: Dict[str, Module] = {}
        self._cache: Dict = {}
        self._shutils: Optional[str] = None
        self._max_async = max_async
        self.busybox: Optional[str] = None

        def normalize_mod_spec(spec) -> Tuple[str, Dict[str, Type[Module]]]:
            if isinstance(spec, str):
                return (spec, {})
            else:
                [(name, params)] = spec.items()
                return (name, params)

        normalized_modules: List[Tuple[str, Dict[str, Type[Module]]]] = sorted(
            map(
                normalize_mod_spec,
                itertools.chain(
                    self.default_modules if load_default_modules else [],
                    modules or [],
                    self.platform.modules or [],
                )
            ),
            key=itemgetter(0),
        )

        # Ensure that we did not ask for the same module but different
        # configurations. Empty configurations are ignored, so any
        # user-provided conf will win against an empty conf.
        def elect(name: str, specs: List[Tuple[str, Dict[str, Type[Module]]]]) -> Tuple[str, Dict[str, Type[Module]]]:
            specs = list(specs)

            confs = set(
                tuple(sorted(params.items()))
                for _, params in specs
                if params
            )
            if len(confs) > 1:
                raise ValueError(f'Attempted to load the module "{name}" with multiple different configuration')
            else:
                if any(
                    params is None
                    for _, params in specs
                ):
                    params = {}
                else:
                    params = dict(confs.pop()) if confs else {}

                return (name, params)

        modules = dict(itertools.starmap(
            elect,
            itertools.groupby(normalized_modules, key=itemgetter(0))
        ))

        def get_kind(name: str) -> str:
            return get_module(name).kind or ''

        def kind_conflict(kind: str, names: List[str]):
            if kind:
                raise ValueError(f'Cannot enable multiple modules sharing the same kind "{kind}": {sorted(names)}')

        list(itertools.starmap(
            kind_conflict,
            itertools.groupby(
                sorted(
                    modules.keys(),
                    key=get_kind
                ),
                key=get_kind
            )
        ))
        self._modules = modules

        atexit.register(
            WeakMethod(self.disconnect, atexit.unregister)
        )

        self._update_modules('early')
        if connect:
            self.connect(max_async=max_async)

    @property
    def core_names(self) -> Union[List[caseless_string], List[str]]:
        """
        A list of CPU core names in the order they appear
        registered with the OS. If they are not specified,
        they will be queried at run time.

        :return: CPU core names in order (e.g. ["A53", "A53", "A72", "A72"]).
        """
        if self.platform:
            return self.platform.core_names
        raise ValueError("No Platform set for this target, cannot access core_names")

    @property
    def core_clusters(self) -> List[int]:
        """
        A list with cluster ids of each core (starting with
        0). If this is not specified, clusters will be
        inferred from core names (cores with the same name are
        assumed to be in a cluster).

        :return: A list of integer cluster IDs for each core.
        """
        if self.platform:
            return self.platform.core_clusters
        raise ValueError("No Platform set for this target cannot access core_clusters")

    @property
    def big_core(self) -> Optional[str]:
        """
        The name of the big core in a big.LITTLE system. If this is
        not specified it will be inferred (on systems with exactly
        two clusters).

        :return: Big core name, or None if not defined.
        """
        if self.platform:
            return self.platform.big_core
        raise ValueError("No Platform set for this target cannot access big_core")

    @property
    def little_core(self) -> Optional[str]:
        """
        The name of the little core in a big.LITTLE system. If this is
        not specified it will be inferred (on systems with exactly
        two clusters).

        :return: Little core name, or None if not defined.
        """
        if self.platform:
            return self.platform.little_core
        raise ValueError("No Platform set for this target cannot access little_core")

    @property
    def is_connected(self) -> bool:
        """
        Indicates whether there is an active connection to the target.

        :return: True if connected, else False.
        """
        return self.conn is not None

    @property
    def connected_as_root(self) -> Optional[bool]:
        """
        Indicates whether the connection user on the target is root (uid=0).

        :return: True if root, False otherwise, or None if unknown.
        """
        if self.conn:
            if self.conn.connected_as_root:
                return True
        return False

    @property
    def is_rooted(self) -> Optional[bool]:
        """
        Indicates whether superuser privileges (root or sudo) are available.

        :return: True if superuser privileges are accessible, False if not,
            or None if undetermined.
        """
        if self._is_rooted is None:
            try:
                self.execute('ls /', timeout=5, as_root=True)
                self._is_rooted = True
            except (TargetError, TimeoutError):
                self._is_rooted = False

        return self._is_rooted or self.connected_as_root

    @property
    @memoized
    def needs_su(self) -> Optional[bool]:
        """
        Whether the current user must escalate privileges to run root commands.

        :return: True if the device is rooted but not connected as root.
        """
        return not self.connected_as_root and self.is_rooted

    @property
    @memoized
    def kernel_version(self) -> 'KernelVersion':
        """
        The kernel version from ``uname -r -v``, wrapped in a KernelVersion object.

        :raises ValueError: If busybox is unavailable for executing the uname command.
        :return: Kernel version details.
        """
        if self.busybox:
            return KernelVersion(self.execute('{} uname -r -v'.format(quote(self.busybox))).strip())
        raise ValueError("busybox not set. Cannot get kernel version")

    @property
    def hostid(self) -> int:
        """
        A numeric ID representing the system's host identity.

        :return: The hostid as an integer (parsed from hex).
        """
        return int(self.execute('{} hostid'.format(self.busybox)).strip(), 16)

    @property
    def hostname(self) -> str:
        """
        System hostname from ``hostname`` or ``uname -n``.

        :return: Hostname of the target.
        """
        return self.execute('{} hostname'.format(self.busybox)).strip()

    @property
    def os_version(self) -> Dict[str, str]:  # pylint: disable=no-self-use
        """
        A mapping of OS version info. Empty by default; child classes may override.

        :return: OS version details.
        """
        return {}

    @property
    def model(self) -> Optional[str]:
        """
        Hardware model name, if any.

        :return: Model name, or None if not defined.
        """
        return self.platform.model

    @property
    def abi(self) -> str:  # pylint: disable=no-self-use
        """
        The primary application binary interface (ABI) of this target.

        :return: ABI name (e.g. "armeabi-v7a"), or '' if unknown.
        """
        raise NotImplementedError("abi must be implemented by subclass")

    @property
    def supported_abi(self) -> List[str]:
        """
        A list of all supported ABIs.

        :return: List of ABI strings.
        """
        return [self.abi]

    @property
    @memoized
    def cpuinfo(self) -> 'Cpuinfo':
        """
        Parsed data from ``/proc/cpuinfo``.

        :return: A :class:`Cpuinfo` instance with CPU details.
        """
        return Cpuinfo(self.execute('cat /proc/cpuinfo'))

    @property
    @memoized
    def number_of_cpus(self) -> int:
        """
        Count of CPU cores, determined by listing ``/sys/devices/system/cpu/cpu*``.

        :return: Number of CPU cores.
        """
        num_cpus: int = 0
        corere = re.compile(r'^\s*cpu\d+\s*$')
        output: str = self.execute('ls /sys/devices/system/cpu', as_root=self.is_rooted)
        for entry in output.split():
            if corere.match(entry):
                num_cpus += 1
        return num_cpus

    @property
    @memoized
    def number_of_nodes(self) -> int:
        """
        Number of NUMA nodes detected by enumerating ``/sys/devices/system/node``.

        :return: NUMA node count, or 1 if unavailable.
        """
        if self.busybox:
            cmd = 'cd /sys/devices/system/node && {busybox} find . -maxdepth 1'.format(busybox=quote(self.busybox))
        else:
            raise ValueError('busybox not set. cannot form cmd')
        try:
            output: str = self.execute(cmd, as_root=self.is_rooted)
        except TargetStableError:
            return 1
        else:
            nodere = re.compile(r'^\./node\d+\s*$')
            num_nodes: int = 0
            for entry in output.splitlines():
                if nodere.match(entry):
                    num_nodes += 1
            return num_nodes

    @property
    @memoized
    def list_nodes_cpus(self) -> List[int]:
        """
        Aggregated list of CPU IDs across all NUMA nodes.

        :return: A list of CPU IDs from each detected node.
        """
        nodes_cpus: List[int] = []
        for node in range(self.number_of_nodes):
            path: str = self.path.join('/sys/devices/system/node/node{}/cpulist'.format(node))
            output: str = self.read_value(path)
            if output:
                nodes_cpus.extend(ranges_to_list(output))
        return nodes_cpus

    @property
    @memoized
    def config(self) -> 'KernelConfig':
        """
        Parsed kernel config from ``/proc/config.gz`` or ``/boot/config-*``.

        :return: A :class:`KernelConfig` instance.
        """
        try:
            return KernelConfig(self.execute('zcat /proc/config.gz'))
        except TargetStableError:
            for path in ['/boot/config-$({} uname -r)'.format(self.busybox), '/boot/config']:
                try:
                    return KernelConfig(self.execute('cat {}'.format(path)))
                except TargetStableError:
                    pass
        return KernelConfig('')

    @property
    @memoized
    def user(self) -> str:
        """
        The username for the active shell on the target.

        :return: Username (e.g., "root" or "shell").
        """
        return self.getenv('USER')

    @property
    @memoized
    def page_size_kb(self) -> int:
        """
        Page size in kilobytes, derived from ``/proc/self/smaps``.

        :return: Page size in KiB, or 0 if unknown.
        """
        cmd = "cat /proc/self/smaps | {0} grep KernelPageSize | {0} head -n 1 | {0} awk '{{ print $2 }}'"
        return int(self.execute(cmd.format(self.busybox)) or 0)

    @property
    def shutils(self) -> Optional[str]:
        """
        Path to shell utilities (if installed by devlib). Internal usage.

        :return: The path or None if uninitialized.
        """
        if self._shutils is None:
            self._setup_scripts()
        return self._shutils

    def is_running(self, comm: str) -> bool:
        """
        Check if a process with the specified name/command is running on the target.

        :param comm: The process name to search for.
        :return: True if a matching process is found, else False.
        """
        cmd_ps = f'''{self.busybox} ps -A -T -o stat,comm'''
        cmd_awk = f'''{self.busybox} awk 'BEGIN{{found=0}} {{state=$1; $1=""; if ($state != "Z" && $0 == " {comm}") {{found=1}}}} END {{print found}}' '''
        result: str = self.execute(f"{cmd_ps} | {cmd_awk}")
        return bool(int(result))

    @tls_property
    def _conn(self) -> 'ConnectionBase':
        """
        The underlying connection object. This will be ``None`` if an active
        connection does not exist (e.g. if ``connect=False`` as passed on
        initialization and :meth:`connect()` has not been called).

        :returns: The thread-local :class:`ConnectionBase` instance.

        .. note:: a :class:`~devlib.target.Target` will automatically create a
             connection per thread. This will always be set to the connection
             for the current thread.
        """
        try:
            with self._lock:
                return self._unused_conns.pop()
        except KeyError:
            return self.get_connection()

    # Add a basic property that does not require calling to get the value
    conn: SupportedConnections = cast(SupportedConnections, _conn.basic_property)

    @tls_property
    def _async_manager(self) -> 'AsyncManager':
        """
        Thread-local property that holds an async manager for concurrency tasks.

        :return: Async manager instance for the current thread.
        """
        return asyn.AsyncManager()

    # Add a basic property that does not require calling to get the value
    async_manager: 'AsyncManager' = cast('AsyncManager', _async_manager.basic_property)

    def __getstate__(self) -> Dict[str, Any]:
        """
        For pickling: exclude thread-local objects from the state.

        :return: A dictionary representing the object's state.
        """
        # tls_property will recreate the underlying value automatically upon
        # access and is typically used for dynamic content that cannot be
        # pickled or should not transmitted to another thread.
        ignored: set[str] = {
            k
            for k, v in inspect.getmembers(self.__class__)
            if isinstance(v, _BoundTLSProperty)
        }
        ignored.update((
            '_async_pool',
            '_unused_conns',
            '_lock',
        ))
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in ignored
        }

    def __setstate__(self, dct: Dict[str, Any]) -> None:
        """
        Restores the object's state after unpickling, reinitializing ephemeral objects.

        :param dct: The saved state dictionary.
        """
        self.__dict__ = dct
        pool_size = self._async_pool_size
        if pool_size is None:
            self._async_pool = None
        else:
            self._async_pool = ThreadPoolExecutor(pool_size)
        self._unused_conns = set()
        self._lock = threading.RLock()

    # connection and initialization

    @asyn.asyncf
    async def connect(self, timeout: Optional[int] = None,
                      check_boot_completed: Optional[bool] = True,
                      max_async: Optional[int] = None) -> None:
        """
        Connect to the target (e.g., via SSH or another transport).

        :param timeout: Timeout (in seconds) for connecting.
        :param check_boot_completed: If ``True``, verify the target has booted.
        :param max_async: The number of parallel async connections to allow.

        :raises TargetError: If the device fails to connect within the specified time.
        """
        self.platform.init_target_connection(self)  # type: ignore
        # Forcefully set the thread-local value for the connection, with the
        # timeout we want
        self.conn = self.get_connection(timeout=timeout)
        if check_boot_completed:
            self.wait_boot_complete(timeout)
        self.check_connection()
        self._resolve_paths()
        assert self.working_directory
        if self.executables_directory is None:
            self.executables_directory = self.path.join(
                self.working_directory,
                'bin'
            )

        for path in (self.working_directory, self.executables_directory):
            self.makedirs(path)

        self.busybox = self.install(os.path.join(PACKAGE_BIN_DIRECTORY, self.abi, 'busybox'), timeout=30)
        self.conn.busybox = self.busybox

        # If neither the mktemp call nor _resolve_paths() managed to get a
        # temporary directory, we just make one in the working directory.
        if self.tmp_directory is None:
            assert self.busybox
            try:
                tmp = await self.execute.asyn(f'{quote(self.busybox)} mktemp -d')
            except Exception:
                # Some Android platforms don't have a working mktemp unless
                # TMPDIR is set, so we let AndroidTarget._resolve_paths() deal
                # with finding a suitable location.
                tmp = self.path.join(self.working_directory, 'tmp')
            else:
                tmp = tmp.strip()
            self.tmp_directory = tmp
        self.makedirs(self.tmp_directory)

        self._detect_max_async(max_async or self._max_async)
        self.platform.update_from_target(self)  # type: ignore
        self._update_modules('connected')

    def _detect_max_async(self, max_async: int) -> None:
        """
        Attempt to detect the maximum number of parallel asynchronous
        commands the target can handle by opening multiple connections.

        :param max_async: Upper bound for parallel async connections.
        """
        self.logger.debug('Detecting max number of async commands ...')

        def make_conn(_) -> Optional[SupportedConnections]:
            """
            create a connection to target to execute a command
            """
            try:
                conn = self.get_connection()
            except Exception:
                return None
            else:
                payload = 'hello'
                # Sanity check the connection, in case we managed to connect
                # but it's actually unusable.
                try:
                    res = conn.execute(f'echo {quote(payload)}')
                except Exception:
                    return None
                else:
                    if res.strip() == payload:
                        return conn
                    else:
                        return None

        # Logging needs to be disabled before the thread pool is created,
        # otherwise the logging config will not be taken into account
        logging.disable()
        try:
            # Aggressively attempt to create all the connections in parallel,
            # so that this setup step does not take too much time.
            with ThreadPoolExecutor(max_async) as pool:
                conns = pool.map(make_conn, range(max_async))
        # Avoid polluting the log with errors coming from broken
        # connections.
        finally:
            logging.disable(logging.NOTSET)

        resultconns = {conn for conn in conns if conn is not None}

        # Keep the connection so it can be reused by future threads
        self._unused_conns.update(resultconns)
        max_conns = len(resultconns)

        self.logger.debug(f'Detected max number of async commands: {max_conns}')
        self._async_pool_size = max_conns
        self._async_pool = ThreadPoolExecutor(max_conns)

    @asyn.asyncf
    async def check_connection(self) -> None:
        """
        Perform a quick command to verify the target's shell is responsive.

        :raises TargetStableError: If the shell is present but not functioning
            correctly (e.g., output on stderr).
        :raises TargetNotRespondingError: If the target is unresponsive.
        """
        async def check(*, as_root: Union[Literal[False], Literal[True], str] = False) -> None:
            out = await self.execute.asyn('true', as_root=as_root)
            if out:
                raise TargetStableError('The shell seems to not be functional and adds content to stderr: {!r}'.format(out))

        await check(as_root=False)
        # If we are rooted, we usually run with sudo. Unfortunately, PAM
        # modules can write random text to stdout such as:
        # Your password will expire in XXX days.
        if self.is_rooted:
            await check(as_root=True)

    def disconnect(self) -> None:
        """
        Close all active connections to the target and terminate any
        connection threads or asynchronous operations.
        """
        with self._lock:
            thread_conns: Set[SupportedConnections] = self._conn.get_all_values()  # type: ignore
            # Now that we have all the connection objects, we simply reset the
            # TLS property so that the connections we obtained will not be
            # reused anywhere.
            del self._conn  # type: ignore

            unused_conns = list(self._unused_conns)
            self._unused_conns.clear()

            for conn in itertools.chain(thread_conns, unused_conns):
                conn.close()

            pool = self._async_pool
            self._async_pool = None
            if pool is not None:
                pool.__exit__(None, None, None)

    def __enter__(self):
        """
        Context manager entrypoint. Returns self.
        """
        return self

    def __exit__(self, *args, **kwargs):
        """
        Context manager exitpoint. Automatically disconnects from the device.
        """
        self.disconnect()

    async def __aenter__(self):
        """
        Async context manager entry.
        """
        return self.__enter__()

    async def __aexit__(self, *args, **kwargs):
        """
        Async context manager exit.
        """
        return self.__exit__(*args, **kwargs)

    def get_connection(self, timeout: Optional[int] = None) -> SupportedConnections:
        """
        Get an additional connection to the target. A connection can be used to
        execute one blocking command at time. This will return a connection that can
        be used to interact with a target in parallel while a blocking operation is
        being executed.

        This should *not* be used to establish an initial connection; use
        :meth:`connect()` instead.

        :param timeout: Timeout (in seconds) for establishing the connection.
        :returns: A new connection object to be used by the caller.
        :raises ValueError: If no connection class (`conn_cls`) is set.

        .. note:: :class:`~devlib.target.Target` will automatically create a connection
             per thread, so you don't normally need to use this explicitly in
             threaded code. This is generally useful if you want to perform a
             blocking operation (e.g. using :class:`background()`) while at the same
             time doing something else in the same host-side thread.
        """
        if self.conn_cls is None:
            raise ValueError('Connection class not specified on Target creation.')
        conn: SupportedConnections = self.conn_cls(timeout=timeout, **self.connection_settings)  # pylint: disable=not-callable
        # This allows forwarding the detected busybox for connections created in new threads.
        conn.busybox = self.busybox
        return conn

    def wait_boot_complete(self, timeout: Optional[int] = 10) -> None:
        """
        Wait for the device to boot. Must be overridden by derived classes
        if the device needs a specific boot-completion check.

        :param timeout: How long to wait for the device to finish booting.
        :raises NotImplementedError: If not implemented in child classes.
        """
        raise NotImplementedError()

    @asyn.asyncf
    async def setup(self, executables: Optional[List[str]] = None) -> None:
        """
        This will perform an initial one-time set up of a device for devlib
        interaction. This involves deployment of tools relied on the
        :class:`~devlib.target.Target`, creation of working locations on the device,
        etc.

        Usually, it is enough to call this method once per new device, as its effects
        will persist across reboots. However, it is safe to call this method multiple
        times. It may therefore be a good practice to always call it once at the
        beginning of a script to ensure that subsequent interactions will succeed.

        Optionally, this may also be used to deploy additional tools to the device
        by specifying a list of binaries to install in the ``executables`` parameter.

        :param executables: Optional list of host-side binaries to install
            on the target during setup.
        """
        await self._setup_scripts.asyn()

        for host_exe in (executables or []):  # pylint: disable=superfluous-parens
            await self.install.asyn(host_exe)

        # Check for platform dependent setup procedures
        self.platform.setup(self)  # type: ignore

        # Initialize modules which requires Busybox (e.g. shutil dependent tasks)
        self._update_modules('setup')

    def reboot(self, hard: bool = False, connect: bool = True, timeout: int = 180) -> None:
        """
        Reboot the target. Optionally performs a hard reset if supported
        by a :class:`HardRestModule`.

        :param hard: If ``True``, use a hard reset.
        :param connect: If ``True``, reconnect after reboot finishes.
        :param timeout: Timeout in seconds for reconnection.

        :raises TargetStableError: If hard reset is requested but not supported.
        :raises TargetTransientError: If the target is not currently connected
            and a soft reset is requested.
        """
        if hard:
            if not self.has('hard_reset'):
                raise TargetStableError('Hard reset not supported for this target.')
            cast(HardRestModule, self.hard_reset)()  # pylint: disable=no-member
        else:
            if not self.is_connected:
                message = 'Cannot reboot target because it is disconnected. ' +\
                          'Either connect() first, or specify hard=True ' +\
                          '(in which case, a hard_reset module must be installed)'
                raise TargetTransientError(message)
            self.reset()
            # Wait a fixed delay before starting polling to give the target time to
            # shut down, otherwise, might create the connection while it's still shutting
            # down resulting in subsequent connection failing.
            self.logger.debug('Waiting for target to power down...')
            reset_delay = 20
            time.sleep(reset_delay)
            timeout = max(timeout - reset_delay, 10)
        if self.has('boot'):
            cast(BootModule, self.boot)()  # pylint: disable=no-member
        self.conn.connected_as_root = None
        if connect:
            self.connect(timeout=timeout)

    # file transfer

    @asyn.asynccontextmanager
    async def _xfer_cache_path(self, name: str) -> AsyncGenerator[str, None]:
        """
        Context manager to provide a unique path in the transfer cache with the
        basename of the given name.
        """
        # Make sure basename will work on folders too
        name = os.path.normpath(name)
        name = os.path.basename(name)
        async with self.make_temp() as tmp:
            yield self.path.join(tmp, name)

    @asyn.asyncf
    async def _prepare_xfer(self, action: str, sources: List[str], dest: str,
                            pattern: Optional[str] = None, as_root: bool = False) -> Dict[Tuple[str, ...], str]:
        """
        Check the sanity of sources and destination and prepare the ground for
        transfering multiple sources.
        """

        def once(f):
            cache = dict()

            @functools.wraps(f)
            async def wrapper(path):
                try:
                    return cache[path]
                except KeyError:
                    x = await f(path)
                    cache[path] = x
                    return x

            return wrapper

        _target_cache: Dict[str, Optional[str]] = {}

        async def target_paths_kind(paths: List[str], as_root: bool = False) -> List[Optional[str]]:
            def process(x: str) -> Optional[str]:
                x = x.strip()
                if x == 'notexist':
                    return None
                else:
                    return x

            _paths: List[str] = [
                path
                for path in paths
                if path not in _target_cache
            ]
            if _paths:
                cmd = '; '.join(
                    'if [ -d {path} ]; then echo dir; elif [ -e {path} ]; then echo file; else echo notexist; fi'.format(
                        path=quote(path)
                    )
                    for path in _paths
                )
                res = await self.execute.asyn(cmd, as_root=as_root)
                _target_cache.update(zip(_paths, map(process, res.split())))

            return [
                _target_cache[path]
                for path in paths
            ]

        _host_cache: Dict[str, Optional[str]] = {}

        async def host_paths_kind(paths: List[str], as_root: bool = False) -> List[Optional[str]]:
            def path_kind(path: str) -> Optional[str]:
                if os.path.isdir(path):
                    return 'dir'
                elif os.path.exists(path):
                    return 'file'
                else:
                    return None

            for path in paths:
                if path not in _host_cache:
                    _host_cache[path] = path_kind(path)

            return [
                _host_cache[path]
                for path in paths
            ]

        # TODO: Target.remove() and Target.makedirs() would probably benefit
        # from being implemented by connections, with the current
        # implementation in ConnectionBase. This would allow SshConnection to
        # use SFTP for these operations, which should be cheaper than
        # Target.execute()
        if action == 'push':
            src_excep: Type[DevlibError] = HostError
            src_path_kind = host_paths_kind

            _dst_mkdir = once(self.makedirs.asyn)

            dst_path_join = self.path.join
            dst_paths_kind = target_paths_kind

            @once
            async def dst_remove_file(path: str):  # type:ignore
                return await self.remove.asyn(path, as_root=as_root)
        elif action == 'pull':
            src_excep = TargetStableError
            src_path_kind = target_paths_kind

            @once
            async def _dst_mkdir(path: str):
                return os.makedirs(path, exist_ok=True)
            dst_path_join = os.path.join
            dst_paths_kind = host_paths_kind

            @once
            async def dst_remove_file(path: str):
                return os.remove(path)
        else:
            raise ValueError('Unknown action "{}"'.format(action))

        # Handle the case where path is None
        async def dst_mkdir(path: Optional[str]) -> None:
            if path:
                await _dst_mkdir(path)

        async def rewrite_dst(src: str, dst: str) -> str:
            new_dst: str = dst_path_join(dst, os.path.basename(src))

            src_kind, = await src_path_kind([src], as_root)
            # Batch both checks to avoid a costly extra execute()
            dst_kind, new_dst_kind = await dst_paths_kind([dst, new_dst], as_root)

            if src_kind == 'file':
                if dst_kind == 'dir':
                    if new_dst_kind == 'dir':
                        raise IsADirectoryError(new_dst)
                    if new_dst_kind == 'file':
                        await dst_remove_file(new_dst)
                        return new_dst
                    else:
                        return new_dst
                elif dst_kind == 'file':
                    await dst_remove_file(dst)
                    return dst
                else:
                    await dst_mkdir(os.path.dirname(dst))
                    return dst
            elif src_kind == 'dir':
                if dst_kind == 'dir':
                    # Do not allow writing over an existing folder
                    if new_dst_kind == 'dir':
                        raise FileExistsError(new_dst)
                    if new_dst_kind == 'file':
                        raise FileExistsError(new_dst)
                    else:
                        return new_dst
                elif dst_kind == 'file':
                    raise FileExistsError(dst_kind)
                else:
                    await dst_mkdir(os.path.dirname(dst))
                    return dst
            else:
                raise FileNotFoundError(src)

        if pattern:
            if not sources:
                raise src_excep('No file matching source pattern: {}'.format(pattern))

            if (await dst_paths_kind([dest])) != ['dir']:
                raise NotADirectoryError('A folder dest is required for multiple matches but destination is a file: {}'.format(dest))

        async def f(src):
            return await rewrite_dst(src, dest)
        mapping = await self.async_manager.map_concurrently(f, sources)

        # TODO: since rewrite_dst() will currently return a different path for
        # each source, it will not bring anything. In order to be useful,
        # connections need to be able to understand that if the destination is
        # an empty folder, the source is supposed to be transfered into it with
        # the same basename.
        return groupby_value(mapping)

    @asyn.asyncf
    @call_conn
    async def push(self, source: str, dest: str, as_root: bool = False,
                   timeout: Optional[int] = None, globbing: bool = False) -> None:  # pylint: disable=arguments-differ
        """
        Transfer a file from the host machine to the target device.

        If transfer polling is supported (ADB connections and SSH connections),
        ``poll_transfers`` is set in the connection, and a timeout is not specified,
        the push will be polled for activity. Inactive transfers will be
        cancelled. (See :ref:`connection-types` for more information on polling).

        :param source: path on the host
        :param dest: path on the target
        :param as_root: whether root is required. Defaults to false.
        :param timeout: timeout (in seconds) for the transfer; if the transfer does
            not complete within this period, an exception will be raised. Leave unset
            to utilise transfer polling if enabled.
        :param globbing: If ``True``, the ``source`` is interpreted as a globbing
                pattern instead of being take as-is. If the pattern has multiple
                matches, ``dest`` must be a folder (or will be created as such if it
                does not exists yet).

        :raises TargetStableError: If any failure occurs in copying
            (e.g., insufficient permissions).

        """
        source = str(source)
        dest = str(dest)

        sources: List[str] = glob.glob(source) if globbing else [source]
        mapping: Dict[List[str], str] = await self._prepare_xfer.asyn('push', sources, dest, pattern=source if globbing else None, as_root=as_root)

        def do_push(sources: List[str], dest: str) -> None:
            for src in sources:
                self.async_manager.track_access(
                    asyn.PathAccess(namespace='host', path=src, mode='r')
                )
            self.async_manager.track_access(
                asyn.PathAccess(namespace='target', path=dest, mode='w')
            )
            self.conn.push(sources, dest, timeout=timeout)

        if as_root:
            for sources, dest in mapping.items():
                async def f(source):
                    async with self._xfer_cache_path(source) as device_tempfile:
                        do_push([source], device_tempfile)
                        await self.execute.asyn("mv -f -- {} {}".format(quote(device_tempfile), quote(dest)), as_root=True)
                await self.async_manager.map_concurrently(f, sources)
        else:
            for sources_map, dest_map in mapping.items():
                do_push(sources_map, dest_map)

    @asyn.asyncf
    async def _expand_glob(self, pattern: str, **kwargs: Dict[str, bool]) -> Optional[List[str]]:
        """
        Expand the given path globbing pattern on the target using the shell
        globbing.
        """
        # Since we split the results based on new lines, forbid them in the
        # pattern
        if '\n' in pattern:
            raise ValueError(r'Newline character \n are not allowed in globbing patterns')

        # If the pattern is in fact a plain filename, skip the expansion on the
        # target to avoid an unncessary command execution.
        #
        # fnmatch char list from: https://docs.python.org/3/library/fnmatch.html
        special_chars = ['*', '?', '[', ']']
        if not any(char in pattern for char in special_chars):
            return [pattern]

        # Characters to escape that are impacting parameter splitting, since we
        # want the pattern to be given in one piece. Unfortunately, there is no
        # fool-proof way of doing that without also escaping globbing special
        # characters such as wildcard which would defeat the entire purpose of
        # that function.
        for c in [' ', "'", '"']:
            pattern = pattern.replace(c, '\\' + c)

        cmd = "exec printf '%s\n' {}".format(pattern)
        if self.busybox:
            # Make sure to use the same shell everywhere for the path globbing,
            # ensuring consistent results no matter what is the default platform
            # shell
            cmd = '{} sh -c {} 2>/dev/null'.format(quote(self.busybox), quote(cmd))
            # On some shells, match failure will make the command "return" a
            # non-zero code, even though the command was not actually called
            result: str = await self.execute.asyn(cmd, strip_colors=False, check_exit_code=False, **kwargs)
            paths: List[str] = result.splitlines()
            if not paths:
                raise TargetStableError('No file matching: {}'.format(pattern))

            return paths
        return None

    @asyn.asyncf
    @call_conn
    async def pull(self, source: str, dest: str, as_root: bool = False,
                   timeout: Optional[int] = None, globbing: bool = False,
                   via_temp: bool = False) -> None:  # pylint: disable=arguments-differ
        """
        Transfer a file from the target device to the host machine.

        If transfer polling is supported (ADB connections and SSH connections),
        ``poll_transfers`` is set in the connection, and a timeout is not specified,
        the pull will be polled for activity. Inactive transfers will be
        cancelled. (See :ref:`connection-types` for more information on polling).

        :param source: path on the target
        :param dest: path on the host
        :param as_root: whether root is required. Defaults to false.
        :param timeout: timeout (in seconds) for the transfer; if the transfer does
            not  complete within this period, an exception will be raised.
        :param globbing: If ``True``, the ``source`` is interpreted as a globbing
                pattern instead of being take as-is. If the pattern has multiple
                matches, ``dest`` must be a folder (or will be created as such if it
                does not exists yet).
        :param via_temp: If ``True``, copy the file first to a temporary location on
                the target, and then pull it. This can avoid issues some filesystems,
                notably paramiko + OpenSSH combination having performance issues when
                pulling big files from sysfs.

        :raises TargetStableError: If a transfer error occurs.
        """
        source = str(source)
        dest = str(dest)

        if globbing:
            sources: Optional[List[str]] = await self._expand_glob.asyn(source, as_root=as_root)
            if sources is None:
                sources = [source]
        else:
            sources = [source]

        # The SSH server might not have the right permissions to read the file,
        # so use a temporary copy instead.
        via_temp |= as_root

        mapping: Dict[List[str], str] = await self._prepare_xfer.asyn('pull', sources, dest, pattern=source if globbing else None, as_root=as_root)

        def do_pull(sources: List[str], dest: str) -> None:
            for src in sources:
                self.async_manager.track_access(
                    asyn.PathAccess(namespace='target', path=src, mode='r')
                )
            self.async_manager.track_access(
                asyn.PathAccess(namespace='host', path=dest, mode='w')
            )
            self.conn.pull(sources, dest, timeout=timeout)

        if via_temp:
            for sources, dest in mapping.items():
                async def f(source):
                    async with self._xfer_cache_path(source) as device_tempfile:
                        cp_cmd = f"{quote(self.busybox or '')} cp -rL -- {quote(source)} {quote(device_tempfile)}"
                        chmod_cmd = f"{quote(self.busybox or '')} chmod 0644 -- {quote(device_tempfile)}"
                        await self.execute.asyn(f"{cp_cmd} && {chmod_cmd}", as_root=as_root)
                        do_pull([device_tempfile], dest)
                await self.async_manager.map_concurrently(f, sources)
        else:
            for sources_map, dest_map in mapping.items():
                do_pull(sources_map, dest_map)

    @asyn.asyncf
    async def get_directory(self, source_dir: str, dest: str,
                            as_root: bool = False) -> None:
        """ Pull a directory from the device, after compressing dir """
        # Create all file names
        tar_file_name: str = source_dir.lstrip(self.path.sep).replace(self.path.sep, '.')
        # Host location of dir
        outdir: str = os.path.join(dest, tar_file_name)
        # Host location of archive
        tar_file_name = '{}.tar'.format(tar_file_name)
        tmpfile: str = os.path.join(dest, tar_file_name)

        # If root is required, use tmp location for tar creation.
        tar_file_cm: Union[Callable[[str], AsyncContextManager[str]], Callable[[str], nullcontext]] = self._xfer_cache_path if as_root else nullcontext

        # Does the folder exist?
        await self.execute.asyn('ls -la {}'.format(quote(source_dir)), as_root=as_root)

        async with tar_file_cm(tar_file_name) as tar_file:
            # Try compressing the folder
            try:
                # FIXME - should we raise an error in the else case here when busybox or tar_file is None
                if self.busybox and tar_file:
                    await self.execute.asyn('{} tar -cvf {} {}'.format(
                        quote(self.busybox), quote(tar_file), quote(source_dir)
                    ), as_root=as_root)
            except TargetStableError:
                self.logger.debug('Failed to run tar command on target! '
                                  'Not pulling directory {}'.format(source_dir))
            # Pull the file
            if not os.path.exists(dest):
                os.mkdir(dest)
            await self.pull.asyn(tar_file, tmpfile)
            # Decompress
            with tarfile.open(tmpfile, 'r') as f:
                safe_extract(f, outdir)
            os.remove(tmpfile)

    # execution

    def _prepare_cmd(self, command: SubprocessCommand, force_locale: str) -> SubprocessCommand:
        """
        Internal helper to prepend environment settings (e.g., PATH, locale)
        to a command string before execution.

        :param command: The command to execute.
        :param force_locale: The locale to enforce (e.g. 'C') or None for none.
        :return: The updated command string with environment preparation.
        """
        tmpdir = f'TMPDIR={quote(self.tmp_directory)}' if self.tmp_directory else ''

        # Force the locale if necessary for more predictable output
        if force_locale:
            # Use an explicit export so that the command is allowed to be any
            # shell statement, rather than just a command invocation
            command = f'export LC_ALL={quote(force_locale)} {tmpdir} && {cast(str, command)}'

        # Ensure to use deployed command when availables
        if self.executables_directory:
            command = f"export PATH={quote(self.executables_directory)}:$PATH && {cast(str, command)}"

        return command

    class _BrokenConnection(Exception):
        pass

    @asyn.asyncf
    @call_conn
    async def _execute_async(self, *args: Any, **kwargs: Any) -> str:
        """
        Internal asynchronous handler for command execution.

        This is typically invoked by the asynchronous version of :meth:`execute`.
        It may create a background thread or use an existing thread pool
        to run the blocking command.

        :param args: Positional arguments forwarded to the blocking command.
        :param kwargs: Keyword arguments forwarded to the blocking command.
        :return: The stdout of the command executed.
        :raises DevlibError: If any error occurs during command execution.
        """
        execute = functools.partial(
            self._execute,
            *args, **kwargs
        )
        pool: Optional[ThreadPoolExecutor] = self._async_pool

        if pool is None:
            return execute()
        else:

            def thread_f() -> str:
                # If we cannot successfully connect from the thread, it might
                # mean that something external opened a connection on the
                # target, so we just revert to the blocking path.
                try:
                    self.conn
                except Exception:
                    raise self._BrokenConnection
                else:
                    return execute()

            loop: AbstractEventLoop = asyncio.get_running_loop()
            try:
                return await loop.run_in_executor(pool, thread_f)
            except self._BrokenConnection:
                return execute()

    @call_conn
    def _execute(self, command: SubprocessCommand, timeout: Optional[int] = None, check_exit_code: bool = True,
                 as_root: bool = False, strip_colors: bool = True, will_succeed: bool = False,
                 force_locale: str = 'C') -> str:
        """
        Internal blocking command executor. Actual synchronous logic is placed here,
        usually invoked by :meth:`execute`.

        :param command: The command to be executed.
        :param timeout: Timeout (in seconds) for the execution of the command. If
            specified, an exception will be raised if execution does not complete
            with the specified period.
        :param check_exit_code: If ``True`` (the default) the exit code (on target)
            from execution of the command will be checked, and an exception will be
            raised if it is not ``0``.
        :param as_root: The command will be executed as root. This will fail on
            unrooted targets.
        :param strip_colours: The command output will have colour encodings and
            most ANSI escape sequences striped out before returning.
        :param will_succeed: The command is assumed to always succeed, unless there is
            an issue in the environment like the loss of network connectivity. That
            will make the method always raise an instance of a subclass of
            :class:`DevlibTransientError` when the command fails, instead of a
            :class:`DevlibStableError`.
        :param force_locale: Prepend ``LC_ALL=<force_locale>`` in front of the
            command to get predictable output that can be more safely parsed.
            If ``None``, no locale is prepended.
        """

        command = self._prepare_cmd(command, force_locale)
        return self.conn.execute(command, timeout=timeout,
                                 check_exit_code=check_exit_code, as_root=as_root,
                                 strip_colors=strip_colors, will_succeed=will_succeed)

    execute = asyn._AsyncPolymorphicFunction(
        asyn=_execute_async.asyn,
        blocking=_execute,
    )

    @call_conn
    def background(self, command: SubprocessCommand, stdout: int = subprocess.PIPE,
                   stderr: int = subprocess.PIPE, as_root: Optional[bool] = False,
                   force_locale: str = 'C', timeout: Optional[int] = None) -> BackgroundCommand:
        """
        Execute the command on the target, invoking it via subprocess on the host.
        This will return :class:`subprocess.Popen` instance for the command.

        :param command: The command to be executed.
        :param stdout: By default, standard output will be piped from the subprocess;
            this may be used to redirect it to an alternative file handle.
        :param stderr: By default, standard error will be piped from the subprocess;
            this may be used to redirect it to an alternative file handle.
        :param as_root: The command will be executed as root. This will fail on
            unrooted targets.
        :param force_locale: Prepend ``LC_ALL=<force_locale>`` in front of the
            command to get predictable output that can be more safely parsed.
            If ``None``, no locale is prepended.
        :param timeout: Timeout (in seconds) for the execution of the command. When
            the timeout expires, :meth:`BackgroundCommand.cancel` is executed to
            terminate the command.

        :return: A handle to the background command.

        .. note:: This **will block the connection** until the command completes.
        """
        command = self._prepare_cmd(command, force_locale)
        bg_cmd: BackgroundCommand = self.conn.background(command, stdout, stderr, as_root)
        if timeout is not None:
            timer = threading.Timer(timeout, function=bg_cmd.cancel)
            timer.daemon = True
            timer.start()
        return bg_cmd

    def invoke(self, binary: str, args: Optional[Union[str, Iterable[str]]] = None, in_directory: Optional[str] = None,
               on_cpus: Optional[Union[int, List[int], str]] = None, redirect_stderr: bool = False, as_root: bool = False,
               timeout: Optional[int] = 30) -> str:
        """
        Executes the specified binary under the specified conditions.

        :param binary: binary to execute. Must be present and executable on the device.
        :param args: arguments to be passed to the binary. The can be either a list or
               a string.
        :param in_directory:  execute the binary in the  specified directory. This must
                        be an absolute path.
        :param on_cpus:  taskset the binary to these CPUs. This may be a single ``int`` (in which
                   case, it will be interpreted as the mask), a list of ``ints``, in which
                   case this will be interpreted as the list of cpus, or string, which
                   will be interpreted as a comma-separated list of cpu ranges, e.g.
                   ``"0,4-7"``.
        :param redirect_stderr: redirect stderr to stdout
        :param as_root: Specify whether the command should be run as root
        :param timeout: If the invocation does not terminate within this number of seconds,
                  a ``TimeoutError`` exception will be raised. Set to ``None`` if the
                  invocation should not timeout.

        :return: The captured output of the command.
        """
        command = binary
        if args:
            if isiterable(args):
                args = ' '.join(args)
            command = '{} {}'.format(command, args)
        if on_cpus:
            on_cpus_bitmask = bitmask(on_cpus)
            if self.busybox:
                command = '{} taskset 0x{:x} {}'.format(quote(self.busybox), on_cpus_bitmask, command)
        if in_directory:
            command = 'cd {} && {}'.format(quote(in_directory), command)
        if redirect_stderr:
            command = '{} 2>&1'.format(command)
        return self.execute(command, as_root=as_root, timeout=timeout)

    def background_invoke(self, binary: str, args: Optional[Union[str, Iterable[str]]] = None, in_directory: Optional[str] = None,
                          on_cpus: Optional[Union[int, List[int], str]] = None, as_root: bool = False) -> BackgroundCommand:
        """
        Runs the specified binary as a background task, possibly pinned to CPUs or
        launched in a certain directory.

        :param binary: binary to execute. Must be present and executable on the device.
        :param args: arguments to be passed to the binary. The can be either a list or
               a string.
        :param in_directory:  execute the binary in the  specified directory. This must
                        be an absolute path.
        :param on_cpus:  taskset the binary to these CPUs. This may be a single ``int`` (in which
                   case, it will be interpreted as the mask), a list of ``ints``, in which
                   case this will be interpreted as the list of cpus, or string, which
                   will be interpreted as a comma-separated list of cpu ranges, e.g.
                   ``"0,4-7"``.
        :param as_root: Specify whether the command should be run as root

        :returns: the subprocess instance handling that command

        :raises TargetError: If the binary does not exist or is not executable.
        """
        command = binary
        if args:
            if isiterable(args):
                args = ' '.join(args)
            command = '{} {}'.format(command, args)
        if on_cpus:
            on_cpus_bitmask = bitmask(on_cpus)
            if self.busybox:
                command = '{} taskset 0x{:x} {}'.format(quote(self.busybox), on_cpus_bitmask, command)
            else:
                raise TargetStableError("busybox not set. cannot execute command")
        if in_directory:
            command = 'cd {} && {}'.format(quote(in_directory), command)
        return self.background(command, as_root=as_root)

    @asyn.asyncf
    async def kick_off(self, command: str, as_root: Optional[bool] = None) -> None:
        """
        Kick off the specified command on the target and return immediately. Unlike
        ``background()`` this will not block the connection; on the other hand, there
        is not way to know when the command finishes (apart from calling ``ps()``)
        or to get its output (unless its redirected into a file that can be pulled
        later as part of the command).

        :param command: The command to be executed.
        :param as_root: The command will be executed as root. This will fail on
            unrooted targets.

        :raises TargetError: If the command cannot be launched.
        """
        if self.working_directory and self.busybox:
            cmd = 'cd {wd} && {busybox} sh -c {cmd} >/dev/null 2>&1'.format(
                wd=quote(self.working_directory),
                busybox=quote(self.busybox),
                cmd=quote(command)
            )
        else:
            raise TargetStableError("working directory or busybox not set. cannot kick off command")
        self.background(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, as_root=as_root)

    # sysfs interaction
    R = TypeVar('R')

    @asyn.asyncf
    async def read_value(self, path: str, kind: Optional[Callable[[str], R]] = None) -> Union[str, R]:
        """
        Read the value from the specified path. This is primarily intended for
        sysfs/procfs/debugfs etc.

        :param path: file to read
        :param kind: Optionally, read value will be converted into the specified
                kind (which should be a callable that takes exactly one parameter)

        :return: The contents of the file, possibly parsed via ``kind``.
        :raises TargetStableError: If the file does not exist or is unreadable.
        """
        self.async_manager.track_access(
            asyn.PathAccess(namespace='target', path=path, mode='r')
        )
        output: str = await self.execute.asyn('cat {}'.format(quote(path)), as_root=self.needs_su)  # pylint: disable=E1103
        output = output.strip()
        if kind and callable(kind) and output:
            try:
                return kind(output)
            except Exception as e:
                raise ValueError(f"Error converting output using {kind}: {e}")
        else:
            return output

    @asyn.asyncf
    async def read_int(self, path: str) -> int:
        """
        Equivalent to ``Target.read_value(path, kind=devlib.utils.types.integer)``

        :param path: The file path to read.
        :return: The integer value contained in the file.
        :raises ValueError: If the file contents cannot be parsed as an integer.
        """
        return await self.read_value.asyn(path, kind=integer)

    @asyn.asyncf
    async def read_bool(self, path: str) -> bool:
        """
        Equivalent to ``Target.read_value(path, kind=devlib.utils.types.boolean)``

        :param path: File path to read.
        :return: True or False, parsed from the file content.
        :raises ValueError: If the file contents cannot be interpreted as a boolean.
        """
        return await self.read_value.asyn(path, kind=boolean)

    @asyn.asynccontextmanager
    async def revertable_write_value(self, path: str, value: Any, verify: bool = True, as_root: bool = True) -> AsyncGenerator:
        """
        Same as :meth:`Target.write_value`, but as a context manager that will write
        back the previous value on exit.

        :param path: The file path to write to on the target.
        :param value: The value to write, converted to a string.
        :param verify: If True, read the file back to confirm the change.
        :param as_root: If True, write as root.
        :yield: Allows running code in the context while the value is changed.
        """
        orig_value: str = self.read_value(path)
        try:
            await self.write_value.asyn(path, value, verify=verify, as_root=as_root)
            yield
        finally:
            await self.write_value.asyn(path, orig_value, verify=verify, as_root=as_root)

    def batch_revertable_write_value(self, kwargs_list: List[Dict[str, Any]]) -> '_GeneratorContextManager':
        """
        Calls :meth:`Target.revertable_write_value` with all the keyword arguments
        dictionary given in the list. This is a convenience method to update
        multiple files at once, leaving them in their original state on exit. If one
        write fails, all the already-performed writes will be reverted as well.

        :param kwargs_list: A list of dicts, each containing the kwargs for
            :meth:`revertable_write_value`, e.g., {"path": <str>, "value": <obj>, ...}.
        :return: A context manager that applies all writes on entry, then reverts them.
        """
        return batch_contextmanager(self.revertable_write_value, kwargs_list)

    @asyn.asyncf
    async def write_value(self, path: str, value: Any, verify: bool = True, as_root: bool = True) -> None:
        """
        Write the value to the specified path on the target. This is primarily
        intended for sysfs/procfs/debugfs etc.

        :param path: file to write into
        :param value: value to be written
        :param verify: If ``True`` (the default) the value will be read back after
            it is written to make sure it has been written successfully. This due to
            some sysfs entries silently failing to set the written value without
            returning an error code.
        :param as_root: specifies if writing requires being root. Its default value
            is ``True``.

        :raises TargetStableError: If the write or verification fails.
        """
        self.async_manager.track_access(
            asyn.PathAccess(namespace='target', path=path, mode='w')
        )
        string_value = str(value)

        if verify:
            # Check in a loop for a while since updates to sysfs files can take
            # some time to be observed, typically when a write triggers a
            # lengthy kernel-side request, and the read is based on some piece
            # of state that may take some time to be updated by the write
            # request, such as hotplugging a CPU.
            cmd = '''
orig=$(cat {path} 2>/dev/null || printf "")
printf "%s" {string_value} > {path} || exit 10
if [ {string_value} != "$orig" ]; then
   trials=0
   while [ "$(cat {path} 2>/dev/null)" != {string_value} ]; do
       if [ $trials -ge 10 ]; then
           cat {path}
           exit 11
       fi
       sleep 0.01
       trials=$((trials + 1))
   done
fi
'''
        else:
            cmd = '{busybox} printf "%s" {string_value} > {path}'
        if self.busybox:
            cmd = cmd.format(busybox=quote(self.busybox), path=quote(path), string_value=quote(string_value))

        try:
            await self.execute.asyn(cmd, check_exit_code=True, as_root=as_root)
        except TargetCalledProcessError as e:
            if e.returncode == 10:
                raise TargetStableError('Could not write "{string_value}" to {path}: {e.output}'.format(
                    string_value=string_value, path=path, e=e))
            elif verify and e.returncode == 11:
                out = e.output
                message = 'Could not set the value of {} to "{}" (read "{}")'.format(path, string_value, out)
                raise TargetStableError(message)
            else:
                raise

    @asyn.asynccontextmanager
    async def make_temp(self, is_directory: Optional[bool] = True, directory: Optional[str] = None,
                        prefix: Optional[str] = None) -> AsyncGenerator:
        """
        Creates temporary file/folder on target and deletes it once it's done.

        :param is_directory: Specifies if temporary object is a directory, defaults to True.

        :param directory: Temp object will be created under this directory,
            defaults to ``Target.working_directory``.

        :param prefix: Prefix of temp object's name.

        :yield: Full path of temp object.
        """

        directory = directory or self.tmp_directory
        prefix = f'{prefix}-' if prefix else ''
        temp_obj = None
        try:
            if directory is not None:
                cmd = f'mktemp -p {quote(directory)} {quote(prefix)}XXXXXX'
                if is_directory:
                    cmd += ' -d'

                    temp_obj = (await self.execute.asyn(cmd)).strip()
                    yield temp_obj
        finally:
            if temp_obj is not None:
                await self.remove.asyn(temp_obj)

    def reset(self) -> None:
        """
        Soft reset the target. Typically, this means executing ``reboot`` on the
        target.
        """
        try:
            self.execute('reboot', as_root=self.needs_su, timeout=2)
        except (TargetError, subprocess.CalledProcessError):
            # on some targets "reboot" doesn't return gracefully
            pass
        self.conn.connected_as_root = None

    @call_conn
    def check_responsive(self, explode: bool = True) -> bool:
        """
        Returns ``True`` if the target appears to be responsive and ``False``
        otherwise.
        """
        try:
            self.conn.execute('ls /', timeout=5)
            return True
        except (DevlibTransientError, subprocess.CalledProcessError):
            if explode:
                raise TargetNotRespondingError('Target {} is not responding'.format(self.conn.name))
            return False

    # process management

    def kill(self, pid: int, signal: Optional[signal.Signals] = None, as_root: Optional[bool] = False) -> None:
        """
        Send a signal (default SIGTERM) to a process by PID.

        :param pid: The PID of the process to kill.
        :param signal: The signal to send (e.g., signal.SIGKILL).
        :param as_root: If True, run the kill command as root.
        """
        signal_string = '-s {}'.format(signal) if signal else ''
        self.execute('{} kill {} {}'.format(self.busybox, signal_string, pid), as_root=as_root)

    def killall(self, process_name: str, signal: Optional[signal.Signals] = None,
                as_root: Optional[bool] = False) -> None:
        """
        Send a signal to all processes matching the given name.

        :param process_name: Name of processes to kill.
        :param signal: The signal to send.
        :param as_root: If True, run the kill command as root.
        """
        for pid in self.get_pids_of(process_name):
            try:
                self.kill(pid, signal=signal, as_root=as_root)
            except TargetStableError:
                pass

    def get_pids_of(self, process_name: str) -> List[int]:
        """
        Return a list of PIDs of all running instances of the specified process.
        """
        raise NotImplementedError()

    def ps(self, **kwargs: Dict[str, Any]) -> List['PsEntry']:
        """
        Return a list of :class:`PsEntry` instances for all running processes on the
        system.
        """
        raise NotImplementedError()

    # files

    @asyn.asyncf
    async def makedirs(self, path: str, as_root: bool = False) -> None:
        """
        Create a directory (and its parents if needed) on the target.

        :param path: Directory path to create.
        :param as_root: If True, create as root.
        """
        await self.execute.asyn('mkdir -p {}'.format(quote(path)), as_root=as_root)

    @asyn.asyncf
    async def file_exists(self, filepath: str) -> bool:
        """
        Check if a file or directory exists at the specified path.

        :param filepath: The target path to check.
        :return: True if the path exists on the target, else False.
        """
        command = 'if [ -e {} ]; then echo 1; else echo 0; fi'
        output: str = await self.execute.asyn(command.format(quote(filepath)), as_root=self.is_rooted)
        return boolean(output.strip())

    @asyn.asyncf
    async def directory_exists(self, filepath: str) -> bool:
        """
        Check if the path on the target is an existing directory.

        :param filepath: The path to check.
        :return: True if a directory exists at the path, else False.
        """
        output = await self.execute.asyn('if [ -d {} ]; then echo 1; else echo 0; fi'.format(quote(filepath)))
        # output from ssh my contain part of the expression in the buffer,
        # split out everything except the last word.
        return boolean(output.split()[-1])  # pylint: disable=maybe-no-member

    @asyn.asyncf
    async def list_file_systems(self) -> List['FstabEntry']:
        """
        Return a list of currently mounted file systems, parsed into FstabEntry objects.

        :return: A list of file system entries describing mount points.
        """
        output: str = await self.execute.asyn('mount')
        fstab: List['FstabEntry'] = []
        for line in output.split('\n'):
            line = line.strip()
            if not line:
                continue
            match = FSTAB_ENTRY_REGEX.search(line)
            if match:
                fstab.append(FstabEntry(match.group(1), match.group(2),
                                        match.group(3), match.group(4),
                                        None, None))
            else:  # assume pre-M Android
                fstab.append(FstabEntry(*line.split()))
        return fstab

    @asyn.asyncf
    async def list_directory(self, path: str, as_root: bool = False) -> List[str]:
        """
        Internal method that returns the contents of a directory. Called by
        :meth:`list_directory`.

        :param path: Directory path to list.
        :param as_root: If True, list as root.
        :return: A list of filenames within the directory.
        :raises NotImplementedError: If not implemented in a subclass.
        """
        self.async_manager.track_access(
            asyn.PathAccess(namespace='target', path=path, mode='r')
        )
        return await self._list_directory(path, as_root=as_root)

    async def _list_directory(self, path: str, as_root: bool = False) -> List[str]:
        """
        List the contents of the specified directory. Optionally run as root.

        :param path: Directory path to list.
        :param as_root: If True, run the directory listing as root.
        :return: Names of entries in the directory.
        :raises TargetStableError: If the path is not a directory or is unreadable.
        """
        raise NotImplementedError()

    def get_workpath(self, name: str) -> str:
        """
        Join a name with :attr:`working_directory` on the target, returning
        an absolute path for convenience.

        :param name: The filename to append to the working directory.
        :return: The combined absolute path, or None if no working directory is set.
        """
        return self.path.join(self.working_directory, name)

    @asyn.asyncf
    async def tempfile(self, prefix: Optional[str] = None, suffix: Optional[str] = None) -> Optional[str]:
        """
        Generate a unique path for a temporary file in the :attr:`working_directory`.

        :param prefix: An optional prefix for the file name.
        :param suffix: An optional suffix (e.g. ".txt").
        :return: The full path to the file, which does not yet exist.
        """
        prefix = f'{prefix}-' if prefix else ''
        suffix = f'-{suffix}' if suffix else ''
        name = '{prefix}{uuid}{suffix}'.format(
            prefix=prefix,
            uuid=uuid.uuid4().hex,
            suffix=suffix,
        )
        path = self.path.join(self.tmp_directory, name)
        if (await self.file_exists.asyn(path)):
            raise FileExistsError('Path already exists on the target: {}'.format(path))
        else:
            return path

    @asyn.asyncf
    async def remove(self, path: str, as_root=False) -> None:
        """
        Remove a file or directory on the target.

        :param path: Path to remove.
        :param as_root: If True, remove as root.
        """
        await self.execute.asyn('rm -rf -- {}'.format(quote(path)), as_root=as_root)

    # misc
    @asyn.asyncf
    async def read_sysctl(self, parameter: str) -> Optional[str]:
        """
        Read the specified sysctl parameter. Equivalent to reading the file under
        ``/proc/sys/...``.

        :param parameter: The sysctl name, e.g. "kernel.sched_latency_ns".
        :return: The value of the sysctl parameter, or None if not found.
        :raises ValueError: If the sysctl parameter doesn't exist.
        """
        path: str = self.path.join('/', 'proc', 'sys', *parameter.split('.'))
        try:
            return await self.read_value.asyn(path)
        except FileNotFoundError as e:
            raise ValueError(f'systcl parameter {parameter} was not found: {e}')

    def core_cpus(self, core: str) -> List[int]:
        """
        Return numeric CPU IDs corresponding to the given core name.

        :param core: The name of the CPU core (e.g., "A53").
        :return: List of CPU indices that match the given name.
        """
        return [i for i, c in enumerate(self.core_names) if c == core]

    @asyn.asyncf
    async def list_online_cpus(self, core: Optional[str] = None) -> List[int]:
        """
        Return a list of online CPU IDs. If a core name is provided, restricts
        to CPUs that match that name.

        :param core: Optional name of the CPU core (e.g., "A53") to filter results.
        :return: Online CPU IDs.
        :raises ValueError: If the specified core name is invalid.
        """
        path: str = self.path.join('/sys/devices/system/cpu/online')
        output: str = await self.read_value.asyn(path)
        all_online: List[int] = ranges_to_list(output)
        if core:
            cpus: List[int] = self.core_cpus(core)
            if not cpus:
                raise ValueError(core)
            return [o for o in all_online if o in cpus]
        else:
            return all_online

    @asyn.asyncf
    async def list_offline_cpus(self) -> List[int]:
        """
        Return a list of offline CPU IDs, i.e., those not present in
        :meth:`list_online_cpus`.

        :return: Offline CPU IDs.
        """
        online: List[int] = await self.list_online_cpus.asyn()
        return [c for c in range(self.number_of_cpus)
                if c not in online]

    @asyn.asyncf
    async def getenv(self, variable: str) -> str:
        """
        Return the value of the specified environment variable on the device
        """
        var: str = await self.execute.asyn('printf "%s" ${}'.format(variable))
        return var.rstrip('\r\n')

    def capture_screen(self, filepath: str) -> None:
        """
        Take a screenshot on the device and save it to the specified file on the
        host. This may not be supported by the target. You can optionally insert a
        ``{ts}`` tag into the file name, in which case it will be substituted with
        on-target timestamp of the screen shot in ISO8601 format.

        :param filepath: Path on the host where screenshot is stored.
        :raises NotImplementedError: If screenshot capture is not implemented.
        """
        raise NotImplementedError()

    @asyn.asyncf
    def install(self, filepath: str, timeout: Optional[int] = None, with_name: Optional[str] = None) -> str:
        """
        Install an executable from the host to the target. If `with_name` is given,
        the file is renamed on the target.

        :param filepath: Path on the host to the executable.
        :param timeout: Timeout in seconds for the installation.
        :param with_name: If provided, rename the installed file on the target.
        :return: The path to the installed binary on the target.
        :raises NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError()

    def uninstall(self, name: str) -> None:
        """
        Uninstall a previously installed executable.

        :param name: Name of the executable to remove.
        :raises NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError()

    @asyn.asyncf
    async def get_installed(self, name: str, search_system_binaries: bool = True) -> Optional[str]:
        """
        Return the absolute path of an installed executable with the given name,
        or None if not found.

        :param name: The name of the binary.
        :param search_system_binaries: If True, also search the system PATH.
        :return: Full path to the binary on the target, or None if not found.
        """
        # Check user installed binaries first
        if self.file_exists(self.executables_directory):
            if name in (await self.list_directory.asyn(self.executables_directory)) and self.path:
                return self.path.join(self.executables_directory, name)
        # Fall back to binaries in PATH
        if search_system_binaries:
            PATH: str = await self.getenv.asyn('PATH')

            for path in PATH.split(self.path.pathsep):
                try:
                    if name in (await self.list_directory.asyn(path)):
                        return self.path.join(path, name)
                except TargetStableError:
                    pass  # directory does not exist or no executable permissions
        return None

    which: '_AsyncPolymorphicFunction' = get_installed

    @asyn.asyncf
    async def install_if_needed(self, host_path: str, search_system_binaries: bool = True,
                                timeout: Optional[int] = None) -> str:
        """
        Check whether an executable with the name of ``host_path`` is already installed
        on the target. If it is not installed, install it from the specified path.

        :param host_path: The path to the executable on the host system.
        :param search_system_binaries: If ``True``, also search the device's system PATH
            for the binary before installing. If ``False``, only check user-installed
            binaries.
        :param timeout: Maximum time in seconds to wait for installation to complete.
            If ``None``, a default (implementation-defined) timeout is used.
        :return: The absolute path of the binary on the target after ensuring it is installed.

        :raises TargetError: If the target is disconnected.
        :raises TargetStableError: If installation fails or times out (depending on implementation).
        """
        binary_path: str = await self.get_installed.asyn(os.path.split(host_path)[1],
                                                         search_system_binaries=search_system_binaries)
        if not binary_path:
            binary_path = await self.install.asyn(host_path, timeout=timeout)
        return binary_path

    @asyn.asyncf
    async def is_installed(self, name: str) -> bool:
        """
        Determine whether an executable with the specified name is installed on the target.

        :param name: Name of the executable (e.g. "perf").
        :return: ``True`` if the executable is found, otherwise ``False``.

        :raises TargetError: If the target is not currently connected.
        """
        return bool(await self.get_installed.asyn(name))

    def bin(self, name: str) -> str:
        """
        Retrieve the installed path to the specified binary on the target.

        :param name: Name of the binary whose path is requested.
        :return: The path to the binary if installed and recorded by devlib,
            otherwise returns ``name`` unmodified.
        """
        return self._installed_binaries.get(name, name)

    def has(self, modname: str) -> bool:
        """
        Check whether the specified module or feature is present on the target.

        :param modname: Module name to look up.
        :return: ``True`` if the module is present and loadable, otherwise ``False``.

        :raises Exception: If an unexpected error occurs while querying the module.
        (Can be replaced with a more specific exception if desired.)
        """
        modname = identifier(modname)
        try:
            self._get_module(modname, log=False)
        except Exception:
            return False
        else:
            return True

    @asyn.asyncf
    async def lsmod(self) -> List['LsmodEntry']:
        """
        Run the ``lsmod`` command on the target and return the result as a list
        of :class:`LsmodEntry` namedtuples.

        :return: A list of loaded kernel modules, each represented by an LsmodEntry object.
        """
        lines: str = (await self.execute.asyn('lsmod')).splitlines()
        entries: List['LsmodEntry'] = []
        for line in lines[1:]:  # first line is the header
            if not line.strip():
                continue
            name, size, use_count, *remainder = line.split()
            if remainder:
                used_by = ''.join(remainder).split(',')
            else:
                used_by = []
            entries.append(LsmodEntry(name, size, use_count, used_by))
        return entries

    @asyn.asyncf
    async def insmod(self, path: str) -> None:
        """
        Insert a kernel module onto the target via ``insmod``.

        :param path: The path on the *host* system to the kernel module file (.ko).
        :raises TargetStableError: If the module cannot be inserted (e.g., missing dependencies).
        """
        target_path: Optional[str] = self.get_workpath(os.path.basename(path))
        await self.push.asyn(path, target_path)
        if target_path:
            await self.execute.asyn('insmod {}'.format(quote(target_path)), as_root=True)

    @asyn.asyncf
    async def extract(self, path: str, dest: Optional[str] = None) -> Optional[str]:
        """
        Extract the specified on-target file. The extraction method to be used
        (unzip, gunzip, bunzip2, or tar) will be based on the file's extension.
        If ``dest`` is specified, it must be an existing directory on target;
        the extracted contents will be placed there.

        Note that, depending on the archive file format (and therefore the
        extraction method used), the original archive file may or may not exist
        after the extraction.

        The return value is the path to the extracted contents.  In case of
        gunzip and bunzip2, this will be path to the extracted file; for tar
        and uzip, this will be the directory with the extracted file(s)
        (``dest`` if it was specified otherwise, the directory that contained
        the archive).

        :param path: The on-target path of the archive or compressed file.
        :param dest: An optional directory path on the target where the contents
            should be extracted. The directory must already exist.
        :return: Path to the extracted files.
            * If a multi-file archive, returns the directory containing those files.
            * If a single-file compression (e.g., .gz, .bz2), returns the path to
            the decompressed file.
            * If extraction fails or is unknown format, ``None`` might be returned
            (depending on your usage).

        :raises ValueError: If the file’s format is unrecognized.
        :raises TargetStableError: If extraction fails on the target.
        """
        for ending in ['.tar.gz', '.tar.bz', '.tar.bz2',
                       '.tgz', '.tbz', '.tbz2']:
            if path.endswith(ending):
                return await self._extract_archive(path, 'tar xf {} -C {}', dest)

        ext: str = self.path.splitext(path)[1]
        if ext in ['.bz', '.bz2']:
            return await self._extract_file(path, 'bunzip2 -f {}', dest)
        elif ext == '.gz':
            return await self._extract_file(path, 'gunzip -f {}', dest)
        elif ext == '.zip':
            return await self._extract_archive(path, 'unzip {} -d {}', dest)
        else:
            raise ValueError('Unknown compression format: {}'.format(ext))

    @asyn.asyncf
    async def sleep(self, duration: int) -> None:
        """
        Invoke a ``sleep`` command on the target to pause for the specified duration.

        :param duration: The time in seconds the target should sleep.
        :raises TimeoutError: If the sleep operation times out (rare, but can be forced).
        """
        timeout = duration + 10
        await self.execute.asyn('sleep {}'.format(duration), timeout=timeout)

    @asyn.asyncf
    async def read_tree_tar_flat(self, path: str, depth: int = 1, check_exit_code: bool = True,
                                 decode_unicode: bool = True, strip_null_chars: bool = True) -> Dict[str, str]:
        """
        Recursively read file nodes within a tar archive stored on the target, up to
        a given ``depth``. The archive is temporarily extracted in memory, and the
        contents are returned in a flat dictionary mapping each file path to its content.

        :param path: Path to the tar archive on the target.
        :param depth: Maximum directory depth to traverse within the archive.
        :param check_exit_code: If ``True``, raise an error if the helper command exits non-zero.
        :param decode_unicode: If ``True``, attempt to decode each file’s content as UTF-8.
        :param strip_null_chars: If ``True``, strip out any null characters (``\\x00``) from
            decoded text.
        :return: A dictionary mapping file paths (within the archive) to their textual content.

        :raises TargetStableError: If the helper command fails or returns unexpected data.
        :raises UnicodeDecodeError: If a file's content cannot be decoded when
            ``decode_unicode=True``.
        """
        self.async_manager.track_access(
            asyn.PathAccess(namespace='target', path=path, mode='r')
        )
        if path and self.working_directory:
            command = 'read_tree_tgz_b64 {} {} {}'.format(quote(path), depth,
                                                          quote(self.working_directory))
            output: str = await self._execute_util.asyn(command, as_root=self.is_rooted,
                                                        check_exit_code=check_exit_code)

        result: Dict[str, str] = {}

        # Unpack the archive in memory
        tar_gz = base64.b64decode(output)
        tar_gz_bytes = io.BytesIO(tar_gz)
        tar_buf = gzip.GzipFile(fileobj=tar_gz_bytes).read()
        tar_bytes = io.BytesIO(tar_buf)
        with tarfile.open(fileobj=tar_bytes) as tar:
            for member in tar.getmembers():
                try:
                    content_f = tar.extractfile(member)
                # ignore exotic members like sockets
                except Exception:
                    continue
                # if it is a file and not a folder
                if content_f:
                    content = content_f.read()
                    if decode_unicode:
                        try:
                            content_str = content.decode('utf-8').strip()
                            if strip_null_chars:
                                content_str = content_str.replace('\x00', '').strip()
                        except UnicodeDecodeError:
                            content_str = ''
                    name: str = self.path.join(path, member.name)
                    result[name] = content_str

        return result

    @asyn.asyncf
    async def read_tree_values_flat(self, path: str, depth: int = 1, check_exit_code: bool = True) -> Dict[str, str]:
        """
        Recursively read file nodes under a given directory (e.g., sysfs) on the target,
        up to the specified depth, returning a flat dictionary of file paths to contents.

        :param path: The on-target directory path to read from.
        :param depth: Maximum directory depth to traverse.
        :param check_exit_code: If ``True``, raises an error if the helper command fails.
        :return: A dict mapping each discovered file path to the file's textual content.

        :raises TargetStableError: If the read-tree helper command fails or no content is returned.
        """
        self.async_manager.track_access(
            asyn.PathAccess(namespace='target', path=path, mode='r')
        )
        command: str = 'read_tree_values {} {}'.format(quote(path), depth)
        output: str = await self._execute_util.asyn(command, as_root=self.is_rooted,
                                                    check_exit_code=check_exit_code)

        accumulator = defaultdict(list)
        for entry in output.strip().split('\n'):
            if ':' not in entry:
                continue
            path, value = entry.strip().split(':', 1)
            accumulator[path].append(value)

        result: Dict[str, str] = {k: '\n'.join(v).strip() for k, v in accumulator.items()}
        return result

    @asyn.asyncf
    async def read_tree_values(self, path: str, depth: int = 1, dictcls: Type[Dict] = dict,
                               check_exit_code: bool = True, tar: bool = False, decode_unicode: bool = True,
                               strip_null_chars: bool = True) -> Union[str, Dict[str, 'Node']]:
        """
        Recursively read all file nodes under a given directory or tar archive on the target,
        building a **tree-like** structure up to the given depth.

        :param path: On-target path to read. May be a directory path or a tar file path
            if ``tar=True``.
        :param depth: Maximum directory depth to traverse.
        :param dictcls: The dictionary class to use for constructing the tree
            (defaults to the built-in :class:`dict`).
        :param check_exit_code: If ``True``, raises an error if the internal helper command fails.
        :param tar: If ``True``, treat ``path`` as a tar archive and read it. If ``False``,
            read from a normal directory hierarchy.
        :param decode_unicode: If ``True``, decode file contents (in tar mode) as UTF-8.
        :param strip_null_chars: If ``True``, strip out any null characters (``\\x00``) from
            decoded text.
        :return: A hierarchical dictionary (or specialized mapping) containing sub-directories
            and files as nested keys, or a string in some edge cases (depending on usage).

        :raises TargetStableError: If the read-tree operation fails.
        :raises UnicodeDecodeError: If a file content cannot be decoded.
        """
        if not tar:
            value_map: Dict[str, str] = await self.read_tree_values_flat.asyn(path, depth, check_exit_code)
        else:
            value_map = await self.read_tree_tar_flat.asyn(path, depth, check_exit_code,
                                                           decode_unicode,
                                                           strip_null_chars)
        return _build_path_tree(value_map, path, self.path.sep, dictcls)

    def install_module(self, mod, **params):
        mod = get_module(mod)
        if mod.stage == 'early':
            raise TargetStableError(
                f'Module "{mod.name}" cannot be installed after device setup has already occoured'
            )
        else:
            return self._install_module(mod, params)

    # internal methods

    @asyn.asyncf
    async def _setup_scripts(self) -> None:
        """
        Install and prepare the ``shutils`` script on the target. This script provides
        shell utility functions that may be invoked by other devlib features.

        :raises TargetStableError:
            If ``busybox`` is not installed or if pushing/installing ``shutils`` fails.
        :raises IOError:
            If reading the local script file fails on the host system.
        """
        scripts = os.path.join(PACKAGE_BIN_DIRECTORY, 'scripts')
        shutils_ifile = os.path.join(scripts, 'shutils.in')
        with open(shutils_ifile) as fh:
            lines: List[str] = fh.readlines()
        with tempfile.TemporaryDirectory() as folder:
            shutils_ofile: str = os.path.join(folder, 'shutils')
            with open(shutils_ofile, 'w') as ofile:
                if self.busybox:
                    for line in lines:
                        line = line.replace("__DEVLIB_BUSYBOX__", self.busybox)
                        ofile.write(line)
            self._shutils = await self.install.asyn(shutils_ofile)

        await self.install.asyn(os.path.join(scripts, 'devlib-signal-target'))

    @asyn.asyncf
    @call_conn
    async def _execute_util(self, command: SubprocessCommand, timeout: Optional[int] = None,
                            check_exit_code: bool = True, as_root: bool = False) -> Optional[str]:
        """
        Execute a shell utility command via the ``shutils`` script on the target.
        This typically prepends the busybox and shutils script calls before your
        specified command.

        :param command: The command (or SubprocessCommand) string to run.
        :param timeout: Maximum number of seconds to allow for completion. If None,
            an implementation-defined default is used.
        :param check_exit_code: If True, raise an error when the return code is non-zero.
        :param as_root: If True, attempt to run with root privileges (e.g., ``su``
            or ``sudo``).
        :return: The command's output on success, or ``None`` if busybox/shutils is
            unavailable.

        :raises TargetStableError: If the script is not present or the command fails
            with a non-zero code (while ``check_exit_code=True``).
        :raises TimeoutError: If the command runs longer than the specified timeout.
        """
        if self.busybox and self.shutils:
            command_str = '{} sh {} {}'.format(quote(self.busybox), quote(self.shutils), cast(str, command))
            return await self.execute.asyn(
                command_str,
                timeout=timeout,
                check_exit_code=check_exit_code,
                as_root=as_root
            )
        return None

    async def _extract_archive(self, path: str, cmd: str, dest: Optional[str] = None) -> Optional[str]:
        """
        extract files of type -
        '.tar.gz', '.tar.bz', '.tar.bz2', '.tgz', '.tbz', '.tbz2'

        :param path: On-target path of the compressed archive (e.g., .tar.gz).
        :param cmd: A template string for the extraction command (e.g., 'tar xf {} -C {}').
        :param dest: Optional path to a destination directory on the target
            where files are extracted. If not specified, extraction occurs in
            the same directory as ``path``.
        :return: The directory or file path where the archive's contents were extracted,
            or None if ``busybox`` or other prerequisites are missing.

        :raises TargetStableError: If extraction fails or the file/directory cannot be written.
        """
        cmd = '{} ' + cmd  # busybox
        if dest:
            extracted: Optional[str] = dest
        else:
            extracted = self.path.dirname(path)
        if self.busybox and extracted:
            cmdtext = cmd.format(quote(self.busybox), quote(path), quote(extracted))
            await self.execute.asyn(cmdtext)
        return extracted

    async def _extract_file(self, path: str, cmd: str, dest: Optional[str] = None) -> Optional[str]:
        """
        Decompress a single file on the target (e.g., .gz, .bz2).

        :param path: On-target path of the compressed file.
        :param cmd: The decompression command format string (e.g., 'gunzip -f {}').
        :param dest: Optional directory path on the target where the decompressed file
            should be moved. If omitted, the file remains in its original directory
            (with the extension removed).
        :return: The path to the decompressed file after extraction, or None if
            prerequisites are missing.

        :raises TargetStableError: If decompression fails or the file/directory is unwritable.
        """
        cmd = '{} ' + cmd  # busybox
        if self.busybox and self.path:
            cmdtext: str = cmd.format(quote(self.busybox), quote(path))
            await self.execute.asyn(cmdtext)
            extracted: Optional[str] = self.path.splitext(path)[0]
            if dest and extracted:
                await self.execute.asyn('mv -f {} {}'.format(quote(extracted), quote(dest)))
                if dest.endswith('/'):
                    extracted = self.path.join(dest, self.path.basename(extracted))
                else:
                    extracted = dest
            return extracted
        return None

    def _install_module(self, mod: Union[str, Type[Module]],
                        params: Dict[str, Type[Module]], log: bool = True) -> Optional[Module]:
        """
        Installs a devlib module onto the target post-setup.

        :param mod: Either the module's name (string) or a Module type object.
        :param params: A dictionary of parameters for initializing the module.
        :param log: If True, logs errors if installation fails.
        :return: The instantiated Module object if installation succeeds, otherwise None.

        :raises TargetStableError: If the module has been explicitly disabled or if
            initialization fails irrecoverably.
        :raises Exception: If any other unexpected error occurs.
        """
        module = get_module(mod)
        name = module.name
        if name:
            if params is None or self._modules.get(name, {}) is None:
                raise TargetStableError(f'Could not load module "{name}" as it has been explicilty disabled')
            else:
                try:
                    return module.install(self, **params)  # type: ignore
                except Exception as e:
                    if log:
                        self.logger.error(f'Module "{name}" failed to install on target: {e}')
                    raise
        raise TargetStableError('Failed to install module as module name is not present')

    @property
    def modules(self) -> List[str]:
        """
        A list of module names registered on this target, regardless of which
        have been installed.

        :return: Sorted list of module names.
        """
        return sorted(self._modules.keys())

    def _update_modules(self, stage: str) -> None:
        """
        Load or install modules that match the specified stage (e.g., "early",
        "connected", or "setup").

        :param stage: The stage name used for grouping when modules should be installed.

        :raises Exception: If a module fails installation or is not supported
            by the target (caught and logged internally).
        """
        to_install: List[Tuple[Type[Module], Dict[str, Type[Module]]]] = [
            (mod, params)
            for mod, params in (
                (get_module(name), params)
                for name, params in self._modules.items()
            )
            if mod.stage == stage
        ]
        for mod, params in to_install:
            try:
                self._install_module(mod, params)
            except Exception as e:
                self.logger.warning(f'Module {mod.name} is not supported by the target: {e}')

    def _get_module(self, modname: str, log: bool = True) -> Module:
        """
        Retrieve or install a module by name. If not already installed, this
        attempts to install it first.

        :param modname: The name or attribute of the module to retrieve.
        :param log: If True, logs errors if installation fails.
        :return: The installed module object, if successful.
        :raises AttributeError: If the module or attribute cannot be found or installed.
        """
        try:
            return self._installed_modules[modname]
        except KeyError:
            params = {}
            try:
                mod = get_module(modname)
            # We might try to access e.g. "boot" attribute, which is ambiguous
            # since there are multiple modules with the "boot" kind. In that
            # case, we look into the list of modules enabled by the user and
            # get the first "boot" module we find.
            except ValueError:
                for _mod, _params in self._modules.items():
                    try:
                        _module = get_module(_mod)
                    except ValueError:
                        pass
                    else:
                        if _module.attr_name == modname:
                            mod = _module
                            params = _params
                            break
                else:
                    raise AttributeError(
                        f"'{self.__class__.__name__}' object has no attribute '{modname}'"
                    )
            else:
                if mod.name:
                    params = self._modules.get(mod.name, {})

            self._install_module(mod, params, log=log)
            return self.__getattr__(modname)

    def __getattr__(self, attr: str) -> Module:
        """
        Fallback attribute accessor, invoked if a normal attribute or method
        is not found. This checks for a corresponding installed or installable
        module whose name matches ``attr``.

        :param attr: The module name or attribute to fetch.
        :return: The installed module if found/installed, otherwise raises AttributeError.
        :raises AttributeError: If the module does not exist or cannot be installed.
        """
        # When unpickled, objects will have an empty dict so fail early
        if attr.startswith('__') and attr.endswith('__'):
            raise AttributeError(attr)

        try:
            return self._get_module(attr)
        except Exception as e:
            # Raising AttributeError is important otherwise hasattr() will not
            # work as expected
            raise AttributeError(str(e))

    def _resolve_paths(self) -> None:
        """
        Perform final path resolutions, such as setting the target's working directory,
        file transfer cache, or executables directory.

        :raises NotImplementedError: If the target subclass has not overridden this method.
        """
        raise NotImplementedError()

    @asyn.asyncf
    async def is_network_connected(self) -> bool:
        """
        Check if the target has basic network/internet connectivity by using
        ``ping`` to reach a known IP (e.g., 8.8.8.8).

        :return: True if the network appears to be reachable; False otherwise.

        :raises TargetStableError: If the network is known to be unreachable or if
            the shell command reports a fatal error.
        :raises TimeoutError: If repeatedly pinging does not respond within
            the default or user-defined time.
        """
        self.logger.debug('Checking for internet connectivity...')

        timeout_s = 5
        # It would be nice to use busybox for this, but that means we'd need
        # root (ping is usually setuid so it can open raw sockets to send ICMP)
        command = 'ping -q -c 1 -w {} {} 2>&1'.format(timeout_s,
                                                      quote(GOOGLE_DNS_SERVER_ADDRESS))

        # We'll use our own retrying mechanism (rather than just using ping's -c
        # to send multiple packets) so that we don't slow things down in the
        # 'good' case where the first packet gets echoed really quickly.
        attempts = 5
        for _ in range(attempts):
            try:
                await self.execute.asyn(command)
                return True
            except TargetStableError as e:
                err: str = str(e).lower()
                if '100% packet loss' in err:
                    # We sent a packet but got no response.
                    # Try again - we don't want this to fail just because of a
                    # transient drop in connection quality.
                    self.logger.debug('No ping response from {} after {}s'
                                      .format(GOOGLE_DNS_SERVER_ADDRESS, timeout_s))
                    continue
                elif 'network is unreachable' in err:
                    # No internet connection at all, we can fail straight away
                    self.logger.debug('Network unreachable')
                    return False
                else:
                    # Something else went wrong, we don't know what, raise an
                    # error.
                    raise

        self.logger.debug('Failed to ping {} after {} attempts'.format(
            GOOGLE_DNS_SERVER_ADDRESS, attempts))
        return False


class LinuxTarget(Target):
    """
    A specialized :class:`Target` subclass for devices or systems running Linux.
    Adapts path handling to ``posixpath`` and includes additional helpers for
    Linux-specific commands or filesystems.

    :ivar path: Set to ``posixpath``.
    :vartype path: ModuleType
    :ivar os: ``"linux"``
    :vartype os: str
    """

    path: ClassVar[ModuleType] = posixpath
    os = 'linux'

    @property
    @memoized
    def abi(self) -> str:
        """
        Determine the Application Binary Interface (ABI) of the device by
        interpreting the output of ``uname -m`` and mapping it to known
        architecture strings in ``ABI_MAP``.

        :return: The ABI string (e.g., "arm64" or "x86_64"). If unmapped,
            returns the exact output of ``uname -m``.
        """
        value: str = self.execute('uname -m').strip()
        for abi, architectures in ABI_MAP.items():
            if value in architectures:
                result = abi
                break
        else:
            result = value
        return result

    @property
    @memoized
    def os_version(self) -> Dict[str, str]:
        """
        Gather Linux distribution or version info by scanning files in ``/etc/``
        that end with ``-release`` or ``-version``.

        :return: A dictionary mapping the filename (e.g. "os-release") to
            its contents as a single line.
        """
        os_version: Dict[str, str] = {}
        command = 'ls /etc/*-release /etc*-version /etc/*_release /etc/*_version 2>/dev/null'
        version_files: List[str] = self.execute(command, check_exit_code=False).strip().split()
        for vf in version_files:
            name: str = self.path.basename(vf)
            output: str = self.read_value(vf)
            os_version[name] = convert_new_lines(output.strip()).replace('\n', ' ')
        return os_version

    @property
    @memoized
    def system_id(self) -> str:
        """
        Retrieve a Linux-specific system ID by invoking
        a specialized utility command on the target.

        :return: A string uniquely identifying the Linux system.
        """
        return self._execute_util('get_linux_system_id').strip()

    def __init__(self,
                 connection_settings: Optional[UserConnectionSettings] = None,
                 platform: Optional[Platform] = None,
                 working_directory: Optional[str] = None,
                 executables_directory: Optional[str] = None,
                 connect: bool = True,
                 modules: Optional[Dict[str, Dict[str, Type[Module]]]] = None,
                 load_default_modules: bool = True,
                 shell_prompt: Pattern[str] = DEFAULT_SHELL_PROMPT,
                 conn_cls: 'InitCheckpointMeta' = SshConnection,
                 is_container: bool = False,
                 max_async: int = 50,
                 tmp_directory: Optional[str] = None,
                 ):
        super(LinuxTarget, self).__init__(connection_settings=connection_settings,
                                          platform=platform,
                                          working_directory=working_directory,
                                          executables_directory=executables_directory,
                                          connect=connect,
                                          modules=modules,
                                          load_default_modules=load_default_modules,
                                          shell_prompt=shell_prompt,
                                          conn_cls=conn_cls,
                                          is_container=is_container,
                                          max_async=max_async,
                                          tmp_directory=tmp_directory,
                                          )

    def wait_boot_complete(self, timeout: Optional[int] = 10) -> None:
        """
        wait for target to boot up
        """
        pass

    @asyn.asyncf
    async def get_pids_of(self, process_name) -> List[int]:
        """
        Return a list of PIDs of all running processes matching the given name.

        :param process_name: Name of the process to look up.
        :return: List of matching PIDs.
        :raises NotImplementedError: If not overridden by child classes.
        """
        # result should be a column of PIDs with the first row as "PID" header
        result_temp:str = await self.execute.asyn('ps -C {} -o pid'.format(quote(process_name)),  # NOQA
                                                  check_exit_code=False)
        result: List[str] = result_temp.strip().split()
        if len(result) >= 2:  # at least one row besides the header
            return list(map(int, result[1:]))
        else:
            return []

    @asyn.asyncf
    async def ps(self, threads: bool = False, **kwargs: Dict[str, Any]) -> List['PsEntry']:
        """
        Return a list of PsEntry objects for each process on the system.

        :return: A list of processes.
        :raises NotImplementedError: If not overridden.
        """
        ps_flags = '-eo'
        if threads:
            ps_flags = '-eLo'
        command = 'ps {} user,pid,tid,ppid,vsize,rss,wchan,pcpu,state,fname'.format(ps_flags)

        out = await self.execute.asyn(command)

        result: List['PsEntry'] = []
        lines: List[str] = convert_new_lines(out).splitlines()
        # Skip header
        for line in lines[1:]:
            parts: List[str] = re.split(r'\s+', line, maxsplit=9)
            if parts:
                result.append(PsEntry(*(parts[0:1] + list(map(int, parts[1:6])) + parts[6:])))

        if not kwargs:
            return result
        else:
            filtered_result: List['PsEntry'] = []
            for entry in result:
                if all(getattr(entry, k) == v for k, v in kwargs.items()):
                    filtered_result.append(entry)
            return filtered_result

    async def _list_directory(self, path: str, as_root: bool = False) -> List[str]:
        """
        target specific implementation of list_directory
        """
        contents = await self.execute.asyn('ls -1 {}'.format(quote(path)), as_root=as_root)
        return [x.strip() for x in contents.split('\n') if x.strip()]

    @asyn.asyncf
    async def install(self, filepath: str, timeout: Optional[int] = None,
                      with_name: Optional[str] = None) -> str:  # pylint: disable=W0221
        """
        Install an executable on the device.

        :param filepath: path to the executable on the host
        :param timeout: Optional timeout (in seconds) for the installation
        :param with_name: This may be used to rename the executable on the target
        """
        destpath: str = self.path.join(self.executables_directory,
                                       with_name and with_name or self.path.basename(filepath))
        await self.push.asyn(filepath, destpath, timeout=timeout)
        await self.execute.asyn('chmod a+x {}'.format(quote(destpath)), timeout=timeout)
        self._installed_binaries[self.path.basename(destpath)] = destpath
        return destpath

    @asyn.asyncf
    async def uninstall(self, name: str) -> None:
        """
        Uninstall the specified executable from the target
        """
        path: str = self.path.join(self.executables_directory, name)
        await self.remove.asyn(path)

    @asyn.asyncf
    async def capture_screen(self, filepath: str) -> None:
        """
        Take a screenshot on the device and save it to the specified file on the
        host. This may not be supported by the target. You can optionally insert a
        ``{ts}`` tag into the file name, in which case it will be substituted with
        on-target timestamp of the screen shot in ISO8601 format.
        """
        if not (await self.is_installed.asyn('scrot')):
            self.logger.debug('Could not take screenshot as scrot is not installed.')
            return
        try:

            async with self.make_temp(is_directory=False) as tmpfile:
                cmd = 'DISPLAY=:0.0 scrot {} && {} date -u -Iseconds'
                if self.busybox:
                    ts: str = (await self.execute.asyn(cmd.format(quote(tmpfile), quote(self.busybox)))).strip()
                    filepath = filepath.format(ts=ts)
                    await self.pull.asyn(tmpfile, filepath)
                else:
                    raise TargetStableError("busybox is not present")
        except TargetStableError as e:
            if isinstance(e.message, str) and "Can't open X dispay." not in e.message:
                raise e
            if isinstance(e.message, str):
                message = e.message.split('OUTPUT:', 1)[1].strip()  # pylint: disable=no-member
                self.logger.debug('Could not take screenshot: {}'.format(message))

    def _resolve_paths(self) -> None:
        """
        set paths for working directory, file transfer cache and executables directory
        """
        if self.working_directory is None:
            # This usually lands in the home directory
            self.working_directory = self.path.join(self.execute("pwd").strip(), 'devlib-target')


class AndroidTarget(Target):
    """
    A specialized :class:`Target` subclass for devices running Android. This
    provides additional Android-specific features like property retrieval
    (``getprop``), APK installation, ADB connection management, screen controls,
    input injection, and more.

    :param connection_settings: Parameters for connecting to the device
        (e.g., ADB serial or host/port).
    :param platform: A ``Platform`` object describing hardware aspects. If None,
        a generic or default platform is used.
    :param working_directory: A directory on the device for devlib to store
        temporary files. Defaults to a subfolder of external storage.
    :param executables_directory: A directory on the device where devlib
        installs binaries. Defaults to ``/data/local/tmp/bin``.
    :param connect: If True, automatically connect to the device upon instantiation.
        Otherwise, call :meth:`connect`.
    :param modules: Additional modules to load (name -> parameters).
    :param load_default_modules: If True, load all modules in :attr:`default_modules`.
    :param shell_prompt: Regex matching the interactive shell prompt, if used.
    :param conn_cls: The connection class, typically :class:`AdbConnection`.
    :param package_data_directory: Location where installed packages store data.
        Defaults to ``"/data/data"``.
    :param is_container: If True, indicates the device is actually a container environment.
    :param max_async: Maximum number of asynchronous operations to allow in parallel.
    """
    path: ClassVar[ModuleType] = posixpath
    os = 'android'
    ls_command = ''

    @property
    @memoized
    def abi(self) -> str:
        """
        Return the main ABI (CPU architecture) by reading ``ro.product.cpu.abi``
        from the device properties.

        :return: E.g. "arm64" or "armeabi-v7a" for an Android device.
        """
        return self.getprop()['ro.product.cpu.abi'].split('-')[0]

    @property
    @memoized
    def supported_abi(self) -> List[str]:
        """
        List all supported ABIs found in Android system properties. Combines
        values from ``ro.product.cpu.abi``, ``ro.product.cpu.abi2``,
        and ``ro.product.cpu.abilist``.

        :return: A list of ABI strings (some might be mapped to devlib’s known
            architecture list).
        """
        props: Dict[str, str] = self.getprop()
        result: List[str] = [props['ro.product.cpu.abi']]
        if 'ro.product.cpu.abi2' in props:
            result.append(props['ro.product.cpu.abi2'])
        if 'ro.product.cpu.abilist' in props:
            for abi in props['ro.product.cpu.abilist'].split(','):
                if abi not in result:
                    result.append(abi)

        mapped_result: List[str] = []
        for supported_abi in result:
            for abi, architectures in ABI_MAP.items():
                found = False
                if supported_abi in architectures and abi not in mapped_result:
                    mapped_result.append(abi)
                    found = True
                    break
            if not found and supported_abi not in mapped_result:
                mapped_result.append(supported_abi)
        return mapped_result

    @property
    @memoized
    def os_version(self) -> Dict[str, str]:
        """
        Read and parse Android build version info from properties whose keys
        start with ``ro.build.version``.

        :return: Dictionary mapping the last component of each key
            (e.g., "incremental", "release") to its string value.
        """
        os_version: Dict[str, str] = {}
        for k, v in self.getprop().iteritems():
            if k.startswith('ro.build.version'):
                part: str = k.split('.')[-1]
                os_version[part] = v
        return os_version

    @property
    def adb_name(self) -> Optional[str]:
        """
        The ADB device name or serial number for the connected Android device.

        :return:
            - The string serial/ID if connected via ADB (e.g. ``"0123456789ABCDEF"``).
            - ``None`` if unavailable or a different connection type is used (e.g. SSH).
        """
        return getattr(self.conn, 'device', None)

    @property
    def adb_server(self) -> Optional[str]:
        """
        The hostname or IP address of the ADB server, if using a remote ADB
        connection.

        :return:
            - The ADB server address (e.g. ``"127.0.0.1"``).
            - ``None`` if not applicable (local ADB or a non-ADB connection).
        """
        return getattr(self.conn, 'adb_server', None)

    @property
    def adb_port(self) -> Optional[int]:
        """
        The TCP port on which the ADB server is listening, if using a remote ADB
        connection.

        :return:
            - An integer port number (e.g. 5037).
            - ``None`` if not applicable or unknown.
        """
        return getattr(self.conn, 'adb_port', None)

    @property
    @memoized
    def android_id(self) -> str:
        """
        Get the device's ANDROID_ID. Which is

            "A 64-bit number (as a hex string) that is randomly generated when the user
            first sets up the device and should remain constant for the lifetime of the
            user's device."

        .. note:: This will get reset on userdata erasure.

        :return: The ANDROID_ID in hexadecimal form.

        """
        # FIXME - would it be better to just do 'settings get secure android_id' ? when trying to execute the content command,
        # getting some access issues with settings
        output = self.execute('content query --uri content://settings/secure --projection value --where "name=\'android_id\'"').strip()
        return output.split('value=')[-1]

    @property
    @memoized
    def system_id(self) -> str:
        """
        Obtain a unique Android system identifier by using a device utility
        (e.g., 'get_android_system_id' in shutils).

        :return: A device-specific ID string.
        """
        return self._execute_util('get_android_system_id').strip()

    @property
    @memoized
    def external_storage(self) -> str:
        """
        The path to the device's external storage directory (often ``/sdcard`` or
        ``/storage/emulated/0``).

        :return:
            A filesystem path pointing to the shared/SD card area on the Android device.
        :raises TargetStableError:
            If the environment variable ``EXTERNAL_STORAGE`` is unset or an error
            occurs reading it.
        """
        return self.execute('echo $EXTERNAL_STORAGE').strip()

    @property
    @memoized
    def external_storage_app_dir(self) -> Optional[str]:
        """
        The application-specific directory within external storage
        (commonly ``/sdcard/Android/data``).

        :return:
            The path to the app-specific directory under external storage, or
            ``None`` if not determinable (e.g. no external storage).
        """
        return self.path.join(self.external_storage, 'Android', 'data')

    @property
    @memoized
    def screen_resolution(self) -> Tuple[int, int]:
        """
        The current display resolution (width, height), read from ``dumpsys window displays``.

        :return:
            A tuple ``(width, height)`` of the device’s screen resolution in pixels.

        :raises TargetStableError:
            If the resolution cannot be parsed from ``dumpsys`` output.
        """
        output: str = self.execute('dumpsys window displays')
        match = ANDROID_SCREEN_RESOLUTION_REGEX.search(output)
        if match:
            return (int(match.group('width')),
                    int(match.group('height')))
        else:
            return (0, 0)

    def __init__(self,
                 connection_settings: Optional[UserConnectionSettings] = None,
                 platform: Optional[Platform] = None,
                 working_directory: Optional[str] = None,
                 executables_directory: Optional[str] = None,
                 connect: bool = True,
                 modules: Optional[Dict[str, Dict[str, Type[Module]]]] = None,
                 load_default_modules: bool = True,
                 shell_prompt: Pattern[str] = DEFAULT_SHELL_PROMPT,
                 conn_cls: 'InitCheckpointMeta' = AdbConnection,
                 package_data_directory: str = "/data/data",
                 is_container: bool = False,
                 max_async: int = 50,
                 tmp_directory: Optional[str] = None,
                 ):
        """
        Initialize an AndroidTarget instance and optionally connect to the
        device via ADB.
        """
        super(AndroidTarget, self).__init__(connection_settings=connection_settings,
                                            platform=platform,
                                            working_directory=working_directory,
                                            executables_directory=executables_directory,
                                            connect=connect,
                                            modules=modules,
                                            load_default_modules=load_default_modules,
                                            shell_prompt=shell_prompt,
                                            conn_cls=conn_cls,
                                            is_container=is_container,
                                            max_async=max_async,
                                            tmp_directory=tmp_directory,
                                            )
        self.package_data_directory = package_data_directory
        self._init_logcat_lock()

    def _init_logcat_lock(self) -> None:
        """
        Initialize a lock used for serializing logcat clearing operations.
        This prevents overlapping ``logcat -c`` calls from multiple threads.
        """
        self.clear_logcat_lock = threading.Lock()

    def __getstate__(self) -> Dict[str, Any]:
        """
        Extend the base pickling to skip the `clear_logcat_lock`.
        """
        dct = super().__getstate__()
        return {
            k: v
            for k, v in dct.items()
            if k not in ('clear_logcat_lock',)
        }

    def __setstate__(self, dct: Dict[str, Any]) -> None:
        """
        Restore post-pickle state, reinitializing the logcat lock.
        """
        super().__setstate__(dct)
        self._init_logcat_lock()

    @asyn.asyncf
    async def reset(self, fastboot=False):  # pylint: disable=arguments-differ
        """
        Soft reset (reboot) the device. If ``fastboot=True``, attempt to reboot
        into fastboot mode.

        :param fastboot: If True, reboot into fastboot instead of normal reboot.
        :raises DevlibTransientError: If "reboot" command fails or times out.
        """
        try:
            await self.execute.asyn('reboot {}'.format(fastboot and 'fastboot' or ''),
                                    as_root=self.needs_su, timeout=2)
        except (DevlibTransientError, subprocess.CalledProcessError):
            # on some targets "reboot" doesn't return gracefully
            pass
        self.conn.connected_as_root = None

    @asyn.asyncf
    async def wait_boot_complete(self, timeout: Optional[int] = 10) -> None:
        """
        Wait for Android to finish booting, typically by polling ``sys.boot_completed``
        property.

        :param timeout: Seconds to wait. If the property isn't set by this time, raise.
        :raises TargetStableError: If the device remains un-booted after `timeout` seconds.
        """
        start: float = time.time()
        boot_completed: bool = boolean(await self.getprop.asyn('sys.boot_completed'))
        if timeout:
            while not boot_completed and timeout >= time.time() - start:
                time.sleep(5)
                boot_completed = boolean(await self.getprop.asyn('sys.boot_completed'))
            if not boot_completed:
                # Raise a TargetStableError as this usually happens because of
                # an issue with Android more than a timeout that is too small.
                raise TargetStableError('Connected but Android did not fully boot.')

    @asyn.asyncf
    async def connect(self, timeout: Optional[int] = 30,
                      check_boot_completed: Optional[bool] = True,
                      max_async: Optional[int] = None) -> None:  # pylint: disable=arguments-differ
        """
        Establish a connection to the target. It is usually not necessary to call
        this explicitly, as a connection gets automatically established on
        instantiation.

        :param timeout: Time in seconds before giving up on connection attempts.
        :param check_boot_completed: Whether to call :meth:`wait_boot_complete`.
        :param max_async: Override the default concurrency limit if provided.
        :raises TargetError: If the device fails to connect.
        """
        await super(AndroidTarget, self).connect.asyn(
            timeout=timeout,
            check_boot_completed=check_boot_completed,
            max_async=max_async,
        )

    @asyn.asyncf
    async def __setup_list_directory(self) -> None:
        """
        One-time setup to determine if the device supports ``ls -1``. On older
        Android versions, the ``-1`` flag might not be available, so fallback
        to plain ``ls``.
        """
        # In at least Linaro Android 16.09 (which was their first Android 7 release) and maybe
        # AOSP 7.0 as well, the ls command was changed.
        # Previous versions default to a single column listing, which is nice and easy to parse.
        # Newer versions default to a multi-column listing, which is not, but it does support
        # a '-1' option to get into single column mode. Older versions do not support this option
        # so we try the new version, and if it fails we use the old version.
        self.ls_command = 'ls -1'
        try:
            await self.execute.asyn('ls -1 {}'.format(quote(self.working_directory or '')), as_root=False)
        except TargetStableError:
            self.ls_command = 'ls'

    async def _list_directory(self, path: str, as_root: bool = False) -> List[str]:
        """
        Implementation of :meth:`list_directory` for Android. Uses an ls command
        that might be adjusted depending on OS version.

        :param path: Directory path on the device.
        :param as_root: If True, escalate privileges for listing.
        :return: A list of file/directory names in the specified path.
        :raises TargetStableError: If the directory doesn't exist or can't be listed.
        """
        if self.ls_command == '':
            await self.__setup_list_directory.asyn()
        contents = await self.execute.asyn('{} {}'.format(self.ls_command, quote(path)), as_root=as_root)
        return [x.strip() for x in contents.split('\n') if x.strip()]

    @asyn.asyncf
    async def install(self, filepath: str, timeout: Optional[int] = None,
                      with_name: Optional[str] = None) -> str:  # pylint: disable=W0221
        """
        Install a file (APK or binary) onto the Android device. If the file is an APK,
        use :meth:`install_apk`; otherwise, use :meth:`install_executable`.

        :param filepath: Path on the host to the file (APK or binary).
        :param timeout: Optional time in seconds to allow the install.
        :param with_name: If installing a binary, rename it on the device. Ignored for APKs.
        :return: The path or package installed on the device.
        :raises TargetStableError: If the file extension is unsupported or installation fails.
        """
        ext: str = os.path.splitext(filepath)[1].lower()
        if ext == '.apk':
            return await self.install_apk.asyn(filepath, timeout)
        else:
            return await self.install_executable.asyn(filepath, with_name, timeout)

    @asyn.asyncf
    async def uninstall(self, name: str) -> None:
        """
        Uninstall either a package (if installed as an APK) or an executable from
        the device.

        :param name: The package name or binary name to remove.
        """
        if await self.package_is_installed.asyn(name):
            await self.uninstall_package.asyn(name)
        else:
            await self.uninstall_executable.asyn(name)

    @asyn.asyncf
    async def get_pids_of(self, process_name: str) -> List[int]:
        """
        Return a list of process IDs (PIDs) for any processes matching ``process_name``.

        :param process_name: The substring or name to search for in the command name.
        :return: List of integer PIDs matching the name.
        """
        result: List[int] = []
        search_term = process_name[-15:]
        for entry in await self.ps.asyn():
            if search_term in entry.name:
                result.append(entry.pid)
        return result

    @asyn.asyncf
    async def ps(self, threads: bool = False, **kwargs: Dict[str, Any]) -> List['PsEntry']:
        """
        Return a list of process entries on the device (like ``ps`` output),
        optionally including thread info if ``threads=True``.

        :param threads: If True, use ``ps -AT`` to include threads.
        :param kwargs: Key/value filters to match against the returned attributes
            (like user, name, etc.).
        :return: A list of PsEntry objects matching the filter.
        :raises TargetStableError: If the command fails or ps output is malformed.
        """
        maxsplit = 9 if threads else 8
        command = 'ps'
        if threads:
            command = 'ps -AT'

        lines = iter(convert_new_lines(await self.execute.asyn(command)).split('\n'))
        next(lines)  # header
        result: List['PsEntry'] = []
        for line in lines:
            parts: List[str] = line.split(None, maxsplit)
            if not parts:
                continue

            wchan_missing: bool = False
            if len(parts) == maxsplit:
                wchan_missing = True

            if not threads:
                # Duplicate PID into TID location.
                parts.insert(2, parts[1])

            if wchan_missing:
                # wchan was blank; insert an empty field where it should be.
                parts.insert(6, '')
            result.append(PsEntry(*(parts[0:1] + list(map(int, parts[1:6])) + parts[6:])))
        if not kwargs:
            return result
        else:
            filtered_result: List['PsEntry'] = []
            for entry in result:
                if all(getattr(entry, k) == v for k, v in kwargs.items()):
                    filtered_result.append(entry)
            return filtered_result

    @asyn.asyncf
    async def capture_screen(self, filepath: str) -> None:
        """
        Take a screenshot on the device and save it to the specified file on the
        host. This may not be supported by the target. You can optionally insert a
        ``{ts}`` tag into the file name, in which case it will be substituted with
        on-target timestamp of the screen shot in ISO8601 format.

        :param filepath: The host file path to store the screenshot. E.g.
            ``"my_screenshot_{ts}.png"``
        :raises TargetStableError: If the device lacks a necessary screenshot tool (e.g. screencap).
        """
        if self.working_directory:
            on_device_file: str = self.path.join(self.working_directory, 'screen_capture.png')
            cmd = 'screencap -p  {} && {} date -u -Iseconds'
            if self.busybox:
                ts = (await self.execute.asyn(cmd.format(quote(on_device_file), quote(self.busybox)))).strip()
                filepath = filepath.format(ts=ts)
                await self.pull.asyn(on_device_file, filepath)
                await self.remove.asyn(on_device_file)

    # Android-specific

    @asyn.asyncf
    async def input_tap(self, x: int, y: int) -> None:
        """
        Simulate a tap/click event at (x, y) on the device screen.

        :param x: The horizontal coordinate (pixels).
        :param y: The vertical coordinate (pixels).
        :raises TargetStableError: If the ``input`` command is not found or fails.
        """
        command = 'input tap {} {}'
        await self.execute.asyn(command.format(x, y))

    @asyn.asyncf
    async def input_tap_pct(self, x: int, y: int):
        """
        Simulate a tap event using percentage-based coordinates, relative
        to the device screen size.

        :param x: Horizontal position as a percentage of screen width (0 to 100).
        :param y: Vertical position as a percentage of screen height (0 to 100).
        """
        width, height = self.screen_resolution

        x = (x * width) // 100
        y = (y * height) // 100

        await self.input_tap.asyn(x, y)

    @asyn.asyncf
    async def input_swipe(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """
        Issue a swipe gesture from (x1, y1) to (x2, y2), using absolute pixel coordinates.

        :param x1: Start X coordinate in pixels.
        :param y1: Start Y coordinate in pixels.
        :param x2: End X coordinate in pixels.
        :param y2: End Y coordinate in pixels.
        """
        command = 'input swipe {} {} {} {}'
        await self.execute.asyn(command.format(x1, y1, x2, y2))

    @asyn.asyncf
    async def input_swipe_pct(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """
        Issue a swipe gesture from (x1, y1) to (x2, y2) using percentage-based coordinates.

        :param x1: Horizontal start percentage (0-100).
        :param y1: Vertical start percentage (0-100).
        :param x2: Horizontal end percentage (0-100).
        :param y2: Vertical end percentage (0-100).
        """
        width, height = self.screen_resolution

        x1 = (x1 * width) // 100
        y1 = (y1 * height) // 100
        x2 = (x2 * width) // 100
        y2 = (y2 * height) // 100

        await self.input_swipe.asyn(x1, y1, x2, y2)

    @asyn.asyncf
    async def swipe_to_unlock(self, direction: str = "diagonal") -> None:
        """
        Attempt to swipe the lock screen open. Common directions are ``"horizontal"``,
        ``"vertical"``, or ``"diagonal"``.

        :param direction: The direction to swipe; defaults to diagonal for maximum coverage.
        :raises TargetStableError: If the direction is invalid or the swipe fails.
        """
        width, height = self.screen_resolution
        if direction == "diagonal":
            start = 100
            stop = width - start
            swipe_height = height * 2 // 3
            await self.input_swipe.asyn(start, swipe_height, stop, 0)
        elif direction == "horizontal":
            swipe_height = height * 2 // 3
            start = 100
            stop = width - start
            await self.input_swipe.asyn(start, swipe_height, stop, swipe_height)
        elif direction == "vertical":
            swipe_middle = width // 2
            swipe_height = height * 2 // 3
            await self.input_swipe.asyn(swipe_middle, swipe_height, swipe_middle, 0)
        else:
            raise TargetStableError("Invalid swipe direction: {}".format(direction))

    @asyn.asyncf
    async def getprop(self, prop: Optional[str] = None) -> Optional[Union[str, AndroidProperties]]:
        """
        Fetch properties from Android's ``getprop``. If ``prop`` is given,
        return just that property's value; otherwise return a dictionary-like
        :class:`AndroidProperties`.

        :param prop: A specific property key to retrieve (e.g. "ro.build.version.sdk").
        :return:
            - If ``prop`` is None, a dictionary-like object mapping all property keys to values.
            - If ``prop`` is non-empty, the string value of that specific property.
        """
        props = AndroidProperties(await self.execute.asyn('getprop'))
        if prop:
            return props[prop]
        return props

    @asyn.asyncf
    async def capture_ui_hierarchy(self, filepath: str) -> None:
        """
        Capture the current UI hierarchy via ``uiautomator dump``, pull it to
        the host, and optionally format it with pretty XML.

        :param filepath: The host file path to save the UI hierarchy XML.
        :raises TargetStableError: If the device cannot produce a dump or fails to store it.
        """
        on_target_file = self.get_workpath('screen_capture.xml')
        try:
            await self.execute.asyn('uiautomator dump {}'.format(on_target_file))
            await self.pull.asyn(on_target_file, filepath)
        finally:
            await self.remove.asyn(on_target_file)

        parsed_xml: Document = xml.dom.minidom.parse(filepath)
        with open(filepath, 'w') as f:
            f.write(parsed_xml.toprettyxml())

    @asyn.asyncf
    async def is_installed(self, name: str) -> bool:
        """
        Returns ``True`` if an executable with the specified name is installed on the
        target and ``False`` other wise.
        """
        return (await super(AndroidTarget, self).is_installed.asyn(name)) or (await self.package_is_installed.asyn(name))

    @asyn.asyncf
    async def package_is_installed(self, package_name: str) -> bool:
        """
        Check if the given package name is installed on the device.

        :param package_name: Name of the Android package (e.g. "com.example.myapp").
        :return: True if installed, False otherwise.
        """
        return package_name in (await self.list_packages.asyn())

    @asyn.asyncf
    async def list_packages(self) -> List[str]:
        """
        Return a list of installed package names on the device (via ``pm list packages``).

        :return: A list of package identifiers.
        """
        output: str = await self.execute.asyn('pm list packages')
        output = output.replace('package:', '')
        return output.split()

    @asyn.asyncf
    async def get_package_version(self, package: str) -> Optional[str]:
        """
        Obtain the versionName for a given package by parsing ``dumpsys package``.

        :param package: The package name (e.g. "com.example.myapp").
        :return: The versionName string if found, otherwise None.
        """
        output = await self.execute.asyn('dumpsys package {}'.format(quote(package)))
        for line in convert_new_lines(output).split('\n'):
            if 'versionName' in line:
                return line.split('=', 1)[1]
        return None

    @asyn.asyncf
    async def get_package_info(self, package: str) -> Optional['installed_package_info']:
        """
        Return a tuple (apk_path, package_name) for the installed package, or None if not found.

        :param package: The package identifier (e.g. "com.example.myapp").
        :return: A namedtuple with fields (apk_path, package), or None.
        """
        output: str = await self.execute.asyn('pm list packages -f {}'.format(quote(package)))
        for entry in output.strip().split('\n'):
            rest, entry_package = entry.rsplit('=', 1)
            if entry_package != package:
                continue
            _, apk_path = rest.split(':')
            return installed_package_info(apk_path, entry_package)
        return None

    @asyn.asyncf
    async def get_sdk_version(self) -> Optional[int]:
        """
        Return the integer value of ``ro.build.version.sdk`` if parseable; None if not.

        :return: e.g. 29 for Android 10, or None on error.
        """
        try:
            return int(await self.getprop.asyn('ro.build.version.sdk'))
        except (ValueError, TypeError):
            return None

    @asyn.asyncf
    async def install_apk(self, filepath: str, timeout: Optional[int] = None, replace: Optional[bool] = False,
                          allow_downgrade: Optional[bool] = False) -> Optional[str]:  # pylint: disable=W0221
        """
        Install an APK onto the device. If the device is connected via AdbConnection,
        use an ADB install command. Otherwise, push it and run 'pm install'.

        :param filepath: The path to the APK on the host.
        :param timeout: The time in seconds to wait for installation.
        :param replace: If True, pass -r to 'pm install' or `adb install`.
        :param allow_downgrade: If True, allow installing an older version over a newer one.
        :return: The output from the install command, or None if something unexpected occurs.
        :raises TargetStableError: If the file is not an APK or installation fails.
        """
        ext: str = os.path.splitext(filepath)[1].lower()
        if ext == '.apk':
            flags: List[str] = []
            if replace:
                flags.append('-r')  # Replace existing APK
            if allow_downgrade:
                flags.append('-d')  # Install the APK even if a newer version is already installed
            if self.get_sdk_version() >= 23:
                flags.append('-g')  # Grant all runtime permissions
            self.logger.debug("Replace APK = {}, ADB flags = '{}'".format(replace, ' '.join(flags)))
            if isinstance(self.conn, AdbConnection):
                return adb_command(self.adb_name,
                                   "install {} {}".format(' '.join(flags), quote(filepath)),
                                   timeout=timeout, adb_server=self.adb_server,
                                   adb_port=self.adb_port)
            else:
                dev_path: Optional[str] = self.get_workpath(filepath.rsplit(os.path.sep, 1)[-1])
                await self.push.asyn(quote(filepath), dev_path, timeout=timeout)
                if dev_path:
                    result: str = await self.execute.asyn("pm install {} {}".format(' '.join(flags), quote(dev_path)), timeout=timeout)
                    await self.remove.asyn(dev_path)
                    return result
                else:
                    raise TargetStableError('Can\'t install. could not get dev path')
        else:
            raise TargetStableError('Can\'t install {}: unsupported format.'.format(filepath))

    @asyn.asyncf
    async def grant_package_permission(self, package: str, permission: str) -> None:
        """
        Run `pm grant <package> <permission>`. Ignores some errors if the permission
        cannot be granted. This is typically used for runtime permissions on modern Android.

        :param package: The target package.
        :param permission: The permission string to grant (e.g. "android.permission.READ_LOGS").
        :raises TargetStableError: If some unexpected error occurs that is not a known ignorable case.
        """
        try:
            return await self.execute.asyn('pm grant {} {}'.format(quote(package), quote(permission)))
        except TargetStableError as e:
            if isinstance(e.message, str):
                if 'is not a changeable permission type' in e.message:
                    pass  # Ignore if unchangeable
                elif 'Unknown permission' in e.message:
                    pass  # Ignore if unknown
                elif 'has not requested permission' in e.message:
                    pass  # Ignore if not requested
                elif 'Operation not allowed' in e.message:
                    pass  # Ignore if not allowed
                elif 'is managed by role' in e.message:
                    pass  # Ignore if cannot be granted
                else:
                    raise
            else:
                raise

    @asyn.asyncf
    async def refresh_files(self, file_list: List[str]) -> None:
        """
        Attempt to force a re-index of the device media scanner for the given files.
        On newer Android (7+), if not rooted, we fallback to scanning each file individually.

        :param file_list: A list of file paths on the device that may need indexing (e.g. new media).
        """
        if (self.is_rooted or (await self.get_sdk_version.asyn()) < 24):  # MM and below
            common_path: str = commonprefix(file_list, sep=self.path.sep)
            await self.broadcast_media_mounted.asyn(common_path, self.is_rooted)
        else:
            for f in file_list:
                await self.broadcast_media_scan_file.asyn(f)

    @asyn.asyncf
    async def broadcast_media_scan_file(self, filepath: str) -> None:
        """
        Send a broadcast intent to the Android media scanner for a single file path.

        :param filepath: File path on the device to be scanned by mediaserver.
        """
        command = 'am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d {}'
        await self.execute.asyn(command.format(quote('file://' + filepath)))

    @asyn.asyncf
    async def broadcast_media_mounted(self, dirpath: str, as_root: bool = False) -> None:
        """
        Broadcast that media at a directory path is newly mounted, prompting scanning
        of its contents.

        :param dirpath: Directory path on the device.
        :param as_root: If True, escalate privileges for the broadcast command.
        """
        command = 'am broadcast -a  android.intent.action.MEDIA_MOUNTED -d {} '\
                  '-n com.android.providers.media/.MediaScannerReceiver'
        await self.execute.asyn(command.format(quote('file://' + dirpath)), as_root=as_root)

    @asyn.asyncf
    async def install_executable(self, filepath: str, with_name: Optional[str] = None,
                                 timeout: Optional[int] = None) -> str:
        """
        Install a single executable (non-APK) onto the device. Typically places
        it in :attr:`executables_directory`, making it executable with chmod.

        :param filepath: The path on the host to the binary.
        :param with_name: Optional name to rename the binary on the device.
        :param timeout: Time in seconds to allow the push & setup.
        :return: Path to the installed binary on the device, or None on failure.
        :raises TargetStableError: If the push or setup steps fail.
        """
        self._ensure_executables_directory_is_writable()
        executable_name: str = with_name or os.path.basename(filepath)

        on_device_file: str = self.path.join(self.working_directory or '', executable_name)
        on_device_executable: str = self.path.join(self.executables_directory or '', executable_name)
        await self.push.asyn(filepath, on_device_file, timeout=timeout)
        if on_device_file != on_device_executable:
            await self.execute.asyn('cp -f -- {} {}'.format(quote(on_device_file), quote(on_device_executable)),
                                    as_root=self.needs_su, timeout=timeout)
            await self.remove.asyn(on_device_file, as_root=self.needs_su)
        await self.execute.asyn("chmod 0777 {}".format(quote(on_device_executable)), as_root=self.needs_su)
        self._installed_binaries[executable_name] = on_device_executable
        return on_device_executable

    @asyn.asyncf
    async def uninstall_package(self, package: str) -> None:
        """
        Uninstall an Android package by name (using ``adb uninstall`` or
        ``pm uninstall``).

        :param package: The package name to remove.
        """
        if isinstance(self.conn, AdbConnection):
            adb_command(self.adb_name, "uninstall {}".format(quote(package)), timeout=30,
                        adb_server=self.adb_server, adb_port=self.adb_port)
        else:
            await self.execute.asyn("pm uninstall {}".format(quote(package)), timeout=30)

    @asyn.asyncf
    async def uninstall_executable(self, executable_name: str) -> None:
        """
        Remove an installed executable from :attr:`executables_directory`.

        :param executable_name: The name of the binary to remove.
        """
        on_device_executable = self.path.join(self.executables_directory or '', executable_name)
        self._ensure_executables_directory_is_writable()
        await self.remove.asyn(on_device_executable, as_root=self.needs_su)

    @asyn.asyncf
    async def dump_logcat(self, filepath: str, filter: Optional[str] = None,
                          logcat_format: Optional[str] = None,
                          append: bool = False, timeout: int = 60) -> None:  # pylint: disable=redefined-builtin
        """
        Collect logcat output from the device and save it to ``filepath`` on the host.

        :param filepath: The file on the host to store the log output.
        :param filter: If provided, a filter specifying which tags to match (e.g. '-s MyTag').
        :param logcat_format: Logcat format (e.g., 'threadtime'), if any.
        :param append: If True, append to the host file instead of overwriting.
        :param timeout: How many seconds to allow for reading the log.
        """
        op = '>>' if append else '>'
        filtstr = ' -s {}'.format(quote(filter)) if filter else ''
        formatstr = ' -v {}'.format(quote(logcat_format)) if logcat_format else ''
        logcat_opts = '-d' + formatstr + filtstr
        if isinstance(self.conn, AdbConnection):
            command = 'logcat {} {} {}'.format(logcat_opts, op, quote(filepath))
            adb_command(self.adb_name, command, timeout=timeout, adb_server=self.adb_server,
                        adb_port=self.adb_port)
        else:
            dev_path = self.get_workpath('logcat')
            if dev_path:
                command = 'logcat {} {} {}'.format(logcat_opts, op, quote(dev_path))
                await self.execute.asyn(command, timeout=timeout)
                await self.pull.asyn(dev_path, filepath)
                await self.remove.asyn(dev_path)

    @asyn.asyncf
    async def clear_logcat(self) -> None:
        """
        Clear the device's logcat (``logcat -c``). Uses a lock to avoid concurrency issues.
        """
        locked = self.clear_logcat_lock.acquire(blocking=False)
        if locked:
            try:
                if isinstance(self.conn, AdbConnection):
                    adb_command(self.adb_name, 'logcat -c', timeout=30, adb_server=self.adb_server,
                                adb_port=self.adb_port)
                else:
                    await self.execute.asyn('logcat -c', timeout=30)
            finally:
                self.clear_logcat_lock.release()

    def get_logcat_monitor(self, regexps: Optional[List[str]] = None) -> LogcatMonitor:
        """
        Create a :class:`LogcatMonitor` object for capturing logcat output from the device.

        :param regexps: An optional list of uncompiled regex strings to filter log entries.
        :return: A new LogcatMonitor instance referencing this AndroidTarget.
        """
        return LogcatMonitor(self, regexps)  # type: ignore

    @call_conn
    def wait_for_device(self, timeout: int = 30) -> None:
        """
        Instruct ADB to wait until the device is present (``adb wait-for-device``).

        :param timeout: Seconds to wait before failing.
        :raises TargetStableError: If waiting times out or if the connection is not ADB.
        """
        if isinstance(self.conn, AdbConnection):
            self.conn.wait_for_device()

    @call_conn
    def reboot_bootloader(self, timeout: int = 30) -> None:
        """
        Reboot the device into fastboot/bootloader mode.

        :param timeout: Time in seconds to allow for device to transition.
        :raises TargetStableError: If not using ADB or the command fails.
        """
        if isinstance(self.conn, AdbConnection):
            self.conn.reboot_bootloader()

    @asyn.asyncf
    async def is_screen_locked(self) -> bool:
        """
        Determine if the lock screen is active (e.g., phone is locked).

        :return: True if the screen is locked, False otherwise.
        """
        screen_state = await self.execute.asyn('dumpsys window')
        return 'mDreamingLockscreen=true' in screen_state

    @asyn.asyncf
    async def is_screen_on(self) -> bool:
        """
        Check if the device screen is currently on.

        :return:
            - True if the screen is on or in certain "doze" states.
            - False if the screen is off or fully asleep.
        :raises TargetStableError: If unable to parse display power state.
        """
        output: str = await self.execute.asyn('dumpsys power')
        match = ANDROID_SCREEN_STATE_REGEX.search(output)
        if match:
            if 'DOZE' in match.group(1).upper():
                return True
            if match.group(1) == 'Dozing':
                return False
            if match.group(1) == 'Asleep':
                return False
            if match.group(1) == 'Awake':
                return True
            return boolean(match.group(1))
        else:
            raise TargetStableError('Could not establish screen state.')

    @asyn.asyncf
    async def ensure_screen_is_on(self, verify: bool = True) -> None:
        """
        If the screen is off, press the power button (keyevent 26) to wake it.
        Optionally verify the screen is on afterwards.

        :param verify: If True, raise an error if the screen doesn't turn on.
        :raises TargetStableError: If the screen is still off after the attempt.
        """
        if not await self.is_screen_on.asyn():
            # The adb shell input keyevent 26 command is used to
            # simulate pressing the power button on an Android device.
            self.execute('input keyevent 26')
        if verify and not await self.is_screen_on.asyn():
            raise TargetStableError('Display cannot be turned on.')

    @asyn.asyncf
    async def ensure_screen_is_on_and_stays(self, verify: bool = True, mode: int = 7) -> None:
        """
        Calls ``AndroidTarget.ensure_screen_is_on(verify)`` then additionally
        sets the screen stay on mode to ``mode``.
        mode options -
        0: Never stay on while plugged in.
        1: Stay on while plugged into an AC charger.
        2: Stay on while plugged into a USB charger.
        4: Stay on while on a wireless charger.
        You can combine these values using bitwise OR.
        For example, 3 (1 | 2) will stay on while plugged into either an AC or USB charger

        :param verify: If True, check that the screen does come on.
        :param mode: A bitwise combination of (1 for AC, 2 for USB, 4 for wireless).
        """
        await self.ensure_screen_is_on.asyn(verify=verify)
        await self.set_stay_on_mode.asyn(mode)

    @asyn.asyncf
    async def ensure_screen_is_off(self, verify: bool = True) -> None:
        """
        Checks if the devices screen is on and if so turns it off.
        If ``verify`` is set to ``True`` then a ``TargetStableError``
        will be raise if the display cannot be turned off. E.g. if
        always on mode is enabled.

        :param verify: Raise an error if the screen remains on afterwards.
        :raises TargetStableError: If the display remains on due to always-on or lock states.
        """
        # Allow 2 attempts to help with cases of ambient display modes
        # where the first attempt will switch the display fully on.
        for _ in range(2):
            if await self.is_screen_on.asyn():
                await self.execute.asyn('input keyevent 26')
                time.sleep(0.5)
        if verify and await self.is_screen_on.asyn():
            msg: str = 'Display cannot be turned off. Is always on display enabled?'
            raise TargetStableError(msg)

    @asyn.asyncf
    async def set_auto_brightness(self, auto_brightness: bool) -> None:
        """
        Enable or disable automatic screen brightness.

        :param auto_brightness: True to enable auto-brightness, False to disable.
        """
        cmd = 'settings put system screen_brightness_mode {}'
        await self.execute.asyn(cmd.format(int(boolean(auto_brightness))))

    @asyn.asyncf
    async def get_auto_brightness(self) -> bool:
        """
        Check if auto-brightness is enabled.

        :return: True if auto-brightness is on, False otherwise.
        """
        cmd = 'settings get system screen_brightness_mode'
        return boolean((await self.execute.asyn(cmd)).strip())

    @asyn.asyncf
    async def set_brightness(self, value: int) -> None:
        """
        Manually set screen brightness to an integer between 0 and 255.
        This also disables auto-brightness first.

        :param value: Desired brightness level (0-255).
        :raises ValueError: If the given value is outside [0..255].
        """
        if not 0 <= value <= 255:
            msg = 'Invalid brightness "{}"; Must be between 0 and 255'
            raise ValueError(msg.format(value))
        await self.set_auto_brightness.asyn(False)
        cmd = 'settings put system screen_brightness {}'
        await self.execute.asyn(cmd.format(int(value)))

    @asyn.asyncf
    async def get_brightness(self) -> int:
        """
        Return the current screen brightness (0..255).

        :return: The brightness setting.
        """
        cmd = 'settings get system screen_brightness'
        return integer((await self.execute.asyn(cmd)).strip())

    @asyn.asyncf
    async def set_screen_timeout(self, timeout_ms: int) -> None:
        """
        Set the screen-off timeout in milliseconds.

        :param timeout_ms: Number of ms before the screen turns off when idle.
        """
        cmd = 'settings put system screen_off_timeout {}'
        await self.execute.asyn(cmd.format(int(timeout_ms)))

    @asyn.asyncf
    async def get_screen_timeout(self) -> int:
        """
        Get the screen-off timeout (ms).

        :return: Milliseconds before screen turns off.
        """
        cmd = 'settings get system screen_off_timeout'
        return int((await self.execute.asyn(cmd)).strip())

    @asyn.asyncf
    async def get_airplane_mode(self) -> bool:
        """
        Check if airplane mode is active (global setting).

        .. note:: Requires the device to be rooted if the device is running Android 7+.

        :return: True if airplane mode is on, otherwise False.
        """
        cmd = 'settings get global airplane_mode_on'
        return boolean((await self.execute.asyn(cmd)).strip())

    @asyn.asyncf
    async def get_stay_on_mode(self) -> int:
        """
        Returns an integer between ``0`` and ``7`` representing the current
        stay-on mode of the device.
        0: Never stay on while plugged in.
        1: Stay on while plugged into an AC charger.
        2: Stay on while plugged into a USB charger.
        4: Stay on while on a wireless charger.
        Combinations of these values can be used (e.g., 3 for both AC and USB chargers)

        :return: The integer bitmask (0..7).
        """
        cmd = 'settings get global stay_on_while_plugged_in'
        return int((await self.execute.asyn(cmd)).strip())

    @asyn.asyncf
    async def set_airplane_mode(self, mode: bool) -> None:
        """
        Enable or disable airplane mode. On Android 7+, requires root.

        :param mode: True to enable airplane mode, False to disable.
        :raises TargetStableError: If root is required but the device is not rooted.
        """
        root_required: bool = await self.get_sdk_version.asyn() > 23
        if root_required and not self.is_rooted:
            raise TargetStableError('Root is required to toggle airplane mode on Android 7+')
        modeint = int(boolean(mode))
        cmd = 'settings put global airplane_mode_on {}'
        await self.execute.asyn(cmd.format(modeint))
        await self.execute.asyn('am broadcast -a android.intent.action.AIRPLANE_MODE '
                                '--ez state {}'.format(mode), as_root=root_required)

    @asyn.asyncf
    async def get_auto_rotation(self) -> bool:
        """
        Check if auto-rotation is enabled (system setting).

        :return: True if accelerometer-based rotation is enabled, False otherwise.
        """
        cmd = 'settings get system accelerometer_rotation'
        return boolean((await self.execute.asyn(cmd)).strip())

    @asyn.asyncf
    async def set_auto_rotation(self, autorotate: bool) -> None:
        """
        Enable or disable auto-rotation of the screen.

        :param autorotate: True to enable, False to disable.
        """
        cmd = 'settings put system accelerometer_rotation {}'
        await self.execute.asyn(cmd.format(int(boolean(autorotate))))

    @asyn.asyncf
    async def set_natural_rotation(self) -> None:
        """
        Sets the screen orientation of the device to its natural (0 degrees)
        orientation.
        """
        await self.set_rotation.asyn(0)

    @asyn.asyncf
    async def set_left_rotation(self) -> None:
        """
        Sets the screen orientation of the device to 90 degrees.
        """
        await self.set_rotation.asyn(1)

    @asyn.asyncf
    async def set_inverted_rotation(self) -> None:
        """
        Sets the screen orientation of the device to its inverted (180 degrees)
        orientation.
        """
        await self.set_rotation.asyn(2)

    @asyn.asyncf
    async def set_right_rotation(self) -> None:
        """
        Sets the screen orientation of the device to 270 degrees.
        """
        await self.set_rotation.asyn(3)

    @asyn.asyncf
    async def get_rotation(self) -> Optional[int]:
        """
        Returns an integer value representing the orientation of the devices
        screen. ``0`` : Natural, ``1`` : Rotated Left, ``2`` : Inverted
        and ``3`` : Rotated Right.

        :return: The rotation value or None if not found.
        """
        output = await self.execute.asyn('dumpsys input')
        match = ANDROID_SCREEN_ROTATION_REGEX.search(output)
        if match:
            return int(match.group('rotation'))
        else:
            return None

    @asyn.asyncf
    async def set_rotation(self, rotation: int) -> None:
        """
        Specify an integer representing the desired screen rotation with the
        following mappings: Natural: ``0``, Rotated Left: ``1``, Inverted : ``2``
        and Rotated Right : ``3``.

        :param rotation: Integer in [0..3].
        :raises ValueError: If rotation is not within [0..3].
        """
        if not 0 <= rotation <= 3:
            raise ValueError('Rotation value must be between 0 and 3')
        await self.set_auto_rotation.asyn(False)
        cmd = 'settings put system user_rotation {}'
        await self.execute.asyn(cmd.format(rotation))

    @asyn.asyncf
    async def set_stay_on_never(self) -> None:
        """
        Sets the stay-on mode to ``0``, where the screen will turn off
        as standard after the timeout.
        """
        await self.set_stay_on_mode.asyn(0)

    @asyn.asyncf
    async def set_stay_on_while_powered(self) -> None:
        """
        Sets the stay-on mode to ``7``, where the screen will stay on
        while the device is charging
        """
        await self.set_stay_on_mode.asyn(7)

    @asyn.asyncf
    async def set_stay_on_mode(self, mode: int) -> None:
        """
        0: Never stay on while plugged in.
        1: Stay on while plugged into an AC charger.
        2: Stay on while plugged into a USB charger.
        4: Stay on while on a wireless charger.
        You can combine these values using bitwise OR.
        For example, 3 (1 | 2) will stay on while plugged into either an AC or USB charger

        :param mode: Value in [0..7].
        :raises ValueError: If outside [0..7].
        """
        if not 0 <= mode <= 7:
            raise ValueError('Screen stay on mode must be between 0 and 7')
        cmd = 'settings put global stay_on_while_plugged_in {}'
        await self.execute.asyn(cmd.format(mode))

    @asyn.asyncf
    async def open_url(self, url: str, force_new: bool = False) -> None:
        """
        Launch an intent to view a given URL, optionally forcing a new task in
        the activity stack.

        :param url: URL to open (e.g. "https://www.example.com").
        :param force_new: If True, use flags to clear the existing activity stack,
            forcing a fresh activity.
        """
        cmd = 'am start -a android.intent.action.VIEW -d {}'

        if force_new:
            cmd = cmd + ' -f {}'.format(INTENT_FLAGS['ACTIVITY_NEW_TASK'] | INTENT_FLAGS['ACTIVITY_CLEAR_TASK'])

        await self.execute.asyn(cmd.format(quote(url)))

    @asyn.asyncf
    async def homescreen(self) -> None:
        """
        Return to the home screen by launching the MAIN/HOME intent.
        """
        await self.execute.asyn('am start -a android.intent.action.MAIN -c android.intent.category.HOME')

    def _resolve_paths(self):
        if self.working_directory is None:
            self.working_directory = self.path.join(self.external_storage, 'devlib-target')
        if self.tmp_directory is None:
            # Do not rely on the generic default here, as we need to provide an
            # android-specific default in case it fails.
            try:
                tmp = self.execute(f'{quote(self.busybox)} mktemp -d') if self.busybox else '/data/local/tmp'
            except Exception:
                tmp = '/data/local/tmp'
            self.tmp_directory = tmp
        if self.executables_directory is None:
            self.executables_directory = self.path.join(self.tmp_directory, 'bin')

    @asyn.asyncf
    async def _ensure_executables_directory_is_writable(self) -> None:
        """
        Check if the executables directory is on a writable mount. If not, attempt
        to remount it read/write as root.

        :raises TargetStableError: If the directory cannot be remounted or found in fstab.
        """
        matched: List['FstabEntry'] = []
        for entry in await self.list_file_systems.asyn():
            if self.executables_directory is not None and self.executables_directory.rstrip('/').startswith(entry.mount_point):
                matched.append(entry)
        if matched:
            entry = sorted(matched, key=lambda x: len(x.mount_point))[-1]
            if 'rw' not in entry.options:
                await self.execute.asyn('mount -o rw,remount {} {}'.format(quote(entry.device),
                                                                           quote(entry.mount_point)),
                                        as_root=True)
        else:
            message = 'Could not find mount point for executables directory {}'
            raise TargetStableError(message.format(self.executables_directory))

    _charging_enabled_path = '/sys/class/power_supply/battery/charging_enabled'

    @property
    def charging_enabled(self) -> Optional[bool]:
        """
        Whether drawing power to charge the battery is enabled

        Not all devices have the ability to enable/disable battery charging
        (e.g. because they don't have a battery). In that case,
        ``charging_enabled`` is None.

        :return:
            - True if charging is enabled
            - False if disabled
            - None if the sysfs entry is absent
        """
        if not self.file_exists(self._charging_enabled_path):
            return None
        return self.read_bool(self._charging_enabled_path)

    @charging_enabled.setter
    def charging_enabled(self, enabled: bool) -> None:
        """
        Enable/disable drawing power to charge the battery

        Not all devices have this facility. In that case, do nothing.

        :param enabled: True to enable charging, False to disable.
        """
        if not self.file_exists(self._charging_enabled_path):
            return
        self.write_value(self._charging_enabled_path, int(bool(enabled)))


FstabEntry = namedtuple('FstabEntry', ['device', 'mount_point', 'fs_type', 'options', 'dump_freq', 'pass_num'])
PsEntry = namedtuple('PsEntry', 'user pid tid ppid vsize rss wchan pc state name')
LsmodEntry = namedtuple('LsmodEntry', ['name', 'size', 'use_count', 'used_by'])


class Cpuinfo(object):
    """
    Represents the parsed contents of ``/proc/cpuinfo`` on the target.

    :param sections: A list of dictionaries, where each dictionary represents a
        block of lines corresponding to a CPU. Key-value pairs correspond to
        lines like ``CPU part: 0xd03`` or ``model name: Cortex-A53``.
    :param text: The full text of the original ``/proc/cpuinfo`` content.
    """
    @property
    @memoized
    def architecture(self) -> Optional[str]:
        """
        architecture as per cpuinfo
        """
        if self.sections:
            for section in self.sections:
                if 'CPU architecture' in section:
                    return section['CPU architecture']
                if 'architecture' in section:
                    return section['architecture']
        return None

    @property
    @memoized
    def cpu_names(self) -> List[caseless_string]:
        """
        A list of CPU names derived from fields like ``CPU part`` or ``model name``.
        If found globally, that name is reused for each CPU. If found per-CPU,
        you get multiple entries.

        :return: List of CPU names, one per processor entry.
        """
        cpu_names: List[Optional[str]] = []
        global_name: Optional[str] = None
        if self.sections:
            for section in self.sections:
                if 'processor' in section:
                    if 'CPU part' in section:
                        cpu_names.append(_get_part_name(section))
                    elif 'model name' in section:
                        cpu_names.append(_get_model_name(section))
                    else:
                        cpu_names.append(None)
                elif 'CPU part' in section:
                    global_name = _get_part_name(section)
        return [caseless_string(c or global_name) for c in cpu_names]

    def __init__(self, text: str):
        self.sections: List[Dict[str, str]] = []
        self.text = ''
        self.parse(text)

    @memoized
    def get_cpu_features(self, cpuid: int = 0) -> List[str]:
        """
        get the Features field of the specified cpu
        """
        global_features: List[str] = []
        if self.sections:
            for section in self.sections:
                if 'processor' in section:
                    if int(section.get('processor') or -1) != cpuid:
                        continue
                    if 'Features' in section:
                        return section.get('Features', '').split()
                    elif 'flags' in section:
                        return section.get('flags', '').split()
                elif 'Features' in section:
                    global_features = section.get('Features', '').split()
                elif 'flags' in section:
                    global_features = section.get('flags', '').split()
        return global_features

    def parse(self, text: str) -> None:
        """
        Parse the provided ``/proc/cpuinfo`` text, splitting it into separate
        sections for each CPU.

        :param text: The full multiline content of /proc/cpuinfo.
        """
        self.sections = []
        current_section: Dict[str, str] = {}
        self.text = text.strip()
        for line in self.text.split('\n'):
            line = line.strip()
            if line:
                key, value = line.split(':', 1)
                current_section[key.strip()] = value.strip()
            else:  # not line
                self.sections.append(current_section)
                current_section = {}
        self.sections.append(current_section)

    def __str__(self):
        return 'CpuInfo({})'.format(self.cpu_names)

    __repr__ = __str__


class KernelVersion(object):
    """
    Class representing the version of a target kernel

    Not expected to work for very old (pre-3.0) kernel version numbers.

    :ivar release: Version number/revision string. Typical output of
                   ``uname -r``
    :ivar version: Extra version info (aside from ``release``) reported by
                   ``uname``
    :ivar version_number: Main version number (e.g. 3 for Linux 3.18)
    :ivar major: Major version number (e.g. 18 for Linux 3.18)
    :ivar minor: Minor version number for stable kernels (e.g. 9 for 4.9.9). May
                 be None
    :ivar rc: Release candidate number (e.g. 3 for Linux 4.9-rc3). May be None.
    :ivar commits: Number of additional commits on the branch. May be None.
    :ivar sha1: Kernel git revision hash, if available (otherwise None)
    :ivar android_version: Android version, if available (otherwise None)
    :ivar gki_abi: GKI kernel abi, if available (otherwise None)

    :ivar parts: Tuple of version number components. Can be used for
                 lexicographically comparing kernel versions.
    """
    def __init__(self, version_string: str):
        if ' #' in version_string:
            release, version = version_string.split(' #')
            self.release: str = release
            self.version: str = version
        elif version_string.startswith('#'):
            self.release = ''
            self.version = version_string
        else:
            self.release = version_string
            self.version = ''

        self.version_number: Optional[int] = None
        self.major: Optional[int] = None
        self.minor: Optional[int] = None
        self.sha1: Optional[str] = None
        self.rc: Optional[int] = None
        self.commits: Optional[int] = None
        self.gki_abi: Optional[str] = None
        self.android_version: Optional[int] = None
        match = KVERSION_REGEX.match(version_string)
        if match:
            groups = match.groupdict()
            self.version_number = int(groups['version'])
            self.major = int(groups['major'])
            if groups['minor'] is not None:
                self.minor = int(groups['minor'])
            if groups['rc'] is not None:
                self.rc = int(groups['rc'])
            if groups['commits'] is not None:
                self.commits = int(groups['commits'])
            if groups['sha1'] is not None:
                self.sha1 = match.group('sha1')
            if groups['gki_abi'] is not None:
                self.gki_abi = match.group('gki_abi')
            if groups['android_version'] is not None:
                self.android_version = int(match.group('android_version'))

        self.parts: Tuple[Optional[int], Optional[int], Optional[int]] = (self.version_number, self.major, self.minor)

    def __str__(self):
        return '{} {}'.format(self.release, self.version)

    __repr__ = __str__


class HexInt(int):
    """
    An int subclass that is displayed in hexadecimal form.

    Example usage:

    .. code-block:: python

        val = HexInt('FF')    # Parse hex string as int
        print(val)            # Prints: 0xff
        print(int(val))       # Prints: 255
    """

    def __new__(cls, val: Union[str, int, bytearray] = 0, base=16):
        """
        Construct a HexInt object, interpreting ``val`` as a base-16 value
        unless it's already a number or bytearray.

        :param val: The initial value. If str, is parsed as base-16 by default;
            if int or bytearray, used directly.
        :param base: Numerical base (defaults to 16).
        :raises TypeError: If ``val`` is not a supported type (str, int, or bytearray).
        """
        super_new = super(HexInt, cls).__new__
        if isinstance(val, Number):
            return super_new(cls, val)
        elif isinstance(val, bytearray):
            val = int.from_bytes(val, byteorder=sys.byteorder)
            return super(HexInt, cls).__new__(cls, val)
        elif isinstance(val, str):
            return super(HexInt, cls).__new__(cls, int(val, base))
        else:
            raise TypeError("Unsupported type for HexInt")

    def __str__(self):
        """
        Return a hexadecimal string representation of the integer, stripping
        any trailing ``L`` in Python 2.x.
        """
        return hex(self).strip('L')


class KernelConfigTristate(Enum):
    """
    Represents a kernel config option that may be ``y``, ``n``, or ``m``.
    Commonly seen in kernel ``.config`` files as:

    - ``CONFIG_FOO=y``
    - ``CONFIG_BAR=n``
    - ``CONFIG_BAZ=m``

    Enum members:
      * ``YES`` -> 'y'
      * ``NO`` -> 'n'
      * ``MODULE`` -> 'm'
    """
    YES = 'y'
    NO = 'n'
    MODULE = 'm'

    def __bool__(self):
        """
        Allow usage in boolean contexts:

        * True if the config is 'y' or 'm'
        * False if the config is 'n'
        """
        return self in (self.YES, self.MODULE)

    def __nonzero__(self):
        """
        Python 2.x compatibility for boolean evaluation.
        """
        return self.__bool__()

    @classmethod
    def from_str(cls, str_: str) -> 'KernelConfigTristate':
        """
        Convert a kernel config string ('y', 'n', or 'm') to the corresponding
        enum member.

        :param str_: The single-character string from kernel config.
        :return: The enum member that matches the provided string.
        :raises ValueError: If the string is not 'y', 'n', or 'm'.
        """
        for state in cls:
            if state.value == str_:
                return state
        raise ValueError('No kernel config tristate value matches "{}"'.format(str_))


class TypedKernelConfig(Mapping):   # type: ignore
    """
    A mapping-like object representing typed kernel config parameters. Keys are
    canonicalized config names (e.g. "CONFIG_FOO"), and values may be strings, ints,
    :class:`HexInt`, or :class:`KernelConfigTristate`.

    :param not_set_regex: A regex that matches lines in the form ``# CONFIG_ABC is not set``.

    :param mapping: An optional initial mapping of config keys to string values.
        Typically set by parsing a kernel .config file or /proc/config.gz content.
    """
    not_set_regex = re.compile(r'# (\S+) is not set')

    @staticmethod
    def get_config_name(name: str) -> str:
        """
        Ensure the config name starts with 'CONFIG_', returning
        the canonical form.

        :param name: A raw config key name (e.g. 'ABC').
        :return: The canonical name (e.g. 'CONFIG_ABC').
        """
        name = name.upper()
        if not name.startswith('CONFIG_'):
            name = 'CONFIG_' + name
        return name

    def __init__(self, mapping: Optional[Maptype] = None):
        """
        Initialize a typed kernel config from an existing dictionary or None.

        :param mapping: Existing config data (raw strings), keyed by config name.
        """
        mapping = mapping if mapping is not None else {}
        self._config: Dict[str, str] = {
            # Ensure we use the canonical name of the config keys for internal
            # representation
            self.get_config_name(k): v
            for k, v in dict(mapping).items()
        }

    @classmethod
    def from_str(cls, text: str) -> 'TypedKernelConfig':
        """
        Build a typed config by parsing raw text of a kernel config file.

        :param text: Contents of the kernel config, including lines such as
            ``CONFIG_ABC=y`` or ``# CONFIG_DEF is not set``.
        :return: A :class:`TypedKernelConfig` reflecting typed config values.
        """
        return cls(cls._parse_text(text))

    @staticmethod
    def _val_to_str(val: Optional[Union[KernelConfigTristate, str]]) -> str:
        "Convert back values to Kconfig-style string value"
        # Special case the gracefully handle the output of get()
        if val is None:
            return ""
        elif isinstance(val, KernelConfigTristate):
            return val.value
        elif isinstance(val, str):
            return '"{}"'.format(val.strip('"'))
        else:
            return str(val)

    def __str__(self):
        """
        Convert the typed config back to a kernel config-style string, e.g.
        "CONFIG_FOO=y\nCONFIG_BAR=\"value\"\n..."

        :return: A multi-line string representation of the typed config.
        """
        return '\n'.join(
            '{}={}'.format(k, self._val_to_str(v))
            for k, v in self.items()
        )

    @staticmethod
    def _parse_val(k: str, v: Union[str, int, HexInt,
                                    KernelConfigTristate]) -> Optional[Union[KernelConfigTristate,
                                                                             HexInt, int, str]]:
        """
        Parse a value of types handled by Kconfig:
            * string
            * bool
            * tristate
            * hex
            * int

        Since bool cannot be distinguished from tristate, tristate is
        always used. :meth:`KernelConfigTristate.__bool__` will allow using
        it as a bool though, so it should not impact user code.

        :param k: The config key name (not used heavily).
        :param v: The raw string or typed object.
        :return: The typed version of the value.
        """
        if not v:
            return None

        if isinstance(v, str):
            # Handle "string" type
            if v.startswith('"'):
                # Strip enclosing "
                return v[1:-1]

            else:
                try:
                    # Handles "bool" and "tristate" types
                    return KernelConfigTristate.from_str(v)
                except ValueError:
                    pass

                try:
                    # Handles "int" type
                    return int(v)
                except ValueError:
                    pass

                try:
                    # Handles "hex" type
                    return HexInt(v)
                except ValueError:
                    pass

                # If no type could be parsed
                raise ValueError('Could not parse Kconfig key: {}={}'.format(
                    k, v
                ), k, v
                )
        return None

    @classmethod
    def _parse_text(cls, text: str) -> Dict[str, Optional[Union[KernelConfigTristate, HexInt, int, str]]]:
        """
        parse the kernel config text and create a dictionary of the configs
        """
        config: Dict[str, Optional[Union[KernelConfigTristate, HexInt, int, str]]] = {}
        for line in text.splitlines():
            line = line.strip()

            # skip empty lines
            if not line:
                continue

            if line.startswith('#'):
                match = cls.not_set_regex.search(line)
                if match:
                    value: str = 'n'
                    name: str = match.group(1)
                else:
                    continue
            else:
                name, value = line.split('=', 1)

            name = cls.get_config_name(name.strip())
            parsed_value: Optional[Union[KernelConfigTristate, HexInt, int, str]] = cls._parse_val(name, value.strip())
            config[name] = parsed_value
        return config

    def __getitem__(self, name: str) -> str:
        name = self.get_config_name(name)
        try:
            return self._config[name]
        except KeyError:
            raise KernelConfigKeyError(
                "{} is not exposed in kernel config".format(name),
                name
            )

    def __iter__(self):
        return iter(self._config)

    def __len__(self):
        return len(self._config)

# FIXME - annotating name as str gives some type errors as Mapping superclass expects object
    def __contains__(self, name):
        name = self.get_config_name(name)
        return name in self._config

    def like(self, name: str) -> Dict[str, str]:
        """
        Return a dictionary of key-value pairs where the keys match the given regular expression pattern.
        """
        regex = re.compile(name, re.I)
        return {
            k: v for k, v in self.items()
            if regex.search(k)
        }

    def is_enabled(self, name: str) -> bool:
        """
        true if the config is enabled in kernel
        """
        return self.get(name) is KernelConfigTristate.YES

    def is_module(self, name: str) -> bool:
        """
        true if the config is of Module type
        """
        return self.get(name) is KernelConfigTristate.MODULE

    def is_not_set(self, name: str) -> bool:
        """
        true if the config is not enabled
        """
        return self.get(name) is KernelConfigTristate.NO

    def has(self, name: str) -> bool:
        """
        true if the config is either enabled or it is a module
        """
        return self.is_enabled(name) or self.is_module(name)


class KernelConfig(object):
    """
    Backward compatibility shim on top of :class:`TypedKernelConfig`.

    This class does not provide a Mapping API and only return string values.
    """
    @staticmethod
    def get_config_name(name: str) -> str:
        return TypedKernelConfig.get_config_name(name)

    def __init__(self, text: str):
        # Expose typed_config as a non-private attribute, so that user code
        # needing it can get it from any existing producer of KernelConfig.
        self.typed_config = TypedKernelConfig.from_str(text)
        # Expose the original text for backward compatibility
        self.text = text

    def __bool__(self):
        return bool(self.typed_config)

    not_set_regex = TypedKernelConfig.not_set_regex

    def iteritems(self) -> Iterator[Tuple[str, str]]:
        """
        Iterate over the items in the typed configuration, converting each value to a string.
        """
        for k, v in self.typed_config.items():
            yield (k, self.typed_config._val_to_str(v))

    items = iteritems

    def get(self, name: str, strict: bool = False) -> Optional[str]:
        """
        Retrieve a value from the typed configuration and convert it to a string.
        """
        if strict:
            val: Optional[str] = self.typed_config[name]
        else:
            val = self.typed_config.get(name)

        return self.typed_config._val_to_str(val)

    def like(self, name: str) -> Dict[str, str]:
        """
        Return a dictionary of key-value pairs where the keys match the given regular expression pattern.
        """
        return {
            k: self.typed_config._val_to_str(v)
            for k, v in self.typed_config.like(name).items()
        }

    def is_enabled(self, name: str) -> bool:
        """
        true if the config is enabled in kernel
        """
        return self.typed_config.is_enabled(name)

    def is_module(self, name: str) -> bool:
        """
        true if the config is of Module type
        """
        return self.typed_config.is_module(name)

    def is_not_set(self, name: str) -> bool:
        """
        true if the config is not enabled
        """
        return self.typed_config.is_not_set(name)

    def has(self, name: str) -> bool:
        """
        true if the config is either enabled or it is a module
        """
        return self.typed_config.has(name)


class LocalLinuxTarget(LinuxTarget):
    """
    A specialized :class:`Target` subclass representing the local Linux system
    (i.e., no remote connection needed). In many respects, this parallels
    :class:`LinuxTarget`, but uses :class:`LocalConnection` under the hood.

    :param connection_settings: Dictionary specifying local connection options
        (often unused or minimal).
    :param platform: A ``Platform`` object if you want to specify architecture,
        kernel version, etc. If None, a default is inferred from the host system.
    :param working_directory: A writable directory on the local machine for devlibs
        temporary operations. If None, a subfolder of /tmp or similar is often used.
    :param executables_directory: Directory for installing binaries from devlib,
        if needed.
    :param connect: Whether to connect (initialize local environment) immediately.
    :param modules: Additional devlib modules to load at construction time.
    :param load_default_modules: If True, also load modules listed in
        :attr:`default_modules`.
    :param shell_prompt: Regex matching the local shell prompt (usually not used
        since local commands are run directly).
    :param conn_cls: Connection class to use, typically :class:`LocalConnection`.
    :param is_container: If True, indicates we’re running in a container environment
        rather than the full host OS.
    :param max_async: Maximum concurrent asynchronous commands allowed.

    """

    def __init__(self,
                 connection_settings: Optional[UserConnectionSettings] = None,
                 platform: Optional[Platform] = None,
                 working_directory: Optional[str] = None,
                 executables_directory: Optional[str] = None,
                 connect: bool = True,
                 modules: Optional[Dict[str, Dict[str, Type[Module]]]] = None,
                 load_default_modules: bool = True,
                 shell_prompt: Pattern[str] = DEFAULT_SHELL_PROMPT,
                 conn_cls: 'InitCheckpointMeta' = LocalConnection,
                 is_container: bool = False,
                 max_async: int = 50,
                 tmp_directory: Optional[str] = None,
                 ):
        """
        Initialize a LocalLinuxTarget, representing the local machine as the devlib
        target. Optionally connect and load modules immediately.
        """
        super(LocalLinuxTarget, self).__init__(connection_settings=connection_settings,
                                               platform=platform,
                                               working_directory=working_directory,
                                               executables_directory=executables_directory,
                                               connect=connect,
                                               modules=modules,
                                               load_default_modules=load_default_modules,
                                               shell_prompt=shell_prompt,
                                               conn_cls=conn_cls,
                                               is_container=is_container,
                                               max_async=max_async,
                                               tmp_directory=tmp_directory,
                                               )

    def _resolve_paths(self) -> None:
        """
        Resolve or finalize local working directories/executables directories.
        By default, uses a subfolder of /tmp if none is set.
        """
        if self.working_directory is None:
            self.working_directory = '/tmp/devlib-target'


def _get_model_name(section: Dict[str, str]) -> str:
    """
    get model name from section of cpu info
    """
    name_string: str = section['model name']
    parts: List[str] = name_string.split('@')[0].strip().split()
    return ' '.join([p for p in parts
                     if '(' not in p and p != 'CPU'])


def _get_part_name(section: Dict[str, str]) -> str:
    """
    get part name from cpu info
    """
    implementer: str = section.get('CPU implementer', '0x0')
    part: str = section['CPU part']
    variant: str = section.get('CPU variant', '0x0')
    name = get_cpu_name(*list(map(integer, [implementer, part, variant])))
    if name is None:
        name = f'{implementer}/{part}/{variant}'
    return name


Node = Union[str, Dict[str, 'Node']]


def _build_path_tree(path_map: Dict[str, str], basepath: str,
                     sep: str = os.path.sep, dictcls=dict) -> Union[str, Dict[str, 'Node']]:
    """
    Convert a flat mapping of paths to values into a nested structure of
    dict-like object (``dict``'s by default), mirroring the directory hierarchy
    represented by the paths relative to ``basepath``.

    """
    def process_node(node: 'Node', path: str, value: str):
        parts = path.split(sep, 1)
        if len(parts) == 1 and not isinstance(node, str):   # leaf
            node[parts[0]] = value
        else:  # branch
            if not isinstance(node, str):
                if parts[0] not in node:
                    node[parts[0]] = dictcls()
                process_node(node[parts[0]], parts[1], value)

    relpath_map: Dict[str, str] = {os.path.relpath(p, basepath): v
                                   for p, v in path_map.items()}

    if len(relpath_map) == 1 and list(relpath_map.keys())[0] == '.':
        result: Union[str, Dict[str, Any]] = list(relpath_map.values())[0]
    else:
        result = dictcls()
        for path, value in relpath_map.items():
            if not isinstance(result, str):
                process_node(result, path, value)

    return result


class ChromeOsTarget(LinuxTarget):
    """
    :class:`ChromeOsTarget` is a subclass of :class:`LinuxTarget` with
    additional features specific to a device running ChromeOS for example,
    if supported, its own android container which can be accessed via the
    ``android_container`` attribute. When making calls to or accessing
    properties and attributes of the ChromeOS target, by default they will
    be applied to Linux target as this is where the majority of device
    configuration will be performed and if not available, will fall back to
    using the android container if available. This means that all the
    available methods from
    :class:`LinuxTarget` and :class:`AndroidTarget` are available for
    :class:`ChromeOsTarget` if the device supports android otherwise only the
    :class:`LinuxTarget` methods will be available.

    :param working_directory: This is the location of the working directory to
        be used for the Linux target container. If not specified will default to
        ``"/mnt/stateful_partition/devlib-target"``.

    :param android_working_directory: This is the location of the working
        directory to be used for the android container. If not specified it will
        use the working directory default for :class:`AndroidTarget.`.

    :param android_executables_directory: This is the location of the
        executables directory to be used for the android container. If not
        specified will default to a ``bin`` subdirectory in the
        ``android_working_directory.``

    :param package_data_directory: This is the location of the data stored
        for installed Android packages on the device.
    """

    os: str = 'chromeos'

    # pylint: disable=too-many-locals,too-many-arguments
    def __init__(self,
                 connection_settings: Optional[UserConnectionSettings] = None,
                 platform: Optional[Platform] = None,
                 working_directory: Optional[str] = None,
                 executables_directory: Optional[str] = None,
                 android_working_directory: Optional[str] = None,
                 android_executables_directory: Optional[str] = None,
                 connect: bool = True,
                 modules: Optional[Dict[str, Dict[str, Type[Module]]]] = None,
                 load_default_modules: bool = True,
                 shell_prompt: Pattern[str] = DEFAULT_SHELL_PROMPT,
                 package_data_directory: str = "/data/data",
                 is_container: bool = False,
                 max_async: int = 50,
                 tmp_directory: Optional[str] = None,
                 ):
        """
        Initialize a ChromeOsTarget for interacting with a device running Chrome OS
        in developer mode (exposing SSH).
        """

        self.supports_android: Optional[bool] = None
        self.android_container: Optional[AndroidTarget] = None

        # Pull out ssh connection settings
        ssh_conn_params: List[str] = ['host', 'username', 'password', 'keyfile',
                                      'port', 'timeout', 'sudo_cmd',
                                      'strict_host_check', 'use_scp',
                                      'total_transfer_timeout', 'poll_transfers',
                                      'start_transfer_poll_delay']
        self.ssh_connection_settings: SshUserConnectionSettings = {}
        if connection_settings:
            update_dict = cast(SshUserConnectionSettings,
                               {key: value for key, value in connection_settings.items() if key in ssh_conn_params})
            self.ssh_connection_settings.update(update_dict)

        super().__init__(connection_settings=self.ssh_connection_settings,
                         platform=platform,
                         working_directory=working_directory,
                         executables_directory=executables_directory,
                         connect=False,
                         modules=modules,
                         load_default_modules=load_default_modules,
                         shell_prompt=shell_prompt,
                         conn_cls=SshConnection,
                         is_container=is_container,
                         max_async=max_async,
                         tmp_directory=tmp_directory)

        # We can't determine if the target supports android until connected to the linux host so
        # create unconditionally.
        # Pull out adb connection settings
        adb_conn_params = ['device', 'adb_server', 'adb_port', 'timeout']
        self.android_connection_settings: AdbUserConnectionSettings = {}
        if connection_settings:
            update_dict_adb = cast(AdbUserConnectionSettings,
                                   {key: value for key, value in connection_settings.items() if key in adb_conn_params})
            self.android_connection_settings.update(update_dict_adb)

            # If adb device is not explicitly specified use same as ssh host
            if not connection_settings.get('device', None):
                device = connection_settings.get('host', None)
                if device:
                    self.android_connection_settings['device'] = device

            self.android_container = AndroidTarget(connection_settings=self.android_connection_settings,
                                                   platform=platform,
                                                   working_directory=android_working_directory,
                                                   executables_directory=android_executables_directory,
                                                   connect=False,
                                                   load_default_modules=False,
                                                   shell_prompt=shell_prompt,
                                                   conn_cls=AdbConnection,
                                                   package_data_directory=package_data_directory,
                                                   is_container=True)
            if connect:
                self.connect()

    def __getattr__(self, attr: str):
        """
        By default use the linux target methods and attributes however,
        if not present, use android implementation if available.
        """
        try:
            return super().__getattribute__(attr)
        except AttributeError:
            if hasattr(self.android_container, attr):
                return getattr(self.android_container, attr)
            raise

    @asyn.asyncf
    async def connect(self, timeout: int = 30, check_boot_completed: bool = True, max_async: Optional[int] = None) -> None:
        super().connect(
            timeout=timeout,
            check_boot_completed=check_boot_completed,
            max_async=max_async,
        )

        # Assume device supports android apps if container directory is present
        if self.supports_android is None:
            self.supports_android = self.directory_exists('/opt/google/containers/android/')

        if self.supports_android and self.android_container:
            self.android_container.connect(timeout)
        else:
            self.android_container = None

    def _resolve_paths(self) -> None:
        """
        Finalize any path logic specific to Chrome OS. Some directories
        may be restricted or read-only, depending on dev mode settings.
        """
        if self.working_directory is None:
            self.working_directory = '/mnt/stateful_partition/devlib-target'
