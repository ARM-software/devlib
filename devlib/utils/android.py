#    Copyright 2013-2025 ARM Limited
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
Utility functions for working with Android devices through adb.
"""
# pylint: disable=E1103
import functools
import glob
import os
import pexpect
import re
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
import threading

from collections import defaultdict
from io import StringIO
from lxml import etree
from shlex import quote

from devlib.exception import (TargetTransientError, TargetStableError, HostError,
                              TargetTransientCalledProcessError, TargetStableCalledProcessError, AdbRootError)
from devlib.utils.misc import check_output, which, ABI_MAP, redirect_streams, get_subprocess, get_logger
from devlib.connection import (ConnectionBase, AdbBackgroundCommand,
                               PopenTransferHandle)

from typing import (Optional, TYPE_CHECKING, cast, Tuple, Union,
                    List, DefaultDict, Dict, Iterator,
                    Match, Callable)
from collections.abc import Generator
from typing_extensions import Required, TypedDict, Literal
if TYPE_CHECKING:
    from devlib.utils.annotation_helpers import SubprocessCommand
    from threading import Lock
    from lxml.etree import _ElementTree, _Element, XMLParser
    from devlib.platform import Platform
    from subprocess import Popen, CompletedProcess
    from devlib.target import AndroidTarget
    from io import TextIOWrapper
    from tempfile import _TemporaryFileWrapper
    from pexpect import spawn

PartsType = Tuple[Union[str, Tuple[str, ...]], ...]

logger = get_logger('android')

MAX_ATTEMPTS = 5
AM_START_ERROR = re.compile(r"Error: Activity.*")
AAPT_BADGING_OUTPUT = re.compile(r"no dump ((file)|(apk)) specified", re.IGNORECASE)

# See:
# http://developer.android.com/guide/topics/manifest/uses-sdk-element.html#ApiLevels
ANDROID_VERSION_MAP: Dict[int, str] = {
    29: 'Q',
    28: 'PIE',
    27: 'OREO_MR1',
    26: 'OREO',
    25: 'NOUGAT_MR1',
    24: 'NOUGAT',
    23: 'MARSHMALLOW',
    22: 'LOLLYPOP_MR1',
    21: 'LOLLYPOP',
    20: 'KITKAT_WATCH',
    19: 'KITKAT',
    18: 'JELLY_BEAN_MR2',
    17: 'JELLY_BEAN_MR1',
    16: 'JELLY_BEAN',
    15: 'ICE_CREAM_SANDWICH_MR1',
    14: 'ICE_CREAM_SANDWICH',
    13: 'HONEYCOMB_MR2',
    12: 'HONEYCOMB_MR1',
    11: 'HONEYCOMB',
    10: 'GINGERBREAD_MR1',
    9: 'GINGERBREAD',
    8: 'FROYO',
    7: 'ECLAIR_MR1',
    6: 'ECLAIR_0_1',
    5: 'ECLAIR',
    4: 'DONUT',
    3: 'CUPCAKE',
    2: 'BASE_1_1',
    1: 'BASE',
}

# See https://developer.android.com/reference/android/content/Intent.html#setFlags(int)
INTENT_FLAGS: Dict[str, int] = {
    'ACTIVITY_NEW_TASK': 0x10000000,
    'ACTIVITY_CLEAR_TASK': 0x00008000
}


class AndroidProperties(object):
    """
    Represents Android system properties as reported by the ``getprop`` command.
    Allows easy retrieval of property values.

    :param text: Full string output from ``adb shell getprop`` (or similar).
    """
    def __init__(self, text: str):
        self._properties: Dict[str, str] = {}
        self.parse(text)

    def parse(self, text: str) -> None:
        """
        Parse the output text and update the internal property dictionary.

        :param text: String containing the property lines.
        """
        self._properties = dict(re.findall(r'\[(.*?)\]:\s+\[(.*?)\]', text))

    def iteritems(self) -> Iterator[Tuple[str, str]]:
        """
        Return an iterator of (property_key, property_value) pairs.

        :returns: An iterator of tuples like (key, value).
        """
        return iter(self._properties.items())

    def __iter__(self):
        """
        Iterate over the property keys.
        """
        return iter(self._properties)

    def __getattr__(self, name: str):
        """
        Return a property value by attribute-style lookup.
        Defaults to None if the property is missing.
        """
        return self._properties.get(name)

    __getitem__ = __getattr__


class AdbDevice(object):
    """
    Represents a single device as seen by ``adb devices`` (usually a USB or IP
    device).

    :param name: The serial number or identifier of the device.
    :param status: The device status, e.g. "device", "offline", or "unauthorized".
    """
    def __init__(self, name: str, status: str):
        self.name = name
        self.status = status

    # replace __cmp__ of python 2 with explicit comparison methods
    # of python 3
    def __lt__(self, other: Union['AdbDevice', str]) -> bool:
        """
        Compare this device's name with another device or string for ordering.
        """
        if isinstance(other, AdbDevice):
            return self.name < other.name
        return self.name < other

    def __eq__(self, other: object) -> bool:
        """
        Check if this device's name matches another device's name or a string.
        """
        if isinstance(other, AdbDevice):
            return self.name == other.name
        return self.name == other

    def __le__(self, other: Union['AdbDevice', str]) -> bool:
        """
        Test if this device's name is <= another device/string.
        """
        return self < other or self == other

    def __gt__(self, other: Union['AdbDevice', str]) -> bool:
        """
        Test if this device's name is > another device/string.
        """
        return not self <= other

    def __ge__(self, other: Union['AdbDevice', str]) -> bool:
        """
        Test if this device's name is >= another device/string.
        """
        return not self < other

    def __ne__(self, other: object) -> bool:
        """
        Invert the __eq__ comparison.
        """
        return not self == other

    def __str__(self) -> str:
        """
        Return a string representation of this device for debugging.
        """
        return 'AdbDevice({}, {})'.format(self.name, self.status)

    __repr__ = __str__


class BuildToolsInfo(TypedDict, total=False):
    """
    Typed dictionary capturing build tools info.

    :param build_tools: The path to the build-tools directory.
    :param aapt: Path to the aapt or aapt2 binary.
    :param aapt_version: Integer 1 or 2 indicating which aapt is used.
    """
    build_tools: Required[Optional[str]]
    aapt: Required[Optional[str]]
    aapt_version: Required[Optional[int]]


class Android_Env_Type(TypedDict, total=False):
    """
    Typed dictionary representing environment paths for Android tools.

    :param android_home: ANDROID_HOME path, if set.
    :param platform_tools: Path to the 'platform-tools' directory containing adb/fastboot.
    :param adb: Path to the 'adb' executable.
    :param fastboot: Path to the 'fastboot' executable.
    :param build_tools: Path to the 'build-tools' directory if available.
    :param aapt: Path to aapt or aapt2, if found.
    :param aapt_version: 1 or 2 indicating which aapt variant is used.
    """
    android_home: Required[Optional[str]]
    platform_tools: Required[str]
    adb: Required[str]
    fastboot: Required[str]
    build_tools: Required[Optional[str]]
    aapt: Required[Optional[str]]
    aapt_version: Required[Optional[int]]


Android_Env_TypeKeys = Union[Literal['android_home'],
                             Literal['platform_tools'],
                             Literal['adb'],
                             Literal['fastboot'],
                             Literal['build_tools'],
                             Literal['aapt'],
                             Literal['aapt_version']]


class ApkInfo(object):
    """
    Extracts and stores metadata about an APK, including package name, version,
    supported ABIs, permissions, etc. The parsing relies on the 'aapt' or 'aapt2'
    command from Android build-tools.

    :param path: Optional path to the APK file on the host. If provided, it is
        immediately parsed.
    """
    version_regex = re.compile(r"name='(?P<name>[^']+)' versionCode='(?P<vcode>[^']+)' versionName='(?P<vname>[^']+)'")
    name_regex = re.compile(r"name='(?P<name>[^']+)'")
    permission_regex = re.compile(r"name='(?P<permission>[^']+)'")
    activity_regex = re.compile(r'\s*A:\s*android:name\(0x\d+\)=".(?P<name>\w+)"')

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.package: Optional[str] = None
        self.activity: Optional[str] = None
        self.label: Optional[str] = None
        self.version_name: Optional[str] = None
        self.version_code: Optional[str] = None
        self.native_code: Optional[List[str]] = None
        self.permissions: List[str] = []
        self._apk_path: Optional[str] = None
        self._activities: Optional[List[str]] = None
        self._methods: Optional[List[Tuple[str, str]]] = None
        self._aapt: str = cast(str, _ANDROID_ENV.get_env('aapt'))
        self._aapt_version: int = cast(int, _ANDROID_ENV.get_env('aapt_version'))

        if path:
            self.parse(path)

    # pylint: disable=too-many-branches
    def parse(self, apk_path: str) -> None:
        """
        Parse the given APK file with the aapt or aapt2 utility, retrieving
        metadata such as package name, version, and permissions.

        :param apk_path: The path to the APK file on the host system.
        :raises HostError: If aapt fails to run or returns an error message.
        """
        output: str = self._run([self._aapt, 'dump', 'badging', apk_path])
        for line in output.split('\n'):
            if line.startswith('application-label:'):
                self.label = line.split(':')[1].strip().replace('\'', '')
            elif line.startswith('package:'):
                match = self.version_regex.search(line)
                if match:
                    self.package = match.group('name')
                    self.version_code = match.group('vcode')
                    self.version_name = match.group('vname')
            elif line.startswith('launchable-activity:'):
                match = self.name_regex.search(line)
                self.activity = match.group('name') if match else None
            elif line.startswith('native-code'):
                apk_abis: List[str] = [entry.strip() for entry in line.split(':')[1].split("'") if entry.strip()]
                mapped_abis: List[str] = []
                for apk_abi in apk_abis:
                    found: bool = False
                    for abi, architectures in ABI_MAP.items():
                        if apk_abi in architectures:
                            mapped_abis.append(abi)
                            found = True
                            break
                    if not found:
                        mapped_abis.append(apk_abi)
                self.native_code = mapped_abis
            elif line.startswith('uses-permission:'):
                match = self.permission_regex.search(line)
                if match:
                    self.permissions.append(match.group('permission'))
            else:
                pass  # not interested

        self._apk_path = apk_path
        self._activities = None
        self._methods = None

    @property
    def activities(self) -> List[str]:
        """
        Return a list of activity names declared in this APK.

        :returns: A list of activity names found in AndroidManifest.xml.
        """
        if self._activities is None:
            cmd: List[str] = [self._aapt, 'dump', 'xmltree', self._apk_path if self._apk_path else '']
            if self._aapt_version == 2:
                cmd += ['--file']
            cmd += ['AndroidManifest.xml']
            matched_activities: Iterator[Match[str]] = self.activity_regex.finditer(self._run(cmd))
            self._activities = [m.group('name') for m in matched_activities]
        return self._activities

    @property
    def methods(self) -> Optional[List[Tuple[str, str]]]:
        """
        Return a list of (method_name, class_name) pairs, if any can be extracted
        by dexdump. If no classes.dex is found or an error occurs, returns an empty list.

        :returns: A list of (method_name, class_name) tuples, or None if not parsed yet.
        """
        if self._methods is None:
            # Only try to extract once
            self._methods = []
            with tempfile.TemporaryDirectory() as tmp_dir:
                if self._apk_path:
                    with zipfile.ZipFile(self._apk_path, 'r') as z:
                        try:
                            extracted: str = z.extract('classes.dex', tmp_dir)
                        except KeyError:
                            return []
                dexdump: str = os.path.join(os.path.dirname(self._aapt), 'dexdump')
                command: List[str] = [dexdump, '-l', 'xml', extracted]
                dump: str = self._run(command)

            # Dexdump from build tools v30.0.X does not seem to produce
            # valid xml from certain APKs so ignore errors and attempt to recover.
            parser: XMLParser = etree.XMLParser(encoding='utf-8', recover=True)
            xml_tree: _ElementTree = etree.parse(StringIO(dump), parser)

            package: List[_Element] = []
            for i in xml_tree.iter('package'):
                if i.attrib['name'] == self.package:
                    package.append(i)

            for elem in package:
                self._methods.extend([(meth.attrib['name'], klass.attrib['name'])
                                      for klass in elem.iter('class')
                                      for meth in klass.iter('method')])
        return self._methods

    def _run(self, command: List[str]) -> str:
        """
        Execute a local shell command (e.g., aapt) and return its output as a string.

        :param command: List of command arguments to run.
        :returns: Combined stdout+stderr as a decoded string.
        :raises HostError: If the command fails or returns a nonzero exit code.
        """
        logger.debug(' '.join(command))
        try:
            output_tmp: bytes = subprocess.check_output(command, stderr=subprocess.STDOUT)
            output: str = output_tmp.decode(sys.stdout.encoding or 'utf-8', 'replace')
        except subprocess.CalledProcessError as e:
            raise HostError('Error while running "{}":\n{}'
                            .format(command, e.output))
        return output


class AdbConnection(ConnectionBase):
    """
    A connection to an android device via ``adb`` (Android Debug Bridge).
    ``adb`` is part of the Android SDK (though stand-alone versions are also
    available).

    :param device: The name of the adb device. This is usually a unique hex
                   string for USB-connected devices, or an ip address/port
                   combination. To see connected devices, you can run ``adb
                   devices`` on the host.
    :param timeout: Connection timeout in seconds. If a connection to the device
                    is not established within this period, :class:`HostError`
                    is raised.
    :param platform: An optional Platform object describing hardware aspects.
    :param adb_server: Allows specifying the address of the adb server to use.
    :param adb_port: If specified, connect to a custom adb server port.
    :param adb_as_root: Specify whether the adb server should be restarted in root mode.
    :param connection_attempts: Specify how many connection attempts, 10 seconds
                                apart, should be attempted to connect to the device.
                                Defaults to 5.
    :param poll_transfers: Specify whether file transfers should be polled. Polling
                           monitors the progress of file transfers and periodically
                           checks whether they have stalled, attempting to cancel
                           the transfers prematurely if so.
    :param start_transfer_poll_delay: If transfers are polled, specify the length of
                                      time after a transfer has started before polling
                                      should start.
    :param total_transfer_timeout: If transfers are polled, specify the total amount of time
                                   to elapse before the transfer is cancelled, regardless
                                   of its activity.
    :param transfer_poll_period: If transfers are polled, specify the period at which
                                 the transfers are sampled for activity. Too small values
                                 may cause the destination size to appear the same over
                                 one or more sample periods, causing improper transfer
                                 cancellation.

    :raises AdbRootError: If root mode is requested but multiple connections are active or device does not allow it.
    :raises HostError: If the device fails to connect or is invalid.
    """
    # maintains the count of parallel active connections to a device, so that
    # adb disconnect is not invoked untill all connections are closed
    active_connections: Tuple['Lock', DefaultDict[str, int]] = (threading.Lock(), defaultdict(int))
    # Track connected as root status per device
    _connected_as_root: DefaultDict[str, Optional[bool]] = defaultdict(lambda: None)
    default_timeout: int = 10
    ls_command: str = 'ls'
    su_cmd: str = 'su -c {}'

    @property
    def name(self) -> str:
        """
        :returns: The device serial number or IP:port used by this connection.
        """
        return self.device

    @property
    def connected_as_root(self) -> Optional[bool]:
        """
        Check if the current connection is effectively root on the device.

        :returns: True if root, False if not, or None if undetermined.
        """
        if self._connected_as_root[self.device] is None:
            result = self.execute('id')
            self._connected_as_root[self.device] = 'uid=0(' in result
        return self._connected_as_root[self.device]

    @connected_as_root.setter
    def connected_as_root(self, state: Optional[bool]) -> None:
        """
        Manually set the known state of root usage on this device connection.

        :param state: True if connected as root, False if not, None to reset.
        """
        self._connected_as_root[self.device] = state

    # pylint: disable=unused-argument
    def __init__(
        self,
        device: Optional[str] = None,
        timeout: Optional[int] = None,
        platform: Optional['Platform'] = None,
        adb_server: Optional[str] = None,
        adb_port: Optional[int] = None,
        adb_as_root: bool = False,
        connection_attempts: int = MAX_ATTEMPTS,

        poll_transfers: bool = False,
        start_transfer_poll_delay: int = 30,
        total_transfer_timeout: int = 3600,
        transfer_poll_period: int = 30,
    ):
        super().__init__(
            poll_transfers=poll_transfers,
            start_transfer_poll_delay=start_transfer_poll_delay,
            total_transfer_timeout=total_transfer_timeout,
            transfer_poll_period=transfer_poll_period,
        )

        self.logger.debug('server=%s port=%s device=%s as_root=%s',
                          adb_server, adb_port, device, adb_as_root)

        self.timeout = timeout if timeout is not None else self.default_timeout
        if device is None:
            device = adb_get_device(timeout=timeout, adb_server=adb_server, adb_port=adb_port)
        self.device = device
        self.adb_server = adb_server
        self.adb_port = adb_port
        self.adb_as_root = adb_as_root
        self._restore_to_adb_root = False
        lock, nr_active = AdbConnection.active_connections
        with lock:
            nr_active[self.device] += 1

        if self.adb_as_root:
            try:
                self._restore_to_adb_root = self._adb_root(enable=True)
            # Exception will be raised if we are not the only connection
            # active. adb_root() requires restarting the server, which is not
            # acceptable if other connections are active and can apparently
            # lead to commands hanging forever in some situations.
            except AdbRootError:
                pass
        adb_connect(self.device, adb_server=self.adb_server, adb_port=self.adb_port, attempts=connection_attempts)
        self._setup_ls()
        self._setup_su()

    def push(self, sources: List[str], dest: str,
             timeout: Optional[int] = None) -> None:
        """
        Upload (push) one or more files/directories from the host to the device.

        :param sources: Paths on the host system to be pushed.
        :param dest: Target path on the device. If multiple sources, dest should be a dir.
        :param timeout: Max time in seconds for each file push. If exceeded, an error is raised.
        """
        return self._push_pull('push', sources, dest, timeout)

    def pull(self, sources: List[str], dest: str,
             timeout: Optional[int] = None) -> None:
        """
        Download (pull) one or more files/directories from the device to the host.

        :param sources: Paths on the device to be pulled.
        :param dest: Destination path on the host.
        :param timeout: Max time in seconds for each file. If exceeded, an error is raised.
        """
        return self._push_pull('pull', sources, dest, timeout)

    def _push_pull(self, action: Union[Literal['push'], Literal['pull']],
                   sources: List[str], dest: str, timeout: Optional[int]) -> None:
        """
        Internal helper that runs 'adb push' or 'adb pull' with optional timeouts
        and transfer polling.
        """
        sourcesList: List[str] = list(sources)
        pathsList: List[str] = sourcesList + [dest]

        # Quote twice to avoid expansion by host shell, then ADB globbing
        do_quote: Callable[[str], str] = lambda x: quote(glob.escape(x))
        paths: str = ' '.join(map(do_quote, pathsList))

        command = "{} {}".format(action, paths)
        if timeout:
            adb_command(self.device, command, timeout=timeout, adb_server=self.adb_server, adb_port=self.adb_port)
        else:
            popen = adb_command_popen(
                device=self.device,
                conn=self,
                command=command,
                adb_server=self.adb_server,
                adb_port=self.adb_port,
            )

            handle = PopenTransferHandle(
                manager=self.transfer_manager,
                popen=popen,
                dest=dest,
                direction=action
            )
            with popen, self.transfer_manager.manage(sources, dest, action, handle):
                popen.communicate()

    # pylint: disable=unused-argument
    def execute(self, command: 'SubprocessCommand', timeout: Optional[int] = None,
                check_exit_code: bool = False, as_root: Optional[bool] = False,
                strip_colors: bool = True, will_succeed: bool = False) -> str:
        """
        Execute a command on the device via ``adb shell``.

        :param command: The command line to run (string or SubprocessCommand).
        :param timeout: Time in seconds before forcibly terminating the command. None for no limit.
        :param check_exit_code: If True, raise an error if the command's exit code != 0.
        :param as_root: If True, attempt to run it as root if available.
        :param strip_colors: If True, strip any ANSI colors (unused in this method).
        :param will_succeed: If True, treat an error as transient rather than stable.
        :returns: The command's output (combined stdout+stderr).
        :raises TargetTransientCalledProcessError: If the command fails but is flagged as transient.
        :raises TargetStableCalledProcessError: If the command fails in a stable (non-transient) way.
        :raises TargetStableError: If there's a stable device/command error.
        """
        if as_root and self.connected_as_root:
            as_root = False
        try:
            return adb_shell(self.device, command, timeout, check_exit_code,
                             as_root, adb_server=self.adb_server, adb_port=self.adb_port, su_cmd=self.su_cmd)
        except subprocess.CalledProcessError as e:
            cls = TargetTransientCalledProcessError if will_succeed else TargetStableCalledProcessError
            raise cls(
                e.returncode,
                command,
                e.output,
                e.stderr,
            )
        except TargetStableError as e:
            if will_succeed:
                raise TargetTransientError(e)
            else:
                raise

    def background(self, command: 'SubprocessCommand', stdout: int = subprocess.PIPE,
                   stderr: int = subprocess.PIPE, as_root: Optional[bool] = False) -> AdbBackgroundCommand:
        """
        Launch a background command via adb shell and return a handle to manage it.

        :param command: The command to run on the device.
        :param stderr: File descriptor or special value for stderr.
        :param as_root: If True, attempt to run the command as root.
        :returns: A handle to the background command.

        .. note:: This **will block the connection** until the command completes.
        """
        if as_root and self.connected_as_root:
            as_root = False
        bg_cmd: AdbBackgroundCommand = self._background(command, stdout, stderr, as_root)
        return bg_cmd

    def _background(self, command: 'SubprocessCommand', stdout: int,
                    stderr: int, as_root: Optional[bool]) -> AdbBackgroundCommand:
        """
        Helper method to run a background shell command via adb.

        :param command: Shell command to run.
        :param stdout: Location for stdout writes.
        :param stderr: Location for stderr writes.
        :param as_root: If True, run as root if possible.
        :returns: An AdbBackgroundCommand object.
        :raises Exception: If PID detection fails or no valid device is set.
        """
        def make_init_kwargs(command):
            adb_popen, pid = adb_background_shell(self, command, stdout, stderr, as_root)
            return dict(
                adb_popen=adb_popen,
                pid=pid,
            )

        bg_cmd = AdbBackgroundCommand.from_factory(
            conn=self,
            cmd=command,
            as_root=as_root,
            make_init_kwargs=make_init_kwargs,
        )
        return bg_cmd

    def _close(self) -> None:
        """
        Close the connection to the device. The :class:`Connection` object should not
        be used after this method is called. There is no way to reopen a previously
        closed connection, a new connection object should be created instead.
        """
        if not hasattr(AdbConnection, "active_connections") or AdbConnection.active_connections is None:
            return  # Prevents AttributeError when closing a non-existent connection

        lock, nr_active = AdbConnection.active_connections
        with lock:
            nr_active[self.device] -= 1
            disconnect = nr_active[self.device] <= 0
            if disconnect:
                del nr_active[self.device]

        if disconnect:
            if self.adb_as_root:
                self.adb_root(enable=self._restore_to_adb_root)
            adb_disconnect(self.device, self.adb_server, self.adb_port)

    def cancel_running_command(self) -> None:
        """
        Cancel a running command (previously started with :func:`background`) and free up the connection.
        It is valid to call this if the command has already terminated (or if no
        command was issued), in which case this is a no-op.
        """
        # adbd multiplexes commands so that they don't interfer with each
        # other, so there is no need to explicitly cancel a running command
        # before the next one can be issued.
        pass

    def adb_root(self, enable=True):
        """
        Enable or disable root mode for this device connection.

        :param enable: True to enable root, False to unroot.
        :raises AdbRootError: If multiple connections are active or device disallows root.
        """
        self._adb_root(enable=enable)

    def _adb_root(self, enable):
        lock, nr_active = AdbConnection.active_connections
        with lock:
            can_root = nr_active[self.device] <= 1

        if not can_root:
            raise AdbRootError('Can only restart adb server if no other connection is active')

        def is_rooted(out):
            return 'adbd is already running as root' in out

        cmd = 'root' if enable else 'unroot'
        try:
            output = adb_command(self.device, cmd, timeout=30, adb_server=self.adb_server, adb_port=self.adb_port)
        except subprocess.CalledProcessError as e:
            was_rooted = is_rooted(e.output)
            # Ignore if we're already root
            if not was_rooted:
                raise AdbRootError(str(e)) from e
        else:
            was_rooted = is_rooted(output)
            # Check separately as this does not cause a error exit code.
            if 'cannot run as root in production builds' in output:
                raise AdbRootError(output)
        AdbConnection._connected_as_root[self.device] = enable
        return was_rooted

    def wait_for_device(self, timeout: Optional[int] = 30) -> None:
        """
        Block until the device is available for commands, up to a specified timeout.

        :param timeout: Time in seconds before giving up.
        """
        adb_command(self.device, 'wait-for-device', timeout, self.adb_server, self.adb_port)

    def reboot_bootloader(self, timeout: int = 30) -> None:
        """
        Reboot the device into its bootloader (fastboot) mode.

        :param timeout: Seconds to wait for the reboot command to be accepted.
        """
        adb_command(self.device, 'reboot-bootloader', timeout, self.adb_server, self.adb_port)

    # Again, we need to handle boards where the default output format from ls is
    # single column *and* boards where the default output is multi-column.
    # We need to do this purely because the '-1' option causes errors on older
    # versions of the ls tool in Android pre-v7.
    def _setup_ls(self) -> None:
        """
        Detect whether 'ls -1' is supported, falling back to plain 'ls' on older devices.
        """
        command = "shell '(ls -1); echo \"\n$?\"'"
        try:
            output = adb_command(self.device, command, timeout=self.timeout, adb_server=self.adb_server, adb_port=self.adb_port)
        except subprocess.CalledProcessError as e:
            raise HostError(
                'Failed to set up ls command on Android device. Output:\n' + e.output)
        lines: List[str] = output.splitlines()
        retval: str = lines[-1].strip()
        if int(retval) == 0:
            self.ls_command = 'ls -1'
        else:
            self.ls_command = 'ls'
        logger.debug("ls command is set to {}".format(self.ls_command))

    def _setup_su(self) -> None:
        """
        Attempt to confirm if 'su -c' is required or a simpler 'su' approach works.
        """
        # Already root, nothing to do
        if self.connected_as_root:
            return
        try:
            # Try the new style of invoking `su`
            self.execute('ls', timeout=self.timeout, as_root=True,
                         check_exit_code=True)
        # If failure assume either old style or unrooted. Here we will assume
        # old style and root status will be verified later.
        except (TargetStableError, TargetTransientError, TimeoutError):
            self.su_cmd = 'echo {} | su'
        logger.debug("su command is set to {}".format(quote(self.su_cmd)))


def fastboot_command(command: str, timeout: Optional[int] = None,
                     device: Optional[str] = None) -> str:
    """
    Execute a fastboot command, optionally targeted at a specific device.

    :param command: The fastboot subcommand (e.g. 'devices', 'flash').
    :param timeout: Time in seconds before the command fails.
    :param device: Fastboot device name. If None, assumes a single device or environment default.
    :returns: Combined stdout+stderr output from the fastboot command.
    :raises HostError: If the command fails or returns an error.
    """
    target: str = '-s {}'.format(quote(device)) if device else ''
    bin_: str = cast(str, _ANDROID_ENV.get_env('fastboot'))
    full_command: str = f'{bin_} {target} {command}'
    logger.debug(full_command)
    output, _ = check_output(full_command, timeout, shell=True)
    return output


def fastboot_flash_partition(partition: str, path_to_image: str) -> None:
    """
    Execute 'fastboot flash <partition> <path_to_image>' to flash a file
    onto a specific partition of the device.

    :param partition: The device partition to flash (e.g. "boot", "system").
    :param path_to_image: Full path to the image file on the host.
    :raises HostError: If fastboot fails or device is not in fastboot mode.
    """
    command: str = 'flash {} {}'.format(quote(partition), quote(path_to_image))
    fastboot_command(command)


def adb_get_device(timeout: Optional[int] = None, adb_server: Optional[str] = None,
                   adb_port: Optional[int] = None) -> str:
    """
    Attempt to auto-detect a single connected device. If multiple or none are found,
    raise an error.

    :param timeout: Maximum time to wait for device detection, or None for no limit.
    :param adb_server: Optional custom server host.
    :param adb_port: Optional custom server port.
    :returns: The device serial number or IP:port if exactly one device is found.
    :raises HostError: If zero or more than one devices are connected.
    """
    # TODO this is a hacky way to issue a adb command to all listed devices

    # Ensure server is started so the 'daemon started successfully' message
    # doesn't confuse the parsing below
    adb_command(None, 'start-server', adb_server=adb_server, adb_port=adb_port)

    # The output of calling adb devices consists of a heading line then
    # a list of the devices sperated by new line
    # The last line is a blank new line. in otherwords, if there is a device found
    # then the output length is 2 + (1 for each device)
    start: float = time.time()
    while True:
        output: List[str] = adb_command(None, "devices", adb_server=adb_server, adb_port=adb_port).splitlines()  # pylint: disable=E1103
        output_length: int = len(output)
        if output_length == 3:
            # output[1] is the 2nd line in the output which has the device name
            # Splitting the line by '\t' gives a list of two indexes, which has
            # device serial in 0 number and device type in 1.
            return output[1].split('\t')[0]
        elif output_length > 3:
            message: str = '{} Android devices found; either explicitly specify ' +\
                           'the device you want, or make sure only one is connected.'
            raise HostError(message.format(output_length - 2))
        else:
            if timeout is not None and timeout < time.time() - start:
                raise HostError('No device is connected and available')
            time.sleep(1)


def adb_connect(device: Optional[str], timeout: Optional[int] = None,
                attempts: int = MAX_ATTEMPTS, adb_server: Optional[str] = None,
                adb_port: Optional[int] = None) -> None:
    """
    Connect to an ADB-over-IP device or ensure a USB device is listed. Re-tries
    until success or attempts are exhausted.

    :param device: The device name, if "." in it, assumes IP-based device.
    :param timeout: Time in seconds for each attempt before giving up.
    :param attempts: Number of times to retry connecting 10 seconds apart.
    :param adb_server: Optional ADB server host.
    :param adb_port: Optional ADB server port.
    :raises HostError: If connection fails after all attempts.
    """
    tries: int = 0
    output: Optional[str] = None
    while tries <= attempts:
        tries += 1
        if device:
            if "." in device:  # Connect is required only for ADB-over-IP
                # ADB does not automatically remove a network device from it's
                # devices list when the connection is broken by the remote, so the
                # adb connection may have gone "stale", resulting in adb blocking
                # indefinitely when making calls to the device. To avoid this,
                # always disconnect first.
                adb_disconnect(device, adb_server, adb_port)
                adb_cmd: str = get_adb_command(None, 'connect', adb_server, adb_port)
                command: str = '{} {}'.format(adb_cmd, quote(device))
                logger.debug(command)
                output, _ = check_output(command, shell=True, timeout=timeout)
        if _ping(device, adb_server, adb_port):
            break
        time.sleep(10)
    else:  # did not connect to the device
        message: str = f'Could not connect to {device or "a device"} at {adb_server}:{adb_port}'
        if output:
            message += f'; got: {output}'
        raise HostError(message)


def adb_disconnect(device: Optional[str], adb_server: Optional[str] = None,
                   adb_port: Optional[int] = None) -> None:
    """
    Issue an 'adb disconnect' for the specified device, if relevant.

    :param device: Device serial or IP:port. If None or no IP in the name, no action is taken.
    :param adb_server: Custom ADB server host if used.
    :param adb_port: Custom ADB server port if used.
    """
    if not device:
        return
    if ":" in device and device in adb_list_devices(adb_server, adb_port):
        adb_cmd: str = get_adb_command(None, 'disconnect', adb_server, adb_port)
        command: str = "{} {}".format(adb_cmd, device)
        logger.debug(command)
        retval: int = subprocess.call(command, stdout=subprocess.DEVNULL, shell=True)
        if retval:
            raise TargetTransientError('"{}" returned {}'.format(command, retval))


def _ping(device: Optional[str], adb_server: Optional[str] = None,
          adb_port: Optional[int] = None) -> bool:
    """
    Ping the specified device by issuing a trivial command (ls /data/local/tmp).
    If it fails, the device is presumably unreachable or offline.

    :param device: The device name or IP:port.
    :param adb_server: ADB server host, if any.
    :param adb_port: ADB server port, if any.
    :returns: True if the device responded, otherwise False.
    """
    adb_cmd: str = get_adb_command(device, 'shell', adb_server, adb_port)
    command: str = "{} {}".format(adb_cmd, quote('ls /data/local/tmp > /dev/null'))
    logger.debug(command)
    try:
        subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        logger.debug(f'ADB ping failed: {e.stdout}')
        return False
    else:
        return True


# pylint: disable=too-many-locals
def adb_shell(device: str, command: 'SubprocessCommand', timeout: Optional[int] = None,
              check_exit_code: bool = False, as_root: Optional[bool] = False, adb_server: Optional[str] = None,
              adb_port:Optional[int]=None, su_cmd:str='su -c {}') -> str:  # NOQA
    """
    Run a command in 'adb shell' mode, capturing both stdout/stderr. Uses a technique
    to capture the actual command's exit code so that we can detect non-zero exit
    reliably on older ADB combos.

    :param device: The device serial or IP:port.
    :param command: The command line to run inside 'adb shell'.
    :param timeout: Time in seconds to wait for the command, or None for no limit.
    :param check_exit_code: If True, raise an error if the command exit code is nonzero.
    :param as_root: If True, prepend an su command to run as root if supported.
    :param adb_server: Optional custom adb server IP/name.
    :param adb_port: Optional custom adb server port.
    :param su_cmd: Command template to wrap as root, e.g. 'su -c {}'.
    :returns: The combined stdout from the command (minus the exit code).
    :raises TargetStableError: If there's an error with the command or exit code extraction fails.
    """
    # On older combinations of ADB/Android versions, the adb host command always
    # exits with 0 if it was able to run the command on the target, even if the
    # command failed (https://code.google.com/p/android/issues/detail?id=3254).
    # Homogenise this behaviour by running the command then echoing the exit
    # code of the executed command itself.
    command = r'({}); echo "\n$?"'.format(cast(str, command))
    command = su_cmd.format(quote(command)) if as_root else command
    command = ('shell', command)
    parts, env = _get_adb_parts(command, device, adb_server, adb_port, quote_adb=False)
    env = {**os.environ, **env}

    logger.debug(' '.join(quote(cast(str, part)) for part in parts))
    try:
        raw_output, error = check_output(cast('SubprocessCommand', parts), timeout, shell=False, env=env)
    except subprocess.CalledProcessError as e:
        raise TargetStableError(str(e))

    if raw_output:
        try:
            output, exit_code, _ = raw_output.replace('\r\n', '\n').replace('\r', '\n').rsplit('\n', 2)
        except ValueError:
            exit_code, _ = raw_output.replace('\r\n', '\n').replace('\r', '\n').rsplit('\n', 1)
            output = ''
    else:  # raw_output is empty
        exit_code = '969696'  # just because
        output = ''

    if check_exit_code:
        exit_code = exit_code.strip()
        re_search = AM_START_ERROR.findall(output)
        if exit_code.isdigit():
            exit_code_i = int(exit_code)
            if exit_code_i:
                raise subprocess.CalledProcessError(
                    exit_code_i,
                    command,
                    output,
                    error,
                )

            elif re_search:
                message = 'Could not start activity; got the following:\n{}'
                raise TargetStableError(message.format(re_search[0]))
        else:  # not all digits
            if re_search:
                message = 'Could not start activity; got the following:\n{}'
                raise TargetStableError(message.format(re_search[0]))
            else:
                message = 'adb has returned early; did not get an exit code. '\
                          'Was kill-server invoked?\nOUTPUT:\n-----\n{}\n'\
                          '-----\nSTDERR:\n-----\n{}\n-----'
                raise TargetTransientError(message.format(raw_output, error))

    return '\n'.join(x for x in (output, error) if x)


def adb_background_shell(conn: AdbConnection, command: 'SubprocessCommand',
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         as_root: Optional[bool] = False) -> Tuple['Popen', int]:
    """
    Run a command in the background on the device via ADB shell, returning a Popen
    object and an integer PID. This approach uses SIGSTOP to freeze the shell
    while the PID is identified.

    :param conn: The AdbConnection managing the device.
    :param command: A shell command to run in the background.
    :param stdout: File descriptor for stdout, default is pipe.
    :param stderr: File descriptor for stderr, default is pipe.
    :param as_root: If True, attempt to run under su if root is available.
    :returns: A tuple of (popen_obj, pid).
    :raises TargetTransientError: If the PID cannot be identified after retries.
    """
    device = conn.device
    adb_server = conn.adb_server
    adb_port = conn.adb_port
    busybox = conn.busybox
    orig_command = command

    stdout, stderr, command = redirect_streams(stdout, stderr, command)
    if as_root:
        command = f'{busybox} printf "%s" {quote(cast(str, command))} | su'

    def with_uuid(cmd: str) -> Tuple[str, str]:
        # Attach a unique UUID to the command line so it can be looked for
        # without any ambiguity with ps
        uuid_: str = uuid.uuid4().hex
        # Unset the var, since not all connection types set it. This will avoid
        # anyone depending on that value.
        cmd = f'DEVLIB_CMD_UUID={uuid_}; unset DEVLIB_CMD_UUID; {cmd}'
        # Ensure we have an sh -c layer so that the UUID will appear on the
        # command line parameters of at least one command.
        cmd = f'exec {busybox} sh -c {quote(cmd)}'
        return (uuid_, cmd)

    # Freeze the command with SIGSTOP to avoid racing with PID detection.
    command = f"{busybox} kill -STOP $$ && exec {busybox} sh -c {quote(cast(str, command))}"
    command_uuid, command = with_uuid(command)

    adb_cmd: str = get_adb_command(device, 'shell', adb_server, adb_port)
    full_command: str = f'{adb_cmd} {quote(cast(str, command))}'
    logger.debug(full_command)
    p: 'Popen' = subprocess.Popen(full_command, stdout=stdout, stderr=stderr, stdin=subprocess.PIPE, shell=True)

    # Out of band PID lookup, to avoid conflicting needs with stdout redirection
    grep_cmd: str = f'{busybox} grep {quote(command_uuid)}'
    # Find the PID and release the blocked background command with SIGCONT.
    # We get multiple PIDs:
    # * One from the grep command itself, but we remove it with another grep command.
    # * One for each sh -c layer in the command itself.
    #
    # For each of the parent layer, we issue SIGCONT as it is harmless and
    # avoids having to rely on PID ordering (which could be misleading if PIDs
    # got recycled).
    find_pid: str = f'''pids=$({busybox} ps -A -o pid,args | {grep_cmd} | {busybox} grep -v {quote(grep_cmd)} | {busybox} awk '{{print $1}}') && {busybox} printf "%s" "$pids" && {busybox} kill -CONT $pids'''

    excep: Optional[Exception] = None
    for _ in range(5):
        try:
            pids: str = conn.execute(find_pid, as_root=as_root)
            # We choose the highest PID as the "control" PID. It actually does not
            # really matter which one we pick, as they are all equivalent sh -c layers.
            pid: int = max(map(int, pids.split()))
        except TargetStableError:
            raise
        except Exception as e:
            excep = e
            time.sleep(10e-3)
            continue
        else:
            break
    else:
        raise TargetTransientError(f'Could not detect PID of background command: {cast(str, orig_command)}') from excep

    return (p, pid)


def adb_kill_server(timeout: Optional[int] = 30, adb_server: Optional[str] = None,
                    adb_port: Optional[int] = None) -> None:
    """
    Issue 'adb kill-server' to forcibly shut down the local ADB server.

    :param timeout: Seconds to wait for the command.
    :param adb_server: Optional custom server host.
    :param adb_port: Optional custom server port.
    """
    adb_command(None, 'kill-server', timeout, adb_server, adb_port)


def adb_list_devices(adb_server: Optional[str] = None, adb_port: Optional[int] = None) -> List[AdbDevice]:
    """
    List all devices known to ADB by running 'adb devices'. Each line is parsed
    into an :class:`AdbDevice`.

    :param adb_server: Custom ADB server hostname.
    :param adb_port: Custom ADB server port.
    :returns: A list of AdbDevice objects describing connected devices.
    """
    output: str = adb_command(None, 'devices', adb_server=adb_server, adb_port=adb_port)
    devices: List[AdbDevice] = []
    for line in output.splitlines():
        parts: List[str] = [p.strip() for p in line.split()]
        if len(parts) == 2:
            devices.append(AdbDevice(*parts))
    return devices


def _get_adb_parts(command: Union[Tuple[str], Tuple[str, str]], device: Optional[str] = None,
                   adb_server: Optional[str] = None, adb_port: Optional[int] = None,
                   quote_adb: bool = True) -> Tuple[PartsType, Dict[str, str]]:
    """
    Build a tuple of adb command parts, plus environment variables.

    :param command: A tuple of command parts (like ('shell', 'ls')).
    :param device: The device name or None if no device param used.
    :param adb_server: Host/IP of custom adb server if set.
    :param adb_port: Port of custom adb server if set.
    :param quote_adb: Whether to quote the server/port args.
    :returns: A tuple containing the command parts, plus a dict of env updates.
    """
    _quote = quote if quote_adb else lambda x: x

    parts: PartsType = (
        cast(str, _ANDROID_ENV.get_env('adb')),
        *(('-H', _quote(adb_server)) if adb_server is not None else ()),
        *(('-P', _quote(str(adb_port))) if adb_port is not None else ()),
        *(('-s', _quote(device)) if device is not None else ()),
        *command,
    )
    env: Dict[str, str] = {'LC_ALL': 'C'}
    return (parts, env)


def get_adb_command(device: Optional[str], command: str, adb_server: Optional[str] = None,
                    adb_port: Optional[int] = None) -> str:
    """
    Build a single-string 'adb' command that can be run in a host shell.

    :param device: The device serial or IP:port, or None to skip.
    :param command: The subcommand, e.g. 'shell', 'push', etc.
    :param adb_server: Optional custom server address.
    :param adb_port: Optional custom server port.
    :returns: A fully expanded command string including environment variables for LC_ALL.
    """
    partstemp, envtemp = _get_adb_parts((command,), device, adb_server, adb_port, quote_adb=True)
    env: List[str] = [quote(f'{name}={val}') for name, val in sorted(envtemp.items())]
    parts = [*env, *partstemp]
    return ' '.join(cast(List[str], parts))


def adb_command(device: Optional[str], command: str, timeout: Optional[int] = None,
                adb_server: Optional[str] = None, adb_port: Optional[int] = None) -> str:
    """
    Build and run an 'adb' command synchronously, returning its combined output.

    :param device: Device name, or None if only one or no device is expected.
    :param command: A subcommand or subcommand + arguments (e.g. 'push file /sdcard/').
    :param timeout: Seconds to wait for completion (None for no limit).
    :param adb_server: Custom ADB server host if needed.
    :param adb_port: Custom ADB server port if needed.
    :returns: The command's output as a decoded string.
    :raises HostError: If the command fails or returns non-zero.
    """
    full_command: str = get_adb_command(device, command, adb_server, adb_port)
    logger.debug(full_command)
    output, _ = check_output(full_command, timeout, shell=True)
    return output


def adb_command_popen(device: Optional[str], conn: AdbConnection, command: str,
                      adb_server: Optional[str] = None, adb_port: Optional[int] = None) -> 'Popen':
    command = get_adb_command(device, command, adb_server, adb_port)
    logger.debug(command)
    popen = get_subprocess(command, shell=True)
    return popen


def grant_app_permissions(target: 'AndroidTarget', package: str) -> None:
    """
    Grant all requested permissions to an installed app package by parsing the
    'dumpsys package' output.

    :param target: The Android target on which the package is installed.
    :param package: The package name (e.g., "com.example.app").
    :raises TargetStableError: If permission granting fails or the package is invalid.
    """
    dumpsys: str = target.execute('dumpsys package {}'.format(package))

    permissions = re.search(
        r'requested permissions:\s*(?P<permissions>(android.permission.+\s*)+)', dumpsys
    )
    if permissions is None:
        return
    permissions_list: List[str] = permissions.group('permissions').replace(" ", "").splitlines()

    for permission in permissions_list:
        try:
            target.execute('pm grant {} {}'.format(package, permission))
        except TargetStableError:
            logger.debug('Cannot grant {}'.format(permission))


# Messy environment initialisation stuff...
class _AndroidEnvironment:
    # Make the initialization lazy so that we don't trigger an exception if the
    # user imports the module (directly or indirectly) without actually using
    # anything from it
    """
    Lazy-initialized environment data for Android tools (adb, aapt, etc.),
    constructed from ANDROID_HOME or by scanning the system PATH.
    """
    @property
    @functools.lru_cache(maxsize=None)
    def env(self) -> Android_Env_Type:
        """
        :returns: The discovered Android environment mapping with keys like 'adb', 'aapt', etc.
        :raises HostError: If we cannot find a suitable ANDROID_HOME or 'adb' in PATH.
        """
        android_home: Optional[str] = os.getenv('ANDROID_HOME')
        if android_home:
            env = self._from_android_home(android_home)
        else:
            env = self._from_adb()

        return env

    def get_env(self, name: Android_Env_TypeKeys) -> Optional[Union[str, int]]:
        """
        Retrieve a specific environment field, such as 'adb', 'aapt', or 'build_tools'.

        :param name: Name of the environment key.
        :returns: The value if found, else None.
        """
        return self.env[name]

    @classmethod
    def _from_android_home(cls, android_home: str) -> Android_Env_Type:
        """
        Build environment info from ANDROID_HOME.

        :param android_home: Path to Android SDK root.
        :returns: Dictionary of environment settings.
        """
        logger.debug('Using ANDROID_HOME from the environment.')
        platform_tools = os.path.join(android_home, 'platform-tools')

        return cast(Android_Env_Type, {
            'android_home': android_home,
            'platform_tools': platform_tools,
            'adb': os.path.join(platform_tools, 'adb'),
            'fastboot': os.path.join(platform_tools, 'fastboot'),
            **cls._init_common(android_home)
        })

    @classmethod
    def _from_adb(cls) -> Android_Env_Type:
        """
        Attempt to derive environment info by locating 'adb' on the system PATH.

        :returns: A dictionary of environment settings.
        :raises HostError: If 'adb' is not found in PATH.
        """
        adb_path = which('adb')
        if adb_path:
            logger.debug('Discovering ANDROID_HOME from adb path.')
            platform_tools = os.path.dirname(adb_path)
            android_home = os.path.dirname(platform_tools)

            return cast(Android_Env_Type, {
                'android_home': android_home,
                'platform_tools': platform_tools,
                'adb': adb_path,
                'fastboot': which('fastboot'),
                **cls._init_common(android_home)
            })
        else:
            raise HostError('ANDROID_HOME is not set and adb is not in PATH. '
                            'Have you installed Android SDK?')

    @classmethod
    def _init_common(cls, android_home: str) -> BuildToolsInfo:
        """
        Discover build tools, aapt, etc., from an Android SDK layout.

        :param android_home: Android SDK root path.
        :returns: Partial dictionary with keys like 'build_tools', 'aapt', 'aapt_version'.
        """
        logger.debug(f'ANDROID_HOME: {android_home}')
        build_tools = cls._discover_build_tools(android_home)
        return cast(BuildToolsInfo, {
            'build_tools': build_tools,
            **cls._discover_aapt(build_tools)
        })

    @staticmethod
    def _discover_build_tools(android_home: str) -> Optional[str]:
        """
        Attempt to locate the build-tools directory under android_home.

        :param android_home: Path to the SDK.
        :returns: Path to build-tools if found, else None.
        """
        build_tools = os.path.join(android_home, 'build-tools')
        if os.path.isdir(build_tools):
            return build_tools
        else:
            return None

    @staticmethod
    def _check_supported_aapt2(binary: str) -> bool:
        """
        Check if a given 'aapt2' binary supports 'dump badging'.

        :param binary: Path to the aapt2 binary.
        :returns: True if the binary appears to support the 'badging' command, else False.
        """
        # At time of writing the version argument of aapt2 is not helpful as
        # the output is only a placeholder that does not distinguish between versions
        # with and without support for badging. Unfortunately aapt has been
        # deprecated and fails to parse some valid apks so we will try to favour
        # aapt2 if possible else will fall back to aapt.
        # Try to execute the badging command and check if we get an expected error
        # message as opposed to an unknown command error to determine if we have a
        # suitable version.
        """
        check if aapt2 is supported
        """
        result: 'CompletedProcess' = subprocess.run([str(binary), 'dump', 'badging'],
                                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                                    universal_newlines=True)
        supported = bool(AAPT_BADGING_OUTPUT.search(result.stderr))
        msg: str = 'Found a {} aapt2 binary at: {}'
        logger.debug(msg.format('supported' if supported else 'unsupported', binary))
        return supported

    @classmethod
    def _discover_aapt(cls, build_tools: Optional[str]) -> Dict[str, Optional[Union[str, int]]]:
        """
        Attempt to find 'aapt2' or 'aapt' in build-tools (or PATH fallback).
        Prefers aapt2 if available.

        :param build_tools: Path to the build-tools directory or None if unknown.
        :returns: A dictionary with 'aapt' and 'aapt_version' keys.
        :raises HostError: If neither aapt nor aapt2 is found.
        """
        if build_tools:

            def find_aapt2(version: str) -> Tuple[Optional[int], Optional[str]]:
                path = os.path.join(build_tools, version, 'aapt2')
                if os.path.isfile(path) and cls._check_supported_aapt2(path):
                    return (2, path)
                else:
                    return (None, None)

            def find_aapt(version: str) -> Tuple[Optional[int], Optional[str]]:
                path: str = os.path.join(build_tools, version, 'aapt')
                if os.path.isfile(path):
                    return (1, path)
                else:
                    return (None, None)

            versions: List[str] = os.listdir(build_tools)
            found: Generator[Tuple[str, Tuple[Optional[int], Optional[str]]]] = (
                (version, finder(version))
                for version in reversed(sorted(versions))
                for finder in (find_aapt2, find_aapt)
            )

            for version, (aapt_version, aapt_path) in found:
                if aapt_path:
                    logger.debug(f'Using {aapt_path} for version {version}')
                    return dict(
                        aapt=aapt_path,
                        aapt_version=aapt_version,
                    )

        # Try detecting aapt2 and aapt from PATH
        aapt2_path: Optional[str] = which('aapt2')
        aapt_path = which('aapt')
        if aapt2_path and cls._check_supported_aapt2(aapt2_path):
            return dict(
                aapt=aapt2_path,
                aapt_version=2,
            )
        elif aapt_path:
            return dict(
                aapt=aapt_path,
                aapt_version=1,
            )
        else:
            raise HostError('aapt/aapt2 not found. Please make sure it is avaliable in PATH or at least one Android platform is installed')


class LogcatMonitor(object):
    """
    Helper class for monitoring Anroid's logcat

    :param target: Android target to monitor

    :param regexps: List of uncompiled regular expressions to filter on the
                    device. Logcat entries that don't match any will not be
                    seen. If omitted, all entries will be sent to host.
    """

    @property
    def logfile(self) -> Optional[Union['TextIOWrapper', '_TemporaryFileWrapper[str]']]:
        """
        Return the file-like object that logcat is writing to, if any.

        :returns: The log file or None.
        """
        return self._logfile

    def __init__(self, target: 'AndroidTarget', regexps: Optional[List[str]] = None,
                 logcat_format: Optional[str] = None):
        super(LogcatMonitor, self).__init__()

        self.target = target
        self._regexps = regexps
        self._logcat_format = logcat_format
        self._logcat: Optional[spawn] = None
        self._logfile: Optional[Union['TextIOWrapper', '_TemporaryFileWrapper[str]']] = None

    def start(self, outfile: Optional[str] = None) -> None:
        """
        Begin capturing logcat output. If outfile is given, logcat lines are
        appended there; otherwise, a temporary file is used.

        :param outfile: A path to a file on the host, or None for a temporary file.
        """
        if outfile:
            self._logfile = open(outfile, 'w')
        else:
            self._logfile = tempfile.NamedTemporaryFile(mode='w')

        self.target.clear_logcat()

        logcat_cmd: str = 'logcat'

        # Join all requested regexps with an 'or'
        if self._regexps:
            regexp: str = '{}'.format('|'.join(self._regexps))
            if len(self._regexps) > 1:
                regexp = '({})'.format(regexp)
            # Logcat on older version of android do not support the -e argument
            # so fall back to using grep.
            if self.target.get_sdk_version() > 23:
                logcat_cmd = '{} -e {}'.format(logcat_cmd, quote(regexp))
            else:
                logcat_cmd = '{} | grep {}'.format(logcat_cmd, quote(regexp))

        if self._logcat_format:
            logcat_cmd = "{} -v {}".format(logcat_cmd, quote(self._logcat_format))

        logcat_cmd = get_adb_command(self.target.conn.device,
                                     logcat_cmd, self.target.adb_server,
                                     self.target.adb_port) if isinstance(self.target.conn, AdbConnection) else ''
        logcat_cmd = f"/bin/bash -c '{logcat_cmd}'"
        logger.debug('logcat command ="{}"'.format(logcat_cmd))
        self._logcat = pexpect.spawn(logcat_cmd, logfile=self._logfile, encoding='utf-8')

    def stop(self) -> None:
        """
        Stop capturing logcat and close the log file if applicable.
        """
        self.flush_log()
        if self._logcat:
            self._logcat.terminate()
        if self._logfile:
            self._logfile.close()

    def get_log(self) -> List[str]:
        """
        Retrieve all captured lines from the log so far.

        :returns: A list of log lines from the log file.
        """
        self.flush_log()
        if self._logfile:
            with open(self._logfile.name) as fh:
                return [line for line in fh]
        else:
            return []

    def flush_log(self) -> None:
        """
        Force-read all pending data from the logcat pexpect spawn to ensure it's
        written to the logfile. Prevents missed lines if pexpect hasn't pulled them yet.
        """
        # Unless we tell pexect to 'expect' something, it won't read from
        # logcat's buffer or write into our logfile. We'll need to force it to
        # read any pending logcat output.
        while True:
            try:
                read_size = 1024 * 8
                # This will read up to read_size bytes, but only those that are
                # already ready (i.e. it won't block). If there aren't any bytes
                # already available it raises pexpect.TIMEOUT.
                buf: str = ''
                if self._logcat:
                    buf = self._logcat.read_nonblocking(read_size, timeout=0)

                # We can't just keep calling read_nonblocking until we get a
                # pexpect.TIMEOUT (i.e. until we don't find any available
                # bytes), because logcat might be writing bytes the whole time -
                # in that case we might never return from this function. In
                # fact, we only care about bytes that were written before we
                # entered this function. So, if we read read_size bytes (as many
                # as we were allowed to), then we'll assume there are more bytes
                # that have already been sitting in the output buffer of the
                # logcat command. If not, we'll assume we read everything that
                # had already been written.
                if len(buf) == read_size:
                    continue
                else:
                    break
            except pexpect.TIMEOUT:
                # No available bytes to read. No prob, logcat just hasn't
                # printed anything since pexpect last read from its buffer.
                break

    def clear_log(self) -> None:
        """
        Erase current content of the log file so subsequent calls to get_log()
        won't return older lines.
        """
        if self._logfile:
            with open(self._logfile.name, 'w') as _:
                pass

    def search(self, regexp: str) -> List[str]:
        """
        Search the captured lines for matches of the given regexp.

        :param regexp: A regular expression pattern.
        :returns: All matching lines found so far.
        """
        return [line for line in self.get_log() if re.match(regexp, line)]

    def wait_for(self, regexp: str, timeout: Optional[int] = 30) -> List[str]:
        """
        Search a line that matches a regexp in the logcat log
        Wait for it to appear if it's not found

        :param regexp: regexp to search

        :param timeout: Timeout in seconds, before rasing RuntimeError.
                        ``None`` means wait indefinitely

        :returns: List of matched strings
        :raises RuntimeError: If the regex is not found within ``timeout`` seconds.
        """
        log: List[str] = self.get_log()
        res: List[str] = [line for line in log if re.match(regexp, line)]

        # Found some matches, return them
        if res:
            return res

        # Store the number of lines we've searched already, so we don't have to
        # re-grep them after 'expect' returns
        next_line_num: int = len(log)

        try:
            if self._logcat:
                self._logcat.expect(regexp, timeout=timeout)
        except pexpect.TIMEOUT:
            raise RuntimeError('Logcat monitor timeout ({}s)'.format(timeout))

        return [line for line in self.get_log()[next_line_num:]
                if re.match(regexp, line)]


_ANDROID_ENV = _AndroidEnvironment()
