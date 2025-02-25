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
import os
import signal
import shutil
import subprocess
import sys
from getpass import getpass
from shlex import quote

from devlib.exception import (
    TargetStableError, TargetTransientCalledProcessError, TargetStableCalledProcessError
)
from devlib.utils.misc import check_output, get_logger
from devlib.connection import ConnectionBase, PopenBackgroundCommand
from typing import Optional, TYPE_CHECKING, cast, Union, List
from typing_extensions import Literal
if TYPE_CHECKING:
    from devlib.platform import Platform
    from devlib.utils.annotation_helpers import SubprocessCommand
    from signal import Signals
    from logging import Logger

if sys.version_info >= (3, 8):
    def copy_tree(src: str, dst: str) -> None:
        """
        Recursively copy an entire directory tree from ``src`` to ``dst``,
        preserving the directory structure but **not** file metadata
        (modification times, modes, etc.). If ``dst`` already exists, this
        overwrites matching files.

        :param src: The source directory path.
        :param dst: The destination directory path.
        :raises OSError: If any file or directory within ``src`` cannot be copied.
        """
        from shutil import copy, copytree
        copytree(
            src,
            dst,
            # dirs_exist_ok=True only exists in Python >= 3.8
            dirs_exist_ok=True,
            # Do not copy creation and modification time to behave like other
            # targets.
            copy_function=copy
        )
else:
    def copy_tree(src, dst):
        """
        Recursively copy an entire directory tree from ``src`` to ``dst``,
        preserving the directory structure but **not** file metadata
        (modification times, modes, etc.). If ``dst`` already exists, this
        overwrites matching files.

        :param src: The source directory path.
        :param dst: The destination directory path.
        :raises OSError: If any file or directory within ``src`` cannot be copied.

        .. note::
            This uses :func:`distutils.dir_util.copy_tree` under Python < 3.8, which
            does not support ``dirs_exist_ok=True``. The behavior is effectively the same
            for overwriting existing paths.
        """
        from distutils.dir_util import copy_tree
        # Mirror the behavior of all other targets which only copy the
        # content without metadata
        copy_tree(src, dst, preserve_mode=False, preserve_times=False)


PACKAGE_BIN_DIRECTORY: str = os.path.join(os.path.dirname(__file__), 'bin')


# pylint: disable=redefined-outer-name
def kill_children(pid: int, signal: 'Signals' = signal.SIGKILL) -> None:
    """
    Recursively kill all child processes of the specified process ID, then kill
    the process itself with the given signal.

    :param pid: The process ID whose children (and itself) will be killed.
    :param signal_: The signal to send (defaults to SIGKILL).
    :raises ProcessLookupError: If any child process does not exist (e.g., race conditions).
    """
    with open('/proc/{0}/task/{0}/children'.format(pid), 'r') as fd:
        for cpid in map(int, fd.read().strip().split()):
            kill_children(cpid, signal)
            os.kill(cpid, signal)


class LocalConnection(ConnectionBase):
    """
    A connection to the local host, allowing the local system to be treated as a
    devlib Target. Commands are run directly via :mod:`subprocess`, rather than
    an SSH or ADB connection.

    :param platform: A devlib Platform object for describing this local system
        (e.g., CPU topology). If None, defaults may be used.
    :param keep_password: If ``True``, cache the user’s sudo password in memory
        after prompting. Defaults to True.
    :param unrooted: If ``True``, assume the local system is non-root and do not
        attempt root commands. This avoids prompting for a password.
    :param password: Password for sudo. If provided, will not prompt the user.
    :param timeout: A default timeout (in seconds) for connection-based operations.
    """
    name = 'local'
    host = 'localhost'

    # pylint: disable=unused-argument
    def __init__(self, platform: Optional['Platform'] = None,
                 keep_password: bool = True, unrooted: bool = False,
                 password: Optional[str] = None, timeout: Optional[int] = None):
        """
        Initialize the LocalConnection instance.
        """
        super().__init__()
        self._connected_as_root: Optional[bool] = None
        self.logger: Logger = get_logger('local_connection')
        self.keep_password: bool = keep_password
        self.unrooted: bool = unrooted
        self.password: Optional[str] = password

    @property
    def connected_as_root(self) -> Optional[bool]:
        """
        Indicate whether the current user context is effectively 'root' (uid=0).

        :return:
            - True if root
            - False if not root
            - None if undetermined
        """
        if self._connected_as_root is None:
            result: str = self.execute('id', as_root=False)
            self._connected_as_root = 'uid=0(' in result
        return self._connected_as_root

    @connected_as_root.setter
    def connected_as_root(self, state: Optional[bool]) -> None:
        """
        Override the known 'connected_as_root' state, if needed.

        :param state: True if effectively root, False if not, or None if unknown.
        """
        self._connected_as_root = state

    def _copy_path(self, source: str, dest: str) -> None:
        """
        Copy a single file or directory from ``source`` to ``dest``. If ``source``
        is a directory, it is copied recursively.

        :param source: The path to the file or directory on the local system.
        :param dest: Destination path.
        :raises OSError: If any part of the copy operation fails.
        """
        self.logger.debug('copying {} to {}'.format(source, dest))
        if os.path.isdir(source):
            copy_tree(source, dest)
        else:
            shutil.copy(source, dest)

    def _copy_paths(self, sources: List[str], dest: str) -> None:
        """
        Copy multiple paths (files or directories) to the same destination.

        :param sources: A tuple of file or directory paths to copy.
        :param dest: The destination path, which may be a directory.
        :raises OSError: If any part of a copy operation fails.
        """
        for source in sources:
            self._copy_path(source, dest)

    def push(self, sources: List[str], dest: str, timeout: Optional[int] = None,
             as_root: bool = False) -> None:  # pylint: disable=unused-argument
        """
        Transfer a list of files **from the local system** to itself (no-op in some contexts).
        In practice, this copies each file in ``sources`` to ``dest``.

        :param sources: List of file or directory paths on the local system.
        :param dest: Destination path on the local system.
        :param timeout: Timeout in seconds for each file copy; unused here (local copy).
        :param as_root: If True, tries to escalate with sudo. Typically a no-op locally.
        :raises TargetStableError: If the system is set to unrooted but as_root=True is used.
        :raises OSError: If copying fails at any point.
        """
        self._copy_paths(sources, dest)

    def pull(self, sources: List[str], dest: str, timeout: Optional[int] = None,
             as_root: bool = False) -> None:  # pylint: disable=unused-argument
        """
        Transfer a list of files **from the local system** to the local system (similar to :meth:`push`).

        :param sources: list of paths on the local system.
        :param dest: Destination directory or file path on local system.
        :param timeout: Timeout in seconds; typically unused.
        :param as_root: If True, attempts to use sudo for the copy, if not already root.
        :raises TargetStableError: If the system is set to unrooted but as_root=True is used.
        :raises OSError: If copying fails.
        """
        self._copy_paths(sources, dest)

    # pylint: disable=unused-argument
    def execute(self, command: 'SubprocessCommand', timeout: Optional[int] = None,
                check_exit_code: bool = True, as_root: Optional[bool] = False,
                strip_colors: bool = True, will_succeed: bool = False) -> str:
        """
        Execute a command locally (via :func:`subprocess.check_output`), returning
        combined stdout+stderr output. Optionally escalates privileges with sudo.

        :param command: The command to execute (string or SubprocessCommand).
        :param timeout: Time in seconds after which the command is forcibly terminated.
        :param check_exit_code: If True, raise an error on nonzero exit codes.
        :param as_root: If True, attempt sudo unless already root. Fails if ``unrooted=True``.
        :param strip_colors: If True, attempt to remove ANSI color codes from output.
                             (Not used in this local example.)
        :param will_succeed: If True, treat a failing command as a transient error
                             rather than stable.
        :return: The combined stdout+stderr of the command.
        :raises TargetTransientCalledProcessError: If the command fails but is considered transient.
        :raises TargetStableCalledProcessError: If the command fails and is considered stable.
        :raises TargetStableError: If run as root is requested but unrooted is True.
        """
        self.logger.debug(command)
        use_sudo: Optional[bool] = as_root and not self.connected_as_root
        if use_sudo:
            if self.unrooted:
                raise TargetStableError('unrooted')
            password: str = self._get_password()
            # Empty prompt with -p '' to avoid adding a leading space to the
            # output.
            command = "echo {} | sudo -k -p '' -S -- sh -c {}".format(quote(password), quote(cast(str, command)))
        ignore: Optional[Union[int, List[int], Literal['all']]] = None if check_exit_code else 'all'
        try:
            stdout, stderr = check_output(command, shell=True, timeout=timeout, ignore=ignore)
        except subprocess.CalledProcessError as e:
            cls = TargetTransientCalledProcessError if will_succeed else TargetStableCalledProcessError
            raise cls(
                e.returncode,
                command,
                e.output,
                e.stderr,
            )

        # Remove the one-character prompt of sudo -S -p
        if use_sudo and stderr:
            stderr = stderr[1:]

        return stdout + stderr

    def background(self, command: 'SubprocessCommand', stdout: int = subprocess.PIPE,
                   stderr: int = subprocess.PIPE, as_root: Optional[bool] = False) -> PopenBackgroundCommand:
        """
        Launch a command on the local system in the background, returning
        a handle to manage its execution via :class:`PopenBackgroundCommand`.

        :param command: The command or SubprocessCommand to run.
        :param stdout: File handle or constant (e.g. subprocess.PIPE) for capturing stdout.
        :param stderr: File handle or constant for capturing stderr.
        :param as_root: If True, attempt to run with sudo unless already root.
        :return: A background command object that can be polled, waited on, or killed.
        :raises TargetStableError: If unrooted is True but as_root is requested.

        .. note:: This **will block the connection** until the command completes.
        """
        if as_root and not self.connected_as_root:
            if self.unrooted:
                raise TargetStableError('unrooted')
            password: str = self._get_password()
            # Empty prompt with -p '' to avoid adding a leading space to the
            # output.
            command = "echo {} | sudo -k -p '' -S -- sh -c {}".format(quote(password), quote(cast(str, command)))

        # Make sure to get a new PGID so PopenBackgroundCommand() can kill
        # all sub processes that could be started without troubles.
        def preexec_fn():
            os.setpgrp()

        def make_init_kwargs(command):
            popen = subprocess.Popen(
                command,
                stdout=stdout,
                stderr=stderr,
                stdin=subprocess.PIPE,
                shell=True,
                preexec_fn=preexec_fn,
            )
            return dict(
                popen=popen,
            )

        return PopenBackgroundCommand.from_factory(
            conn=self,
            cmd=command,
            as_root=as_root,
            make_init_kwargs=make_init_kwargs,
        )

    def _close(self) -> None:
        """
        Close the connection to the device. The :class:`Connection` object should not
        be used after this method is called. There is no way to reopen a previously
        closed connection, a new connection object should be created instead.
        """
        pass

    def cancel_running_command(self) -> None:
        """
        Cancel a running command (previously started with :func:`background`) and free up the connection.
        It is valid to call this if the command has already terminated (or if no
        command was issued), in which case this is a no-op.
        """
        pass

    def wait_for_device(self, timeout: int = 30) -> None:
        """
        Wait for the local system to be 'available'. In practice, this is always a no-op
        since we are already local.
        :param timeout: Ignored.
        """
        return

    def reboot_bootloader(self, timeout: int = 30) -> None:
        """
        Attempt to reboot into a bootloader mode. Not implemented for local usage.

        :param timeout: Time in seconds to wait for the operation to complete.
        :raises NotImplementedError: Always, as local usage does not support bootloader reboots.
        """
        raise NotImplementedError()

    def _get_password(self) -> str:
        """
        Prompt for the user's sudo password if not already cached.

        :return: The password string, either from cache or via user input.
        """
        if self.password:
            return self.password
        password: str = getpass('sudo password:')
        if self.keep_password:
            self.password = password
        return password
