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

from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from shlex import quote
import os
import signal
import subprocess
import threading
import time
import logging
import select
import fcntl

from devlib.utils.misc import InitCheckpoint
from devlib.utils.annotation_helpers import SubprocessCommand
from typing import (Optional, TYPE_CHECKING, Set,
                    Tuple, IO, Dict, List, Union,
                    Generator, Callable)
from typing_extensions import Protocol, Literal

if TYPE_CHECKING:
    from signal import Signals
    from subprocess import Popen
    from threading import Lock, Thread, Event
    from logging import Logger
    from paramiko.channel import Channel
    from paramiko.sftp_client import SFTPClient
    from scp import SCPClient


class HasInitialized(Protocol):
    """
    Protocol indicating that the object includes an ``initialized`` property
    and a ``close()`` method. Used to ensure safe clean-up in destructors.

    :ivar initialized: ``True`` if the object finished initializing successfully,
        otherwise ``False`` if initialization failed or is incomplete.
    :vartype initialized: bool
    """
    initialized: bool

    # other functions referred by the object with the initialized property
    def close(self) -> None:
        """
        Close method expected on objects that provide ``initialized``.
        """
        ...


_KILL_TIMEOUT: int = 3
"""
int: The default time (in seconds) to wait between sending SIGTERM and SIGKILL
during process cancellation (see :meth:`BackgroundCommand.cancel`).
"""


def _kill_pgid_cmd(pgid: int, sig: 'Signals', busybox: Optional[str]) -> str:
    """
    Construct a shell command string that sends a specified signal to a given
    process group.

    :param pgid: The process group ID (PGID) to signal.
    :type pgid: int
    :param sig: The signal to send (e.g., SIGTERM, SIGKILL).
    :type sig: signal.Signals
    :param busybox: Path to a busybox binary on the target, if any. If None,
        the command may assume `kill` is already in PATH.
    :type busybox: str or None
    :return: A complete shell command that, when run, kills the PGID with the given signal.
    :rtype: str
    """
    return '{} kill -{} -{}'.format(busybox, sig.value, pgid)


def _popen_communicate(bg: 'BackgroundCommand', popen: 'Popen', input: bytes,
                       timeout: Optional[int]) -> Tuple[Optional[bytes], Optional[bytes]]:
    """
    Wrapper around ``popen.communicate(...)`` to handle timeouts and
    cancellation of a background command.

    :param bg: The associated :class:`BackgroundCommand` object that may be canceled.
    :type bg: BackgroundCommand
    :param popen: The :class:`subprocess.Popen` instance to communicate with.
    :type popen: subprocess.Popen
    :param input: Bytes to send to stdin.
    :type input: bytes
    :param timeout: The timeout in seconds or None for no timeout.
    :type timeout: int or None
    :return: A tuple (stdout, stderr) if the command completes successfully.
    :rtype: (bytes or None, bytes or None)
    :raises subprocess.TimeoutExpired: If the command doesn't complete in time.
    :raises subprocess.CalledProcessError: If the command exits with a non-zero return code.
    """
    try:
        stdout: Optional[bytes]
        stderr: Optional[bytes]
        stdout, stderr = popen.communicate(input=input, timeout=timeout)
    except subprocess.TimeoutExpired:
        bg.cancel()
        raise

    ret = popen.returncode
    if ret:
        raise subprocess.CalledProcessError(
            ret,
            popen.args,
            stdout,
            stderr,
        )
    else:
        return (stdout, stderr)


class ConnectionBase(InitCheckpoint):
    """
    Base class for all connections.
    A :class:`Connection` abstracts an actual physical connection to a device. The
    first connection is created when :func:`Target.connect` method is called. If a
    :class:`~devlib.target.Target` is used in a multi-threaded environment, it will
    maintain a connection for each thread in which it is invoked. This allows
    the same target object to be used in parallel in multiple threads.

    :class:`Connection` s will be automatically created and managed by
    :class:`~devlib.target.Target` s, so there is usually no reason to create one
    manually. Instead, configuration for a :class:`Connection` is passed as
    `connection_settings` parameter when creating a
    :class:`~devlib.target.Target`. The connection to be used target is also
    specified on instantiation by `conn_cls` parameter, though all concrete
    :class:`~devlib.target.Target` implementations will set an appropriate
    default, so there is typically no need to specify this explicitly.

    :param poll_transfers: If True, manage file transfers by polling for progress.
    :type poll_transfers: bool
    :param start_transfer_poll_delay: Delay in seconds before first checking a
        file transfer's progress.
    :type start_transfer_poll_delay: int
    :param total_transfer_timeout: Cancel transfers if they exceed this many seconds.
    :type total_transfer_timeout: int
    :param transfer_poll_period: Interval (seconds) between transfer progress checks.
    :type transfer_poll_period: int
    """
    def __init__(
        self,
        poll_transfers: bool = False,
        start_transfer_poll_delay: int = 30,
        total_transfer_timeout: int = 3600,
        transfer_poll_period: int = 30,
    ):
        self._current_bg_cmds: Set['BackgroundCommand'] = set()
        self._closed: bool = False
        self._close_lock: Lock = threading.Lock()
        self.busybox: Optional[str] = None
        self.logger: Logger = logging.getLogger('Connection')

        self.transfer_manager = TransferManager(
            self,
            start_transfer_poll_delay=start_transfer_poll_delay,
            total_transfer_timeout=total_transfer_timeout,
            transfer_poll_period=transfer_poll_period,
        ) if poll_transfers else NoopTransferManager()

    def cancel_running_command(self) -> Optional[bool]:
        """
        Cancel all active background commands tracked by this connection.
        """
        bg_cmds: Set['BackgroundCommand'] = set(self._current_bg_cmds)
        for bg_cmd in bg_cmds:
            bg_cmd.cancel()
        return None

    @abstractmethod
    def _close(self) -> None:
        """
        Close the connection.

        The public :meth:`close` method makes sure that :meth:`_close` will
        only be called once, and will serialize accesses to it if it happens to
        be called from multiple threads at once.
        """

    def close(self) -> None:
        """
        Cancel any ongoing commands and finalize the connection. Safe to call multiple times,
        does nothing after the first invocation.
        """
        def finish_bg() -> None:
            bg_cmds: Set['BackgroundCommand'] = set(self._current_bg_cmds)
            n: int = len(bg_cmds)
            if n:
                self.logger.debug(f'Canceling {n} background commands before closing connection')
            for bg_cmd in bg_cmds:
                bg_cmd.cancel()

        # Locking the closing allows any thread to safely call close() as long
        # as the connection can be closed from a thread that is not the one it
        # started its life in.
        with self._close_lock:
            if not self._closed:
                finish_bg()
                self._close()
                self._closed = True

    # Ideally, that should not be relied upon but that will improve the chances
    # of the connection being properly cleaned up when it's not in use anymore.
    def __del__(self: HasInitialized):
        """
        Destructor ensuring the connection is closed if not already. Only runs
        if object initialization succeeded (initialized=True).
        """
        # Since __del__ will be called if an exception is raised in __init__
        # (e.g. we cannot connect), we only run close() when we are sure
        # __init__ has completed successfully.
        if self.initialized:
            self.close()

    @abstractmethod
    def execute(self, command: 'SubprocessCommand', timeout: Optional[int] = None,
                check_exit_code: bool = True, as_root: Optional[bool] = False,
                strip_colors: bool = True, will_succeed: bool = False) -> str:
        """
        Execute a shell command and return the combined stdout/stderr.

        :param command: Command string or SubprocessCommand detailing the command to run.
        :type command: SubprocessCommand
        :param timeout: Timeout in seconds (None for no limit).
        :type timeout: int or None
        :param check_exit_code: If True, raise an error if exit code is non-zero.
        :type check_exit_code: bool
        :param as_root: If True, attempt to run with elevated privileges.
        :type as_root: bool or None
        :param strip_colors: Remove ANSI color codes from output if True.
        :type strip_colors: bool
        :param will_succeed: If True, interpret a failing command as a transient environment error.
        :type will_succeed: bool
        :returns: The command's combined stdout and stderr.
        :rtype: str
        :raises DevlibTransientError: If the command fails and is considered transient (will_succeed=True).
        :raises DevlibStableError: If the command fails in a stable way (exit code != 0, or other error).
        """


class BackgroundCommand(ABC):
    """
    Allows managing a running background command using a subset of the
    :class:`subprocess.Popen` API.

    Instances of this class can be used as context managers, with the same
    semantic as :class:`subprocess.Popen`.

    :param conn: The connection that owns this background command.
    :type conn: ConnectionBase
    """

    def __init__(self, conn: 'ConnectionBase'):
        self.conn = conn

        # Poll currently opened background commands on that connection to make
        # them deregister themselves if they are completed. This avoids
        # accumulating terminated commands and therefore leaking associated
        # resources if the user is not careful and does not use the context
        # manager API.
        for bg_cmd in set(conn._current_bg_cmds):
            try:
                bg_cmd.poll()
            # We don't want anything to fail here because of another command
            except Exception:
                pass

        conn._current_bg_cmds.add(self)

    def _deregister(self) -> None:
        """
        deregister the background command
        """
        try:
            self.conn._current_bg_cmds.remove(self)
        except KeyError:
            pass

    @abstractmethod
    def _send_signal(self, sig: 'Signals') -> None:
        """
        Subclass-specific implementation to send a signal (e.g., SIGTERM) to the process group.
        """
        pass

    def send_signal(self, sig: 'Signals') -> None:
        """
        Send a POSIX signal to the background command's process group ID
        (PGID).

        :param signal: Signal to send.
        :type signal: signal.Signals
        """
        try:
            return self._send_signal(sig)
        finally:
            # Deregister if the command has finished
            self.poll()

    def kill(self) -> None:
        """
        Send SIGKILL to the background command.
        """
        self.send_signal(signal.SIGKILL)

    def cancel(self, kill_timeout: int = _KILL_TIMEOUT) -> None:
        """
        Try to gracefully terminate the process by sending ``SIGTERM``, then
        waiting for ``kill_timeout`` to send ``SIGKILL``.

        :param kill_timeout: Seconds to wait between SIGTERM and SIGKILL.
        :type kill_timeout: int
        """
        try:
            if self.poll() is None:
                return self._cancel(kill_timeout=kill_timeout)
        finally:
            self._deregister()

    @abstractmethod
    def _cancel(self, kill_timeout: int) -> None:
        """
        Subclass-specific logic for :meth:`cancel`. Usually sends SIGTERM, waits,
        then sends SIGKILL if needed.
        """
        pass

    @abstractmethod
    def _wait(self) -> int:
        """
        Wait for the command to complete. Return its exit code.
        """
        pass

    def wait(self) -> int:
        """
        Block until the command completes, returning the exit code.

        :returns: The exit code of the command.
        :rtype: int
        """
        try:
            return self._wait()
        finally:
            self._deregister()

    def communicate(self, input: bytes = b'', timeout: Optional[int] = None) -> Tuple[Optional[bytes], Optional[bytes]]:
        """
        Write to stdin and read all data from stdout/stderr until the command exits.

        :param input: Bytes to send to stdin.
        :type input: bytes
        :param timeout: Max time to wait for the command to exit, or None if indefinite.
        :type timeout: int or None
        :returns: A tuple of (stdout, stderr) if the command exits cleanly.
        :rtype: Tuple[Optional[bytes], Optional[bytes]]
        :raises subprocess.TimeoutExpired: If the process runs past the timeout.
        :raises subprocess.CalledProcessError: If the process exits with a non-zero code.
        """
        try:
            return self._communicate(input=input, timeout=timeout)
        finally:
            self.close()

    @abstractmethod
    def _communicate(self, input: bytes, timeout: Optional[int]) -> Tuple[Optional[bytes], Optional[bytes]]:
        """
        Method to override in subclasses to implement :meth:`communicate`.
        """
        pass

    @abstractmethod
    def _poll(self) -> Optional[int]:
        """
        Method to override in subclasses to implement :meth:`poll`.
        """
        pass

    def poll(self) -> Optional[int]:
        """
        Return the exit code if the command has finished, otherwise None.
        Deregisters if the command is done.

        :returns: Exit code or None if ongoing.
        :rtype: int or None
        """
        retcode = self._poll()
        if retcode is not None:
            self._deregister()
        return retcode

    @property
    @abstractmethod
    def stdin(self) -> Optional[IO]:
        """
        A file-like object representing this command's standard input. May be None if unsupported.
        """

    @property
    @abstractmethod
    def stdout(self) -> Optional[IO]:
        """
        A file-like object representing this command's standard output. May be None.
        """

    @property
    @abstractmethod
    def stderr(self) -> Optional[IO]:
        """
        A file-like object representing this command's standard error. May be None.
        """

    @property
    @abstractmethod
    def pid(self) -> int:
        """
        Process Group ID (PGID) of the background command.

        Since the command is usually wrapped in shell processes for IO
        redirections, sudo etc, the PID cannot be assumed to be the actual PID
        of the command passed by the user. It's is guaranteed to be a PGID
        instead, which means signals sent to it as such will target all
        subprocesses involved in executing that command.
        """

    @abstractmethod
    def _close(self) -> int:
        """
        Subclass hook for final cleanup: close streams, wait for exit, return exit code.
        """
        pass

    def close(self) -> int:
        """
        Close any open streams and finalize the command. Return exit code.

        :returns: The command's final exit code.
        :rtype: int

        .. note:: If the command is writing to its stdout/stderr, it might be
            blocked on that and die when the streams are closed.
        """
        try:
            return self._close()
        finally:
            self._deregister()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()


class PopenBackgroundCommand(BackgroundCommand):
    """
    Runs a command via ``subprocess.Popen`` in the background. Signals are sent
    to the process group. Streams are accessible via ``stdin``, ``stdout``, and ``stderr``.

    :param conn: The parent connection.
    :type conn: ConnectionBase
    :param popen: The Popen object controlling the shell command.
    :type popen: Popen
    """

    def __init__(self, conn: 'ConnectionBase', popen: 'Popen'):
        super().__init__(conn=conn)
        self.popen = popen

    def _send_signal(self, sig: 'Signals') -> None:
        """
        Send a signal to the process group
        """
        return os.killpg(self.popen.pid, sig)

    @property
    def stdin(self) -> Optional[IO]:
        return self.popen.stdin

    @property
    def stdout(self) -> Optional[IO]:
        return self.popen.stdout

    @property
    def stderr(self) -> Optional[IO]:
        return self.popen.stderr

    @property
    def pid(self) -> int:
        return self.popen.pid

    def _wait(self) -> int:
        return self.popen.wait()

    def _communicate(self, input: bytes, timeout: Optional[int]) -> Tuple[Optional[bytes], Optional[bytes]]:
        return _popen_communicate(self, self.popen, input, timeout)

    def _poll(self) -> Optional[int]:
        return self.popen.poll()

    def _cancel(self, kill_timeout: int) -> None:
        popen = self.popen
        os.killpg(os.getpgid(popen.pid), signal.SIGTERM)
        try:
            popen.wait(timeout=kill_timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(popen.pid), signal.SIGKILL)

    def _close(self) -> int:
        self.popen.__exit__(None, None, None)
        return self.popen.returncode

    def __enter__(self):
        super().__enter__()
        self.popen.__enter__()
        return self


class ParamikoBackgroundCommand(BackgroundCommand):
    """
    Background command using a Paramiko :class:`Channel` for remote SSH-based execution.
    Handles signals by running kill commands on the remote, using the PGID.

    :param conn: The SSH-based connection.
    :type conn: ConnectionBase
    :param chan: The Paramiko channel running the remote command.
    :type chan: Channel
    :param pid: Remote process group ID for signaling.
    :type pid: int
    :param as_root: True if run with elevated privileges.
    :type as_root: bool or None
    :param cmd: The shell command executed (for reference).
    :type cmd: SubprocessCommand
    :param stdin: A file-like object to write into the remote stdin.
    :type stdin: IO
    :param stdout: A file-like object for reading from the remote stdout.
    :type stdout: IO
    :param stderr: A file-like object for reading from the remote stderr.
    :type stderr: IO
    :param redirect_thread: A thread that captures data from the channel and writes to
        stdout/stderr pipes.
    :type redirect_thread: Thread
    """
    def __init__(self, conn: 'ConnectionBase', chan: 'Channel', pid: int,
                 as_root: Optional[bool], cmd: 'SubprocessCommand', stdin: IO,
                 stdout: IO, stderr: IO, redirect_thread: 'Thread'):
        super().__init__(conn=conn)
        self.chan = chan
        self.as_root = as_root
        self._pid = pid
        self._stdin = stdin
        self._stdout = stdout
        self._stderr = stderr
        self.redirect_thread = redirect_thread
        self.cmd = cmd

    def _send_signal(self, sig: 'Signals') -> None:
        # If the command has already completed, we don't want to send a signal
        # to another process that might have gotten that PID in the meantime.
        if self.poll() is not None:
            return
        # Use -PGID to target a process group rather than just the process
        # itself
        cmd = _kill_pgid_cmd(self.pid, sig, self.conn.busybox)
        self.conn.execute(cmd, as_root=self.as_root)

    @property
    def pid(self) -> int:
        return self._pid

    def _wait(self) -> int:
        status = self.chan.recv_exit_status()
        # Ensure that the redirection thread is finished copying the content
        # from paramiko to the pipe.
        self.redirect_thread.join()
        return status

    def _communicate(self, input: bytes, timeout: Optional[int]) -> Tuple[Optional[bytes], Optional[bytes]]:
        """
        Implementation for reading from stdout/stderr, writing to stdin,
        handling timeouts, etc. Raise an error if non-zero exit or timeout.
        """
        stdout = self._stdout
        stderr = self._stderr
        chan = self.chan

        # For some reason, file descriptors in the read-list of select() can
        # still end up blocking in .read(), so make the non-blocking to avoid a
        # deadlock. Since _communicate() will consume all input and all output
        # until the command dies, we can do whatever we want with the pipe
        # without affecting external users.
        for s in (stdout, stderr):
            fcntl.fcntl(s.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)

        out: Dict[IO, List[bytes]] = {stdout: [], stderr: []}
        ret: Optional[int] = None
        can_send: bool = True

        select_timeout: int = 1
        if timeout is not None:
            select_timeout = min(select_timeout, 1)

        def create_out() -> Tuple[bytes, bytes]:
            return (
                b''.join(out[stdout]),
                b''.join(out[stderr])
            )

        start: float = time.monotonic()

        while ret is None:
            # Even if ret is not None anymore, we need to drain the streams
            ret = self.poll()

            if timeout is not None and ret is None and time.monotonic() - start >= timeout:
                self.cancel()
                _stdout, _stderr = create_out()
                raise subprocess.TimeoutExpired(self.cmd, timeout, _stdout, _stderr)

            can_send &= (not chan.closed) & bool(input)
            wlist: List[Channel] = [chan] if can_send else []

            if can_send and chan.send_ready():
                try:
                    n: int = chan.send(input)
                # stdin might have been closed already
                except OSError:
                    can_send = False
                    chan.shutdown_write()
                else:
                    input = input[n:]
                    if not input:
                        # Send EOF on stdin
                        chan.shutdown_write()
            rs: List[IO]
            ws: List[IO]
            rs, ws, _ = select.select(
                [x for x in (stdout, stderr) if not x.closed],
                wlist,
                [],
                select_timeout,
            )

            for r in rs:
                chunk: bytes = r.read()
                if chunk:
                    out[r].append(chunk)

        _stdout, _stderr = create_out()

        if ret:
            raise subprocess.CalledProcessError(
                ret,
                self.cmd,
                _stdout,
                _stderr,
            )
        else:
            return (_stdout, _stderr)

    def _poll(self) -> Optional[int]:
        # Wait for the redirection thread to finish, otherwise we would
        # indicate the caller that the command is finished and that the streams
        # are safe to drain, but actually the redirection thread is not
        # finished yet, which would end up in lost data.
        if self.redirect_thread.is_alive():
            return None
        elif self.chan.exit_status_ready():
            return self.wait()
        else:
            return None

    def _cancel(self, kill_timeout: int) -> None:
        self.send_signal(signal.SIGTERM)
        # Check if the command terminated quickly
        time.sleep(10e-3)
        # Otherwise wait for the full timeout and kill it
        if self.poll() is None:
            time.sleep(kill_timeout)
            self.send_signal(signal.SIGKILL)
            self.wait()

    @property
    def stdin(self) -> Optional[IO]:
        return self._stdin

    @property
    def stdout(self) -> Optional[IO]:
        return self._stdout

    @property
    def stderr(self) -> Optional[IO]:
        return self._stderr

    def _close(self) -> int:
        for x in (self.stdin, self.stdout, self.stderr):
            if x is not None:
                x.close()

        exit_code: int = self.wait()
        thread: Thread = self.redirect_thread
        if thread:
            thread.join()

        return exit_code


class AdbBackgroundCommand(BackgroundCommand):
    """
    A background command launched through ADB. Manages signals by sending
    kill commands on the remote Android device.

    :param conn: The ADB-based connection.
    :type conn: ConnectionBase
    :param adb_popen: A subprocess.Popen object representing 'adb shell' or similar.
    :type adb_popen: Popen
    :param pid: Remote process group ID used for signals.
    :type pid: int
    :param as_root: If True, signals are sent as root.
    :type as_root: bool or None
    """

    def __init__(self, conn: 'ConnectionBase', adb_popen: 'Popen',
                 pid: int, as_root: Optional[bool]):
        super().__init__(conn=conn)
        self.as_root = as_root
        self.adb_popen = adb_popen
        self._pid = pid

    def _send_signal(self, sig: 'Signals') -> None:
        self.conn.execute(
            _kill_pgid_cmd(self.pid, sig, self.conn.busybox),
            as_root=self.as_root,
        )

    @property
    def stdin(self) -> Optional[IO]:
        return self.adb_popen.stdin

    @property
    def stdout(self) -> Optional[IO]:
        return self.adb_popen.stdout

    @property
    def stderr(self) -> Optional[IO]:
        return self.adb_popen.stderr

    @property
    def pid(self) -> int:
        return self._pid

    def _wait(self) -> int:
        return self.adb_popen.wait()

    def _communicate(self, input: bytes,
                     timeout: Optional[int]) -> Tuple[Optional[bytes], Optional[bytes]]:
        return _popen_communicate(self, self.adb_popen, input, timeout)

    def _poll(self) -> Optional[int]:
        return self.adb_popen.poll()

    def _cancel(self, kill_timeout: int) -> None:
        self.send_signal(signal.SIGTERM)
        try:
            self.adb_popen.wait(timeout=kill_timeout)
        except subprocess.TimeoutExpired:
            self.send_signal(signal.SIGKILL)
            self.adb_popen.kill()

    def _close(self) -> int:
        self.adb_popen.__exit__(None, None, None)
        return self.adb_popen.returncode

    def __enter__(self):
        super().__enter__()
        self.adb_popen.__enter__()
        return self


class TransferManager:
    """
    Monitors active file transfers (push or pull) in a background thread
    and aborts them if they exceed a time limit or appear inactive.

    :param conn: The ConnectionBase owning this manager.
    :type conn: ConnectionBase
    :param transfer_poll_period: Interval (seconds) between checks for activity.
    :type transfer_poll_period: int
    :param start_transfer_poll_delay: Delay (seconds) before starting to poll a new transfer.
    :type start_transfer_poll_delay: int
    :param total_transfer_timeout: Cancel the transfer if it exceeds this duration.
    :type total_transfer_timeout: int
    """
    def __init__(self, conn: 'ConnectionBase', transfer_poll_period: int = 30,
                 start_transfer_poll_delay: int = 30, total_transfer_timeout: int = 3600):
        self.conn = conn
        self.transfer_poll_period = transfer_poll_period
        self.total_transfer_timeout = total_transfer_timeout
        self.start_transfer_poll_delay = start_transfer_poll_delay

        self.logger = logging.getLogger('FileTransfer')

    @contextmanager
    def manage(self, sources: Tuple[str, ...], dest: str,
               direction: Union[Literal['push'], Literal['pull']],
               handle: 'TransferHandleBase') -> Generator:
        """
        A context manager that spawns a thread to monitor file transfer progress.
        If the transfer stalls or times out, it cancels the operation.

        :param sources: Paths being transferred.
        :type sources: Tuple[str, ...]
        :param dest: Destination path.
        :type dest: str
        :param direction: 'push' or 'pull' for transfer direction.
        :type direction: Literal['push', 'pull']
        :param handle: A TransferHandleBase for polling/canceling.
        :type handle: TransferHandleBase
        :raises TimeoutError: If the transfer times out.
        """
        excep: Optional[TimeoutError] = None
        stop_thread: Event = threading.Event()

        def monitor() -> None:
            """
            thread to monitor the file transfer
            """
            nonlocal excep

            def cancel(reason: str) -> None:
                """
                cancel the file transfer
                """
                self.logger.warning(
                    f'Cancelling file transfer {sources} -> {dest} due to: {reason}'
                )
                handle.cancel()

            start_t = time.monotonic()
            stop_thread.wait(self.start_transfer_poll_delay)
            while not stop_thread.wait(self.transfer_poll_period):
                if not handle.isactive():
                    cancel(reason='transfer inactive')
                elif time.monotonic() - start_t > self.total_transfer_timeout:
                    cancel(reason='transfer timed out')
                    excep = TimeoutError(f'{direction}: {sources} -> {dest}')

        m_thread: Thread = threading.Thread(target=monitor, daemon=True)
        try:
            m_thread.start()
            yield self
        finally:
            stop_thread.set()
            m_thread.join()
            if excep is not None:
                raise excep


class NoopTransferManager:
    """
    A manager that does nothing for transfers. Used if polling is disabled.
    """
    def manage(self, *args, **kwargs) -> nullcontext:
        return nullcontext(self)


class TransferHandleBase(ABC):
    """
    Abstract base for objects tracking a file transfer's progress and allowing cancellations.

    :param manager: The TransferManager that created this handle.
    :type manager: TransferManager
    """
    def __init__(self, manager: 'TransferManager'):
        self.manager = manager

    @property
    def logger(self):
        """
        get the logger for transfer manager
        """
        return self.manager.logger

    @abstractmethod
    def isactive(self) -> bool:
        """
        Check if the transfer still appears to be making progress (return True)
        or if it is idle/complete (return False).
        """
        pass

    @abstractmethod
    def cancel(self) -> None:
        """
        cancel ongoing file transfer
        """
        pass


class PopenTransferHandle(TransferHandleBase):
    """
    File transfer handle implemented using a background command (e.g., scp/rsync).
    It regularly checks the destination size to see if it is increasing.

    :param bg_cmd: The BackgroundCommand driving the file transfer.
    :type bg_cmd: BackgroundCommand
    :param dest: Destination path (local or remote).
    :type dest: str
    :param direction: 'push' or 'pull'.
    :type direction: Literal['push', 'pull']
    """
    def __init__(self, bg_cmd: 'BackgroundCommand', dest: str,
                 direction: Union[Literal['push'], Literal['pull']], *args, **kwargs):
        super().__init__(*args, **kwargs)

        if direction == 'push':
            sample_size: Callable[[str], Optional[int]] = self._push_dest_size
        elif direction == 'pull':
            sample_size = self._pull_dest_size
        else:
            raise ValueError(f'Unknown direction: {direction}')

        self.sample_size = lambda: sample_size(dest)

        self.bg_cmd = bg_cmd
        self.last_sample: int = 0

    @staticmethod
    def _pull_dest_size(dest: str) -> Optional[int]:
        """
        Compute total size of a directory or file at the local ``dest`` path.
        Returns None if it does not exist.
        """
        if os.path.isdir(dest):
            return sum(
                os.stat(os.path.join(dirpath, f)).st_size
                for dirpath, _, fnames in os.walk(dest)
                for f in fnames
            )
        else:
            return os.stat(dest).st_size

    def _push_dest_size(self, dest: str) -> Optional[int]:
        """
        Compute total size of a directory or file on the remote device,
        using busybox du if available.
        """
        conn: 'ConnectionBase' = self.manager.conn
        if conn.busybox:
            cmd: str = '{} du -s -- {}'.format(quote(conn.busybox), quote(dest))
            out: str = conn.execute(cmd)
            return int(out.split()[0])
        return None

    def cancel(self) -> None:
        """
        Cancel the underlying background command, aborting the file transfer.
        """
        self.bg_cmd.cancel()

    def isactive(self) -> bool:
        """
        Check if the file size at the destination has grown since the last poll.
        Returns True if so, otherwise might still be True if we can't read size.
        """
        try:
            curr_size: Optional[int] = self.sample_size()
        except Exception as e:
            self.logger.debug(f'File size polling failed: {e}')
            return True
        else:
            self.logger.debug(f'Polled file transfer, destination size: {curr_size}')
            if curr_size:
                active: bool = curr_size > self.last_sample
                self.last_sample = curr_size
                return active
            # If the file is empty it will never grow in size, so we assume
            # everything is going well.
            else:
                return True


class SSHTransferHandle(TransferHandleBase):
    """
    SCP or SFTP-based file transfer handle that uses a callback to track progress.

    :param handle: The SCPClient or SFTPClient controlling the file transfer.
    :type handle: SCPClient or SFTPClient
    """

    def __init__(self, handle: Union['SCPClient', 'SFTPClient'], *args, **kwargs):
        super().__init__(*args, **kwargs)

        # SFTPClient or SSHClient
        self.handle = handle

        self.progressed: bool = False
        self.transferred: int = 0
        self.to_transfer: int = 0

    def cancel(self) -> None:
        """
        Close the underlying SCP or SFTP client, presumably aborting the transfer.
        """
        self.handle.close()

    def isactive(self):
        """
        Return True if we've seen progress since last poll, otherwise False.
        """
        progressed = self.progressed
        if progressed:
            self.progressed = False
            pc = (self.transferred / self.to_transfer) * 100
            self.logger.debug(
                f'Polled transfer: {pc:.2f}% [{self.transferred}B/{self.to_transfer}B]'
            )
        return progressed

    def progress_cb(self, transferred: int, to_transfer: int) -> None:
        """
        Callback to be called by the SCP/SFTP library on each progress update.

        :param transferred: Bytes transferred so far.
        :type transferred: int
        :param to_transfer: Total bytes to transfer, or 0 if unknown.
        :type to_transfer: int
        """
        self.progressed = True
        self.transferred = transferred
        self.to_transfer = to_transfer
