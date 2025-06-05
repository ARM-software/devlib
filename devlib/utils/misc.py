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
Miscellaneous functions that don't fit anywhere else.

"""
from contextlib import contextmanager
from functools import partial, reduce, wraps
from itertools import groupby
from operator import itemgetter
from weakref import WeakSet
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError, MarkedYAMLError
from devlib.utils.annotation_helpers import SubprocessCommand

import ctypes
import logging
import os
import pkgutil
import random
import re
import string
import subprocess
import sys
import threading
import types
import warnings
import wrapt

try:
    from contextlib import ExitStack
except AttributeError:
    from contextlib2 import ExitStack  # type: ignore

from shlex import quote

# pylint: disable=redefined-builtin
from devlib.exception import HostError, TimeoutError
from typing import (Union, List, Optional, Tuple, Set,
                    Any, Callable, Dict, TYPE_CHECKING,
                    Type, cast)
from collections.abc import Generator
from typing_extensions import Literal
if TYPE_CHECKING:
    from logging import Logger
    from tarfile import TarFile, TarInfo
    from devlib.target import Target


# ABI --> architectures list
ABI_MAP: Dict[str, List[str]] = {
    'armeabi': ['armeabi', 'armv7', 'armv7l', 'armv7el', 'armv7lh', 'armeabi-v7a'],
    'arm64': ['arm64', 'armv8', 'arm64-v8a', 'aarch64'],
}

# Vendor ID --> CPU part ID --> CPU variant ID --> Core Name
# None means variant is not used.
CPU_PART_MAP: Dict[int, Dict[int, Dict[Optional[int], str]]] = {
    0x41: {  # ARM
        0x926: {None: 'ARM926'},
        0x946: {None: 'ARM946'},
        0x966: {None: 'ARM966'},
        0xb02: {None: 'ARM11MPCore'},
        0xb36: {None: 'ARM1136'},
        0xb56: {None: 'ARM1156'},
        0xb76: {None: 'ARM1176'},
        0xc05: {None: 'A5'},
        0xc07: {None: 'A7'},
        0xc08: {None: 'A8'},
        0xc09: {None: 'A9'},
        0xc0e: {None: 'A17'},
        0xc0f: {None: 'A15'},
        0xc14: {None: 'R4'},
        0xc15: {None: 'R5'},
        0xc17: {None: 'R7'},
        0xc18: {None: 'R8'},
        0xc20: {None: 'M0'},
        0xc60: {None: 'M0+'},
        0xc21: {None: 'M1'},
        0xc23: {None: 'M3'},
        0xc24: {None: 'M4'},
        0xc27: {None: 'M7'},
        0xd01: {None: 'A32'},
        0xd03: {None: 'A53'},
        0xd04: {None: 'A35'},
        0xd07: {None: 'A57'},
        0xd08: {None: 'A72'},
        0xd09: {None: 'A73'},
    },
    0x42: {  # Broadcom
        0x516: {None: 'Vulcan'},
    },
    0x43: {  # Cavium
        0x0a1: {None: 'Thunderx'},
        0x0a2: {None: 'Thunderx81xx'},
    },
    0x4e: {  # Nvidia
        0x0: {None: 'Denver'},
    },
    0x50: {  # AppliedMicro
        0x0: {None: 'xgene'},
    },
    0x51: {  # Qualcomm
        0x02d: {None: 'Scorpion'},
        0x04d: {None: 'MSM8960'},
        0x06f: {  # Krait
            0x2: 'Krait400',
            0x3: 'Krait450',
        },
        0x205: {0x1: 'KryoSilver'},
        0x211: {0x1: 'KryoGold'},
        0x800: {None: 'Falkor'},
    },
    0x53: {  # Samsung LSI
        0x001: {0x1: 'MongooseM1'},
    },
    0x56: {  # Marvell
        0x131: {
            0x2: 'Feroceon 88F6281',
        }
    },
}


def get_cpu_name(implementer: int, part: int, variant: int) -> Optional[str]:
    """
    Retrieve the CPU name based on implementer, part, and variant IDs using the CPU_PART_MAP.

    :param implementer: The vendor identifier.
    :param part: The CPU part identifier.
    :param variant: The CPU variant identifier.
    :returns: The CPU name if found; otherwise, None.
    """
    part_data = CPU_PART_MAP.get(implementer, {}).get(part, {})
    if None in part_data:  # variant does not determine core Name for this vendor
        name: Optional[str] = part_data[None]
    else:
        name = part_data.get(variant)
    return name


def preexec_function() -> None:
    """
    Set the process group ID for the current process so that a subprocess and all its children
    can later be killed together. This function is Unix-specific.

    :raises OSError: If setting the process group fails.
    """
    # Change process group in case we have to kill the subprocess and all of
    # its children later.
    # TODO: this is Unix-specific; would be good to find an OS-agnostic way
    #       to do this in case we wanna port WA to Windows.
    os.setpgrp()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


check_output_logger: 'Logger' = get_logger('check_output')


def get_subprocess(command: SubprocessCommand, **kwargs) -> subprocess.Popen:
    """
    Launch a subprocess to run the specified command, overriding stdout to PIPE.
    The process is set to a new process group via a preexec function.

    :param command: The command to execute.
    :param kwargs: Additional keyword arguments to pass to subprocess.Popen.
    :raises ValueError: If 'stdout' is provided in kwargs.
    :returns: A subprocess.Popen object running the command.
    """
    if 'stdout' in kwargs:
        raise ValueError('stdout argument not allowed, it will be overridden.')
    return subprocess.Popen(command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            stdin=subprocess.PIPE,
                            preexec_fn=preexec_function,
                            **kwargs)


def check_subprocess_output(
        process: subprocess.Popen,
        timeout: Optional[float] = None,
        ignore: Optional[Union[int, List[int], Literal['all']]] = None,
        inputtext: Union[str, bytes, None] = None) -> Tuple[str, str]:
    """
    Communicate with the given subprocess and return its decoded output and error streams.
    This function handles timeouts and can ignore specified return codes.

    :param process: The subprocess.Popen instance to interact with.
    :param timeout: The maximum time in seconds to wait for the process to complete.
    :param ignore: A return code (or list of codes) to ignore; use "all" to ignore all nonzero codes.
    :param inputtext: Optional text or bytes to send to the process's stdin.
    :returns: A tuple (output, error) with decoded strings.
    :raises ValueError: If the ignore parameter is improperly formatted.
    :raises TimeoutError: If the process does not complete before the timeout expires.
    :raises subprocess.CalledProcessError: If the process exits with a nonzero code not in ignore.
    """
    output: Union[str, bytes] = ''
    error: Union[str, bytes] = ''

    # pylint: disable=too-many-branches
    if ignore is None:
        ignore = []
    elif isinstance(ignore, int):
        ignore = [ignore]
    elif not isinstance(ignore, list) and ignore != 'all':
        message = 'Invalid value for ignore parameter: "{}"; must be an int or a list'
        raise ValueError(message.format(ignore))

    with process:
        timeout_expired: Optional[subprocess.TimeoutExpired] = None
        try:
            output, error = process.communicate(inputtext, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            timeout_expired = e

        # Currently errors=replace is needed as 0x8c throws an error
        output = cast(str, output.decode(sys.stdout.encoding or 'utf-8', "replace") if isinstance(output, bytes) else output)
        error = cast(str, error.decode(sys.stderr.encoding or 'utf-8', "replace") if isinstance(error, bytes) else error)

        if timeout_expired:
            raise TimeoutError(process.args, output='\n'.join([output, error]))

    retcode: int = process.returncode
    if retcode and ignore != 'all' and retcode not in ignore:
        raise subprocess.CalledProcessError(retcode, process.args, output, error)

    return output, error


def check_output(command: SubprocessCommand, timeout: Optional[int] = None,
                 ignore: Optional[Union[int, List[int], Literal['all']]] = None,
                 inputtext: Union[str, bytes, None] = None, **kwargs) -> Tuple[str, str]:
    """
    This is a version of subprocess.check_output that adds a timeout parameter to kill
    the subprocess if it does not return within the specified time.

    :param command: The command to execute.
    :param timeout: Time in seconds to wait for the command to complete.
    :param ignore: A return code or list of return codes to ignore, or "all" to ignore all.
    :param inputtext: Optional text or bytes to send to the command's stdin.
    :param kwargs: Additional keyword arguments for subprocess.Popen.
    :returns: A tuple (stdout, stderr) of the command's decoded output.
    :raises TimeoutError: If the command does not complete in time.
    :raises subprocess.CalledProcessError: If the command fails and its return code is not ignored.
    """
    process = get_subprocess(command, **kwargs)
    return check_subprocess_output(process, timeout=timeout, ignore=ignore, inputtext=inputtext)


class ExtendedHostError(HostError):
    """
    Exception class that extends HostError with additional attributes.

    :param message: The error message.
    :param module: The name of the module where the error originated.
    :param exc_info: Exception information from sys.exc_info().
    :param orig_exc: The original exception that was caught.
    """
    def __init__(self, message: str, module: Optional[str] = None,
                 exc_info: Any = None, orig_exc: Optional[Exception] = None):
        super().__init__(message)
        self.module = module
        self.exc_info = exc_info
        self.orig_exc = orig_exc


def walk_modules(path: str) -> List[types.ModuleType]:
    """
    Given package name, return a list of all modules (including submodules, etc)
    in that package.

    :param path: The package name to walk (e.g., 'mypackage').
    :returns: A list of module objects.
    :raises HostError: if an exception is raised while trying to import one of the
                       modules under ``path``. The exception will have addtional
                       attributes set: ``module`` will be set to the qualified name
                       of the originating module, and ``orig_exc`` will contain
                       the original exception.

    """

    def __try_import(path: str) -> types.ModuleType:
        try:
            return __import__(path, {}, {}, [''])
        except Exception as e:
            he = HostError('Could not load {}: {}'.format(path, str(e)))
            cast(ExtendedHostError, he).module = path
            cast(ExtendedHostError, he).exc_info = sys.exc_info()
            cast(ExtendedHostError, he).orig_exc = e
            raise he

    root_mod: types.ModuleType = __try_import(path)
    mods: List[types.ModuleType] = [root_mod]
    if not hasattr(root_mod, '__path__'):
        # root is a module not a package -- nothing to walk
        return mods
    for _, name, ispkg in pkgutil.iter_modules(root_mod.__path__):
        submod_path: str = '.'.join([path, name])
        if ispkg:
            mods.extend(walk_modules(submod_path))
        else:
            submod: types.ModuleType = __try_import(submod_path)
            mods.append(submod)
    return mods


def redirect_streams(stdout: int, stderr: int,
                     command: SubprocessCommand) -> Tuple[int, int, SubprocessCommand]:
    """
    Adjust a command string to redirect output streams to specific targets.
    If a stream is set to subprocess.DEVNULL, it replaces it with a redirect
    to /dev/null; for subprocess.STDOUT, it merges stderr into stdout.

    :param stdout: The desired stdout value.
    :param stderr: The desired stderr value.
    :param command: The original command to run.

    :return: A tuple (stdout, stderr, command) with stream set to ``subprocess.PIPE``
        if the `stream` parameter was set to ``subprocess.DEVNULL``.
    """

    def redirect(stream: int, redirection: str) -> Tuple[int, str]:
        """
        redirect output and error streams
        """
        if stream == subprocess.DEVNULL:
            suffix = '{}/dev/null'.format(redirection)
        elif stream == subprocess.STDOUT:
            suffix = '{}&1'.format(redirection)
            # Indicate that there is nothing to monitor for stderr anymore
            # since it's merged into stdout
            stream = subprocess.DEVNULL
        else:
            suffix = ''

        return (stream, suffix)

    stdout, suffix1 = redirect(stdout, '>')
    stderr, suffix2 = redirect(stderr, '2>')

    command = 'sh -c {} {} {}'.format(quote(cast(str, command)), suffix1, suffix2)
    return (stdout, stderr, command)


def ensure_directory_exists(dirpath: str) -> str:
    """A filter for directory paths to ensure they exist."""
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    return dirpath


def ensure_file_directory_exists(filepath: str) -> str:
    """
    A filter for file paths to ensure the directory of the
    file exists and the file can be created there. The file
    itself is *not* going to be created if it doesn't already
    exist.

    :param dirpath: The directory path to check.
    :returns: The directory path.
    :raises OSError: If the directory cannot be created
    """
    ensure_directory_exists(os.path.dirname(filepath))
    return filepath


def merge_dicts(*args, **kwargs) -> Dict:
    """
    Merge multiple dictionaries together.

    :param args: Two or more dictionaries to merge.
    :param kwargs: Additional keyword arguments to pass to the merging function.
    :returns: A new dictionary containing the merged keys and values.
    :raises ValueError: If fewer than two dictionaries are provided.
    """
    if not len(args) >= 2:
        raise ValueError('Must specify at least two dicts to merge.')
    func: partial[Dict] = partial(_merge_two_dicts, **kwargs)
    return reduce(func, args)


def _merge_two_dicts(base: Dict, other: Dict, list_duplicates: str = 'all',
                     match_types: bool = False,  # pylint: disable=R0912,R0914
                     dict_type: Type[Dict] = dict, should_normalize: bool = True,
                     should_merge_lists: bool = True) -> Dict:
    """
    Merge two dictionaries recursively, normalizing their keys. The merging behavior
    for lists and duplicate keys can be controlled via parameters.

    :param base: The first dictionary.
    :param other: The second dictionary to merge into the first.
    :param list_duplicates: Strategy for handling duplicate list entries ("all", "first", or "last").
    :param match_types: If True, enforce that overlapping keys have the same type.
    :param dict_type: The dictionary type to use for constructing merged dictionaries.
    :param should_normalize: If True, normalize keys/values during merge.
    :param should_merge_lists: If True, merge lists; otherwise, override base list.
    :returns: A merged dictionary.
    :raises ValueError: If there is a type mismatch for a key when match_types is True.
    :raises AssertionError: If an unexpected merge key is encountered.
    """
    merged = dict_type()
    base_keys = list(base.keys())
    other_keys = list(other.keys())
    # FIXME - annotate the lambda. type checker is not able to deduce its type
    norm: Callable = normalize if should_normalize else lambda x, y: x       # type:ignore

    base_only: List = []
    other_only: List = []
    both: List = []
    union: List = []
    for k in base_keys:
        if k in other_keys:
            both.append(k)
        else:
            base_only.append(k)
            union.append(k)
    for k in other_keys:
        if k in base_keys:
            union.append(k)
        else:
            union.append(k)
            other_only.append(k)

    for k in union:
        if k in base_only:
            merged[k] = norm(base[k], dict_type)
        elif k in other_only:
            merged[k] = norm(other[k], dict_type)
        elif k in both:
            base_value = base[k]
            other_value = other[k]
            base_type = type(base_value)
            other_type = type(other_value)
            if (match_types and (base_type != other_type) and
                    (base_value is not None) and (other_value is not None)):
                raise ValueError('Type mismatch for {} got {} ({}) and {} ({})'.format(k, base_value, base_type,
                                                                                       other_value, other_type))
            if isinstance(base_value, dict):
                merged[k] = _merge_two_dicts(base_value, other_value, list_duplicates, match_types, dict_type)
            elif isinstance(base_value, list):
                if should_merge_lists:
                    merged[k] = _merge_two_lists(base_value, other_value, list_duplicates, dict_type)
                else:
                    merged[k] = _merge_two_lists([], other_value, list_duplicates, dict_type)

            elif isinstance(base_value, set):
                merged[k] = norm(base_value.union(other_value), dict_type)
            else:
                merged[k] = norm(other_value, dict_type)
        else:  # Should never get here
            raise AssertionError('Unexpected merge key: {}'.format(k))

    return merged


def merge_lists(*args, **kwargs) -> List:
    """
    Merge multiple lists together.

    :param args: Two or more lists to merge.
    :param kwargs: Additional keyword arguments to pass to the merging function.
    :returns: A merged list containing the combined items.
    :raises ValueError: If fewer than two lists are provided.
    """
    if not len(args) >= 2:
        raise ValueError('Must specify at least two lists to merge.')
    func = partial(_merge_two_lists, **kwargs)
    return reduce(func, args)


def _merge_two_lists(base: List, other: List, duplicates: str = 'all',
                     dict_type: Type[Dict] = dict) -> List:  # pylint: disable=R0912
    """
    Merge lists, normalizing their entries.

    :param base: The base list.
    :param other: The list to merge into base.
    :param duplicates: Indicates the strategy of handling entries that appear
                    in both lists. ``all`` will keep occurrences from both
                    lists; ``first`` will only keep occurrences from
                    ``base``; ``last`` will only keep occurrences from
                    ``other``;

    .. note:: duplicate entries that appear in the *same* list
                               will never be removed.
    :param dict_type: The dictionary type to use for normalization if needed.
    :returns: A merged list with duplicate handling applied.
    :raises ValueError: If an unexpected value is provided for duplicates.
    """
    if not isiterable(base):
        base = [base]
    if not isiterable(other):
        other = [other]
    if duplicates == 'all':
        merged_list: List = []
        combined: List = []
        normalized_base = normalize(base, dict_type)
        normalized_other = normalize(other, dict_type)
        if isinstance(normalized_base, (list, tuple)) and isinstance(normalized_other, (list, tuple)):
            combined = list(normalized_base) + list(normalized_other)
        elif isinstance(normalized_base, dict) and isinstance(normalized_other, dict):
            combined = [normalized_base, normalized_other]
        elif isinstance(normalized_base, set) and isinstance(normalized_other, set):
            combined = list(normalized_base.union(normalized_other))
        else:
            combined = list(normalized_base) + list(normalized_other)
        for v in combined:
            if not _check_remove_item(merged_list, v):
                merged_list.append(v)
        return merged_list
    elif duplicates == 'first':
        base_norm = normalize(base, dict_type)
        merged_list = cast(List, normalize(base, dict_type))
        for v in base_norm:
            _check_remove_item(merged_list, v)
        for v in normalize(other, dict_type):
            if not _check_remove_item(merged_list, v):
                if v not in base_norm:
                    cast(List, merged_list).append(v)  # pylint: disable=no-member
        return merged_list
    elif duplicates == 'last':
        other_norm = normalize(other, dict_type)
        merged_list = []
        for v in normalize(base, dict_type):
            if not _check_remove_item(merged_list, v):
                if v not in other_norm:
                    merged_list.append(v)
        for v in other_norm:
            if not _check_remove_item(merged_list, v):
                merged_list.append(v)
        return merged_list
    else:
        raise ValueError('Unexpected value for list duplicates argument: {}. '.format(duplicates) +
                         'Must be in {"all", "first", "last"}.')


def _check_remove_item(the_list: List, item: Any) -> bool:
    """
    Check whether an item should be removed from a list based on certain criteria.
    If the item is a string starting with '~', its unprefixed version is removed from the list.

    :param the_list: The list in which to check for the item.
    :param item: The item to check.
    :returns: True if the item was removed; False otherwise.
    """
    if not isinstance(item, str):
        return False
    if not item.startswith('~'):
        return False
    actual_item = item[1:]
    if actual_item in the_list:
        del the_list[the_list.index(actual_item)]
    return True


def normalize(value: Union[Dict, List, Tuple, Set],
              dict_type: Type[Dict] = dict) -> Union[Dict, List, Tuple, Set]:
    """
    Recursively normalize values by converting dictionary keys to lower-case,
    stripping whitespace, and replacing spaces with underscores.

    :param value: A dict, list, tuple, or set to normalize.
    :param dict_type: The dictionary type to use for normalized dictionaries.
    :returns: A normalized version of the input value.
    """
    if isinstance(value, dict):
        normalized = dict_type()
        for k, v in value.items():
            key = k.strip().lower().replace(' ', '_')
            normalized[key] = normalize(v, dict_type)
        return normalized
    elif isinstance(value, list):
        return [normalize(v, dict_type) for v in value]
    elif isinstance(value, tuple):
        return tuple([normalize(v, dict_type) for v in value])
    else:
        return value


def convert_new_lines(text: str) -> str:
    """
    Convert different newline conventions to a single '\n' format.

    :param text: The input text.
    :returns: The text with unified newline characters.
    """
    return text.replace('\r\n', '\n').replace('\r', '\n')


def sanitize_cmd_template(cmd: str) -> str:
    """
    Replace quoted placeholders with unquoted ones in a command template,
    warning the user if quoted placeholders are detected.

    :param cmd: The command template string.
    :returns: The sanitized command template.
    """
    msg: str = (
        '''Quoted placeholder should not be used, as it will result in quoting the text twice. {} should be used instead of '{}' or "{}" in the template: '''
    )
    for unwanted in ('"{}"', "'{}'"):
        if unwanted in cmd:
            warnings.warn(msg + cmd, stacklevel=2)
            cmd = cmd.replace(unwanted, '{}')

    return cmd


def escape_quotes(text: str) -> str:
    """
    Escape quotes and escaped quotes in the given text.

    .. note:: It is recommended to use shlex.quote when possible.

    :param text: The text to escape.
    :returns: The text with quotes escaped.
    """
    return re.sub(r'\\("|\')', r'\\\\\1', text).replace('\'', '\\\'').replace('\"', '\\\"')


def escape_single_quotes(text: str) -> str:
    """
    Escape single quotes in the provided text.

    .. note:: Prefer using shlex.quote when possible.

    :param text: The text to process.
    :returns: The text with single quotes escaped.
    """
    return re.sub(r'\\("|\')', r'\\\\\1', text).replace('\'', '\'\\\'\'')


def escape_double_quotes(text: str) -> str:
    """
    Escape double quotes in the given text.

    .. note:: Prefer using shlex.quote when possible.

    :param text: The input text.
    :returns: The text with double quotes escaped.
    """
    return re.sub(r'\\("|\')', r'\\\\\1', text).replace('\"', '\\\"')


def escape_spaces(text: str) -> str:
    """
    Escape spaces in the provided text.

    .. note:: Prefer using shlex.quote when possible.

    :param text: The text to process.
    :returns: The text with spaces escaped.
    """
    return text.replace(' ', '\\ ')


def getch(count: int = 1) -> str:
    """
    Read a specified number of characters from standard input.

    :param count: The number of characters to read.
    :returns: A string of characters read from stdin.
    """
    if os.name == 'nt':
        import msvcrt  # pylint: disable=F0401
        return ''.join([msvcrt.getch() for _ in range(count)])  # type:ignore
    else:  # assume Unix
        import tty  # NOQA
        import termios  # NOQA
        fd: int = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(count)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


def isiterable(obj: Any) -> bool:
    """
    Determine if the provided object is iterable, excluding strings.

    :param obj: The object to test.
    :returns: True if the object is iterable and is not a string; otherwise, False.
    """
    return hasattr(obj, '__iter__') and not isinstance(obj, str)


def as_relative(path: str) -> str:
    """
    Convert an absolute path to a relative path by removing leading separators.

    :param path: The absolute path.
    :returns: A relative path.
    """
    path = os.path.splitdrive(path)[1]
    return path.lstrip(os.sep)


def commonprefix(file_list: List[str], sep: str = os.sep) -> str:
    """
    Determine the lowest common base folder among a list of file paths.

    :param file_list: A list of file paths.
    :param sep: The path separator to use.
    :returns: The common prefix path.
    """
    common_path: str = os.path.commonprefix(file_list)
    cp_split: List[str] = common_path.split(sep)
    other_split: List[str] = file_list[0].split(sep)
    last: int = len(cp_split) - 1
    if cp_split[last] != other_split[last]:
        cp_split = cp_split[:-1]
    return sep.join(cp_split)


def get_cpu_mask(cores: List[int]) -> str:
    """
    Compute a hexadecimal CPU mask for the specified core indices.

    :param cores: A list of core numbers.
    :returns: A hexadecimal string representing the CPU mask.
    """
    mask = 0
    for i in cores:
        mask |= 1 << i
    return '0x{0:x}'.format(mask)


def which(name: str) -> Optional[str]:
    """
    Find the full path to an executable by searching the system PATH.
    Provides a platform-independent implementation of the UNIX 'which' utility.

    :param name: The name of the executable to find.
    :returns: The full path to the executable if found, otherwise None.
    """
    if os.name == 'nt':
        path_env = os.getenv('PATH')
        pathext_env = os.getenv('PATHEXT')
        paths: List[str] = path_env.split(os.pathsep) if path_env else []
        exts: List[str] = pathext_env.split(os.pathsep) if pathext_env else []
        for path in paths:
            testpath = os.path.join(path, name)
            if os.path.isfile(testpath):
                return testpath
            for ext in exts:
                testpathext = testpath + ext
                if os.path.isfile(testpathext):
                    return testpathext
        return None
    else:  # assume UNIX-like
        try:
            return check_output(['which', name])[0].strip()  # pylint: disable=E1103
        except subprocess.CalledProcessError:
            return None


# This matches most ANSI escape sequences, not just colors
_bash_color_regex = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')


def strip_bash_colors(text: str) -> str:
    """
    Remove ANSI escape sequences (commonly used for terminal colors) from the given text.

    :param text: The input string potentially containing ANSI escape sequences.
    :returns: The input text with all ANSI escape sequences removed.
    """
    return _bash_color_regex.sub('', text)


def get_random_string(length: int) -> str:
    """Returns a random ASCII string of the specified length)."""
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


class LoadSyntaxError(Exception):

    @property
    def message(self):
        if self.args:
            return self.args[0]
        return str(self)

    def __init__(self, message: str, filepath: str, lineno: Optional[int]):
        super(LoadSyntaxError, self).__init__(message)
        self.filepath = filepath
        self.lineno = lineno

    def __str__(self):
        message = 'Syntax Error in {}, line {}:\n\t{}'
        return message.format(self.filepath, self.lineno, self.message)


def load_struct_from_yaml(filepath: str) -> Dict:
    """
    Parses a config structure from a YAML file.
    The structure should be composed of basic Python types.

    :param filepath: Input file which contains YAML data.

    :raises LoadSyntaxError: if there is a syntax error in YAML data.

    :return: A dictionary which contains parsed YAML data
    """

    try:
        yaml = YAML(typ='safe', pure=True)
        with open(filepath, 'r', encoding='utf-8') as file_handler:
            return yaml.load(file_handler)
    except YAMLError as ex:
        message = str(ex)
        lineno = cast(MarkedYAMLError, ex).problem_mark.line if hasattr(ex, 'problem_mark') else None
        raise LoadSyntaxError(message, filepath=filepath, lineno=lineno) from ex


RAND_MOD_NAME_LEN: int = 30
BAD_CHARS: str = string.punctuation + string.whitespace
TRANS_TABLE: Dict[int, int] = str.maketrans(BAD_CHARS, '_' * len(BAD_CHARS))


def to_identifier(text: str) -> str:
    """Converts text to a valid Python identifier by replacing all
    whitespace and punctuation and adding a prefix if starting with a digit"""
    if text[:1].isdigit():
        text = '_' + text
    return re.sub('_+', '_', str(text).translate(TRANS_TABLE))


def unique(alist: List) -> List:
    """
    Returns a list containing only unique elements from the input list (but preserves
    order, unlike sets).

    """
    result = []
    for item in alist:
        if item not in result:
            result.append(item)
    return result


def ranges_to_list(ranges_string: str) -> List[int]:
    """Converts a sysfs-style ranges string, e.g. ``"0,2-4"``, into a list ,e.g ``[0,2,3,4]``"""
    values: List[int] = []
    for rg in ranges_string.split(','):
        if '-' in rg:
            first, last = list(map(int, rg.split('-')))
            values.extend(range(first, last + 1))
        else:
            values.append(int(rg))
    return values


def list_to_ranges(values: List) -> str:
    """Converts a list, e.g ``[0,2,3,4]``, into a sysfs-style ranges string, e.g. ``"0,2-4"``"""
    values = sorted(values)
    range_groups = []
    for _, g in groupby(enumerate(values), lambda i_x: i_x[0] - i_x[1]):
        range_groups.append(list(map(itemgetter(1), g)))
    range_strings: List[str] = []
    for group in range_groups:
        if len(group) == 1:
            range_strings.append(str(group[0]))
        else:
            range_strings.append('{}-{}'.format(group[0], group[-1]))
    return ','.join(range_strings)


def list_to_mask(values: List[int], base: int = 0x0) -> int:
    """Converts the specified list of integer values into
    a bit mask for those values. Optinally, the list can be
    applied to an existing mask."""
    for v in values:
        base |= (1 << v)
    return base


def mask_to_list(mask: int) -> List[int]:
    """Converts the specfied integer bitmask into a list of
    indexes of bits that are set in the mask."""
    size = len(bin(mask)) - 2  # because of "0b"
    return [size - i - 1 for i in range(size)
            if mask & (1 << size - i - 1)]


__memo_cache: Dict[str, Any] = {}


def reset_memo_cache() -> None:
    """
    Clear the global memoization cache used for caching function results.

    :returns: None
    """
    __memo_cache.clear()


def __get_memo_id(obj: object) -> str:
    """
    An object's id() may be re-used after an object is freed, so it's not
    sufficiently unique to identify params for the memo cache (two different
    params may end up with the same id). this attempts to generate a more unique
    ID string.
    """
    obj_id: int = id(obj)
    try:
        return '{}/{}'.format(obj_id, hash(obj))
    except TypeError:  # obj is not hashable
        obj_pyobj = ctypes.cast(obj_id, ctypes.py_object)
        # TODO: Note: there is still a possibility of a clash here. If Two
        # different objects get assigned the same ID, and are large and are
        # identical in the first thirty two bytes. This shouldn't be much of an
        # issue in the current application of memoizing Target calls, as it's very
        # unlikely that a target will get passed large params; but may cause
        # problems in other applications, e.g. when memoizing results of operations
        # on large arrays. I can't really think of a good way around that apart
        # form, e.g., md5 hashing the entire raw object, which will have an
        # undesirable impact on performance.
        num_bytes = min(ctypes.sizeof(obj_pyobj), 32)
        obj_bytes = ctypes.string_at(ctypes.addressof(obj_pyobj), num_bytes)
        return '{}/{}'.format(obj_id, cast(str, obj_bytes))


def memoized_decor(wrapped: Callable[..., Any], instance: Optional[Any],
                   args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:  # pylint: disable=unused-argument
    """
    Decorator helper function for memoizing the results of a function call.
    The result is cached based on a key derived from the function's arguments.
    Note that this method does not account for changes to mutable arguments.

    .. warning:: this may not detect changes to mutable types. As long as the
                 memoized function was used with an object as an argument
                 before, the cached result will be returned, even if the
                 structure of the object (e.g. a list) has changed in the mean time.

    :param wrapped: The function to be memoized.
    :param instance: The instance on which the function is called (if it is a method), or None.
    :param args: Tuple of positional arguments passed to the function.
    :param kwargs: Dictionary of keyword arguments passed to the function.
    :returns: The cached result if available; otherwise, the result from calling the function.
    :raises Exception: Any exception raised during the execution of the wrapped function is propagated.

    """
    func_id: str = repr(wrapped)

    def memoize_wrapper(*args, **kwargs) -> Dict[str, Any]:
        id_string: str = func_id + ','.join([__get_memo_id(a) for a in args])
        id_string += ','.join('{}={}'.format(k, __get_memo_id(v))
                              for k, v in kwargs.items())
        if id_string not in __memo_cache:
            __memo_cache[id_string] = wrapped(*args, **kwargs)
        return __memo_cache[id_string]

    return memoize_wrapper(*args, **kwargs)


# create memoized decorator from memoized_decor function
memoized = wrapt.decorator(memoized_decor)


@contextmanager
def batch_contextmanager(f: Callable, kwargs_list: List[Dict[str, Any]]) -> Generator:
    """
    Return a context manager that will call the ``f`` callable with the keyword
    arguments dict in the given list, in one go.

    :param f: Callable expected to return a context manager.

    :param kwargs_list: list of kwargs dictionaries to be used to call ``f``.
    """
    with ExitStack() as stack:
        for kwargs in kwargs_list:
            stack.enter_context(f(**kwargs))
        yield


class nullcontext:
    """
    Backport of Python 3.7 ``contextlib.nullcontext``

    This context manager does nothing, so it can be used as a default
    placeholder for code that needs to select at runtime what context manager
    to use.

    :param enter_result: Object that will be bound to the target of the with
        statement, or `None` if nothing is specified.
    """

    def __init__(self, enter_result: Any = None):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    async def __aenter__(self):
        return self.enter_result

    def __exit__(*_):
        return

    async def __aexit__(*_):
        return


class tls_property:
    """
    Use it like `property` decorator, but the result will be memoized per
    thread. When the owning thread dies, the values for that thread will be
    destroyed.

    In order to get the values, it's necessary to call the object
    given by the property. This is necessary in order to be able to add methods
    to that object, like :meth:`_BoundTLSProperty.get_all_values`.

    Values can be set and deleted as well, which will be a thread-local set.

    :param factory: A callable used to generate the property value.
    """

    @property
    def name(self) -> str:
        """
        Retrieve the name of the factory function used for this property.

        :returns: The name of the factory function.
        """
        return self.factory.__name__

    def __init__(self, factory: Callable):
        self.factory = factory
        # Lock accesses to shared WeakKeyDictionary and WeakSet
        self.lock = threading.RLock()

    def __get__(self, instance: 'Target', owner: Optional[Type['Target']] = None) -> '_BoundTLSProperty':
        """
        Retrieve the thread-local property proxy for the given instance.

        :param instance: The target instance.
        :param owner: The class owning the property (optional).
        :returns: A bound TLS property proxy.
        """
        return _BoundTLSProperty(self, instance, owner)

    def _get_value(self, instance: 'Target', owner: Optional[Type['Target']]) -> Any:
        """
        Retrieve or compute the thread-local value for the given instance. If the value
        does not exist, it is created using the factory callable.

        :param instance: The target instance.
        :param owner: The class owning the property (optional).
        :returns: The thread-local value.
        """
        tls, values = self._get_tls(instance)
        try:
            return tls.value
        except AttributeError:
            # Bind the method to `instance`
            f = self.factory.__get__(instance, owner)
            obj = f()
            tls.value = obj
            # Since that's a WeakSet, values will be removed automatically once
            # the threading.local variable that holds them is destroyed
            with self.lock:
                values.add(obj)
            return obj

    def _get_all_values(self, instance: 'Target', owner: Optional[Type['Target']]) -> Set:
        """
        Retrieve all thread-local values currently cached for this property in the given instance.

        :param instance: The target instance.
        :param owner: The class owning the property (optional).
        :returns: A set containing all cached values.
        """
        with self.lock:
            # Grab a reference to all the objects at the time of the call by
            # using a regular set
            tls, values = self._get_tls(instance=instance)
            return set(values)

    def __set__(self, instance: 'Target', value):
        """
        Set the thread-local value for this property on the given instance.

        :param instance: The target instance.
        :param value: The value to set.
        """
        tls, values = self._get_tls(instance)
        tls.value = value
        with self.lock:
            values.add(value)

    def __delete__(self, instance: 'Target'):
        """
        Delete the thread-local value for this property from the given instance.

        :param instance: The target instance.
        """
        tls, values = self._get_tls(instance)
        with self.lock:
            try:
                value = tls.value
            except AttributeError:
                pass
            else:
                values.discard(value)
                del tls.value

    def _get_tls(self, instance: 'Target') -> Any:
        """
        Retrieve the thread-local storage tuple for this property from the instance.
        If not present, a new tuple is created and stored.

        :param instance: The target instance.
        :returns: A tuple (tls, values) where tls is a thread-local object and values is a WeakSet.
        """
        dct = instance.__dict__
        name = self.name
        try:
            # Using instance.__dict__[self.name] is safe as
            # getattr(instance, name) will return the property instead, as
            # the property is a descriptor
            tls = dct[name]
        except KeyError:
            with self.lock:
                # Double check after taking the lock to avoid a race
                if name not in dct:
                    tls = (threading.local(), WeakSet())
                    dct[name] = tls

        return tls

    @property
    def basic_property(self) -> property:
        """
        Return a basic property that can be used to access the TLS value
        without having to call it first.

        The drawback is that it's not possible to do anything over than
        getting/setting/deleting.

        :returns: A property object for direct access.
        """

        def getter(instance, owner=None):
            prop = self.__get__(instance, owner)
            return prop()

        return property(getter, self.__set__, self.__delete__)


class _BoundTLSProperty:
    """
    Simple proxy object to allow either calling it to get the TLS value, or get
    some other informations by calling methods.

    :param tls_property: The tls_property descriptor.
    :param instance: The target instance to which the property is bound.
    :param owner: The owning class (optional).
    """

    def __init__(self, tls_property: tls_property, instance: 'Target', owner: Optional[Type['Target']]):
        self.tls_property = tls_property
        self.instance = instance
        self.owner = owner

    def __call__(self):
        """
        Retrieve the thread-local value by calling the underlying tls_property.

        :returns: The thread-local value.
        """
        return self.tls_property._get_value(
            instance=self.instance,
            owner=self.owner,
        )

    def get_all_values(self) -> Set[Any]:
        """
        Returns all the thread-local values currently in use in the process for
        that property for that instance.

        :returns: A set of all thread-local values.
        """
        return self.tls_property._get_all_values(
            instance=self.instance,
            owner=self.owner,
        )


class InitCheckpointMeta(type):
    """
    Metaclass providing an ``initialized`` and ``is_in_use`` boolean attributes
    on instances.

    ``initialized`` is set to ``True`` once the ``__init__`` constructor has
    returned. It will deal cleanly with nested calls to ``super().__init__``.

    ``is_in_use`` is set to ``True`` when an instance method is being called.
    This allows to detect reentrance.
    """

    def __new__(metacls, name: str, bases: Tuple, dct: Dict, **kwargs: Dict) -> Type:
        """
        Create a new class with the augmented __init__ and methods for tracking initialization
        and usage.

        :param name: The name of the new class.
        :param bases: Base classes for the new class.
        :param dct: Dictionary of attributes for the new class.
        :param kwargs: Additional keyword arguments.
        :returns: The newly created class.
        """
        cls = super().__new__(metacls, name, bases, dct, **kwargs)
        init_f = cls.__init__   # type:ignore

        @wraps(init_f)
        def init_wrapper(self, *args, **kwargs):
            self.initialized = False
            self.is_in_use = False

            # Track the nesting of super()__init__ to set initialized=True only
            # when the outer level is finished
            try:
                stack = self._init_stack
            except AttributeError:
                stack = []
                self._init_stack = stack

            stack.append(init_f)
            try:
                x = init_f(self, *args, **kwargs)
            finally:
                stack.pop()

            if not stack:
                self.initialized = True
                del self._init_stack

            return x

        cls.__init__ = init_wrapper  # type:ignore

        # Set the is_in_use attribute to allow external code to detect if the
        # methods are about to be re-entered.
        def make_wrapper(f):
            if f is None:
                return None

            @wraps(f)
            def wrapper(self, *args, **kwargs):
                f_ = f.__get__(self, self.__class__)
                initial_state = self.is_in_use
                try:
                    self.is_in_use = True
                    return f_(*args, **kwargs)
                finally:
                    self.is_in_use = initial_state

            return wrapper

        # This will not decorate methods defined in base classes, but we cannot
        # use inspect.getmembers() as it uses __get__ to bind the attributes to
        # the class, making staticmethod indistinguishible from instance
        # methods.
        for name, attr in cls.__dict__.items():
            # Only wrap the methods (exposed as functions), not things like
            # classmethod or staticmethod
            if (
                    name not in ('__init__', '__new__') and
                    isinstance(attr, types.FunctionType)
            ):
                setattr(cls, name, make_wrapper(attr))
            elif isinstance(attr, property):
                prop = property(
                    fget=make_wrapper(attr.fget),
                    fset=make_wrapper(attr.fset),
                    fdel=make_wrapper(attr.fdel),
                    doc=attr.__doc__,
                )
                setattr(cls, name, prop)

        return cls


class InitCheckpoint(metaclass=InitCheckpointMeta):
    """
    Inherit from this class to set the :class:`InitCheckpointMeta` metaclass.
    """
    pass


def groupby_value(dct: Dict[Any, Any]) -> Dict[Tuple[Any, ...], Any]:
    """
    Process the input dict such that all keys sharing the same values are
    grouped in a tuple, used as key in the returned dict.
    """
    key = itemgetter(1)
    items = sorted(dct.items(), key=key)
    return {
        tuple(map(itemgetter(0), _items)): v
        for v, _items in groupby(items, key=key)
    }


def safe_extract(tar: 'TarFile', path: str = ".", members: Optional[List['TarInfo']] = None,
                 *, numeric_owner: bool = False) -> None:
    """
    A wrapper around TarFile.extract all to mitigate CVE-2007-4995
    (see https://www.trellix.com/en-us/about/newsroom/stories/research/tarfile-exploiting-the-world.html)
    """

    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not _is_within_directory(path, member_path):
            raise Exception("Attempted Path Traversal in Tar File")

    tar.extractall(path, members, numeric_owner=numeric_owner)


def _is_within_directory(directory: str, target: str) -> bool:
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)

    prefix = os.path.commonprefix([abs_directory, abs_target])

    return prefix == abs_directory
