#    Copyright 2019 ARM Limited
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
import itertools
import shlex


class Command(dict):  # inherit from dict for JSON serializability
    """Provides an abstraction for manipulating CLI commands

    The expected format of the abstracted command is as follows::

        <command> <flags> <kwflags> <options> <end_of_options> <args>

    where

        - `<command>` is the command name or path (used as-is);
        - `<flags>` are space-separated flags with a leading `-` (single
          character flag) or `--` (multiple characters);
        - `<kwflags>` are space-separated key-value flag pairs with a leading
          `-` (single character flag) or `--` (multiple characters), a
          key-value separator (typically `=`) and, if required, a CLI-compliant
          escaped value;
        - `<options>` are space-separated options (used as-is);
        - `<end_of_options>` is a character sequence understood by `<command>`
          as meaning the end of the options (typically `--`);
        - `<args>` are the arguments to the command and could potentially
          themselves be a valid command (_e.g._ POSIX `time`);

    If allowed by the CLI, redirecting the output streams of the command
    (potentially between themselves) may be done through this abstraciton.
    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-few-public-methods
    def __init__(self, command, flags=None, kwflags=None, kwflags_sep=' ',
                 kwflags_join=',', options=None, end_of_options=None,
                 args=None, stdout=None, stderr=None):
        """
        Parameters:
            command    command name or path
            flags      ``str`` or list of ``str`` without the leading `-`/`--`.
                       Flags that evaluate as falsy are ignored;
            kwflags    mapping giving the key-value pairs. The key and value of
                       the pair are separated by a `kwflags_sep`. If the value
                       is a list, it is joined with `kwflags_join`. If a value
                       evaluates as falsy, it is replaced by the empty string;
            kwflags_sep   Key-value separator for `kwflags`;
            kwflags_join  Separator for lists of values in `kwflags`;
            options    same as `flags` but nothing is prepended to the options;
            args       ``str`` or mapping holding keys which are valid
                       arguments to this constructor, for recursive
                       instantiation;
            stdout     file for redirection of ``stdout``. This is passed to
                       the CLI so non-file expressions may be used (*e.g.*
                       `&2`);
            stderr     file for redirection of ``stderr``. This is passed to
                       the CLI so non-file expressions may be used (*e.g.*
                       `&1`);
        """
        # pylint: disable=too-many-arguments
        # pylint: disable=super-init-not-called
        self.command = ' '.join(shlex.split(command))
        self.flags = map(str, filter(None, self._these(flags)))
        self.kwflags_sep = kwflags_sep
        self.kwflags_join = kwflags_join
        self.kwflags = {}
        if kwflags is not None:
            for k in kwflags:
                v = ['' if x is None else str(x)
                     for x in self._these(kwflags[k])]
                self.kwflags[k] = v[0] if len(v) == 1 else v
        self.options = [] if options is None else [
            '' if x is None else str(x) for x in self._these(options)]
        if end_of_options:
            self.options.append(str(end_of_options).strip())
        if isinstance(args, collections.Mapping):
            self.args = Command(**args)
        else:
            self.args = None if args is None else str(args)
        self.stdout = stdout
        self.stderr = stderr

    def __str__(self):
        quoted = itertools.chain(
            shlex.split(self.command),
            map(self._flagged, self.flags),
            ('{}{}{}'.format(self._flagged(k),
                             self.kwflags_sep,
                             self.kwflags_join.join(self._these(v)))
             for k, v in self.kwflags.items()),
            self.options
        )
        words = [shlex.quote(word) for word in quoted]
        if self.args:
            words.append(str(self.args))
        if self.stdout:
            words.append('1>{}'.format(self._filepipe(self.stdout)))
        if self.stderr:
            words.append('2>{}'.format(self._filepipe(self.stderr)))
        return ' '.join(words)

    def __getitem__(self, key):
        return self.__dict__[key]

    @staticmethod
    def _these(x):
        if isinstance(x, str) or not isinstance(x, collections.abc.Iterable):
            return [x]
        return x

    @staticmethod
    def _filepipe(f):
        if isinstance(f, str) and f.startswith('&'):
            return f
        return shlex.quote(f)

    @classmethod
    def _flagged(cls, flag):
        flag = str(flag).strip()
        return '{}{}'.format('--' if len(flag) > 1 else '-', flag)
