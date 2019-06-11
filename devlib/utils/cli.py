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

class Command(dict):
    """Provides an abstraction for manipulating CLI commands
    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-few-public-methods
    def __init__(self, command, flags=None, kwflags=None, kwflags_sep=' ',
                 kwflags_join=',', options=None, end_of_options=None, args=None,
                 stdout=None, stderr=None):
        """
        NB: if ``None`` in ``flags``, ``kwflags``, ``options``, replace with ``''``
        empty flags are ignored
        empty kwflag values are kept but striped
        NB: caller responsible for escaping args as a single string
        """ #TODO
        # pylint: disable=too-many-arguments
        these = lambda x: (x if isinstance(x, collections.abc.Iterable)
                           and not isinstance(x, str) else [x])

        self.command = shlex.split(command)
        self.flags = map(str, filter(None, these(flags)))
        self.kwflags_sep = kwflags_sep
        self.kwflags_join = kwflags_join
        self.kwflags = {} if kwflags is None else {
            key: ['' if x is None else str(x) for x in these(values)]
            for key, values in kwflags.items()
        }
        self.options = [] if options is None else [
            '' if x is None else str(x) for x in these(options)]
        if end_of_options:
            self.options.append(str(end_of_options).strip())
        if isinstance(args, collections.Mapping):
            self.args = Command(**args)
        else:
            self.args = None if args is None else str(args)
        self.stdout = stdout
        self.stderr = stderr

    def __str__(self):
        filepipe = lambda f: (f if isinstance(f, str) and f.startswith('&')
                              else shlex.quote(f))
        quoted = itertools.chain(
            self.command,
            map(self._flagged, self.flags),
            ('{}{}{}'.format(self._flagged(k),
                             self.kwflags_sep,
                             self.kwflags_join.join(v))
             for k, v in self.kwflags.items()),
            self.options
        )
        words = [shlex.quote(word) for word in quoted]
        if self.args:
            words.append(str(self.args))
        if self.stdout:
            words.append('1>{}'.format(filepipe(self.stdout)))
        if self.stderr:
            words.append('2>{}'.format(filepipe(self.stderr)))
        return ' '.join(words)

    def __getitem__(self, key):
        return self.__dict__[key]

    @classmethod
    def _flagged(cls, flag):
        return '{}{}'.format('--' if len(flag) > 1 else '-', str(flag).strip())
