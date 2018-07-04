#    Copyright 2015-2017 ARM Limited
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
from glob import iglob
import os
import signal
import shutil
import subprocess
import logging
from getpass import getpass

from devlib.exception import TargetError
from devlib.utils.misc import check_output

PACKAGE_BIN_DIRECTORY = os.path.join(os.path.dirname(__file__), 'bin')

def kill_children(pid, signal=signal.SIGKILL):
    with open('/proc/{0}/task/{0}/children'.format(pid), 'r') as fd:
        for cpid in map(int, fd.read().strip().split()):
            kill_children(cpid, signal)
            os.kill(cpid, signal)

class LocalConnection(object):

    name = 'local'

    def __init__(self, platform=None, keep_password=True, unrooted=False,
                 password=None, timeout=None):
        self.logger = logging.getLogger('local_connection')
        self.keep_password = keep_password
        self.unrooted = unrooted
        self.password = password

    def push(self, source, dest, timeout=None, as_root=False):  # pylint: disable=unused-argument
        self.logger.debug('cp {} {}'.format(source, dest))
        shutil.copy(source, dest)

    def pull(self, source, dest, timeout=None, as_root=False): # pylint: disable=unused-argument
        self.logger.debug('cp {} {}'.format(source, dest))
        if ('*' in source or '?' in source) and os.path.isdir(dest):
            # Pull all files matching a wildcard expression
            for each_source in iglob(source):
                shutil.copy(each_source, dest)
        else:
            shutil.copy(source, dest)

    def execute(self, command, timeout=None, check_exit_code=True,
                as_root=False, strip_colors=True):
        self.logger.debug(command)
        if as_root:
            if self.unrooted:
                raise TargetError('unrooted')
            password = self._get_password()
            command = 'echo \'{}\' | sudo -S '.format(password) + command
        ignore = None if check_exit_code else 'all'
        try:
            return check_output(command, shell=True, timeout=timeout, ignore=ignore)[0]
        except subprocess.CalledProcessError as e:
            message = 'Got exit code {}\nfrom: {}\nOUTPUT: {}'.format(
                e.returncode, command, e.output)
            raise TargetError(message)

    def background(self, command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, as_root=False):
        if as_root:
            if self.unrooted:
                raise TargetError('unrooted')
            password = self._get_password()
            command = 'echo \'{}\' | sudo -S '.format(password) + command
        return subprocess.Popen(command, stdout=stdout, stderr=stderr, shell=True)

    def close(self):
        pass

    def cancel_running_command(self):
        pass

    def _get_password(self):
        if self.password:
            return self.password
        password = getpass('sudo password:')
        if self.keep_password:
            self.password = password
        return password
