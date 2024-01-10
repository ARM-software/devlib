#
#    Copyright 2024 ARM Limited
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
import shutil
import tempfile
from unittest import TestCase

from devlib import LocalLinuxTarget


class TestReadTreeValues(TestCase):

    def test_read_multiline_values(self):
        data = {
            'test1': '1',
            'test2': '2\n\n',
            'test3': '3\n\n4\n\n',
        }

        tempdir = tempfile.mkdtemp(prefix='devlib-test-')
        for key, value in data.items():
            path = os.path.join(tempdir, key)
            with open(path, 'w') as wfh:
                wfh.write(value)

        t = LocalLinuxTarget(connection_settings={'unrooted': True})
        raw_result = t.read_tree_values_flat(tempdir)
        result = {os.path.basename(k): v for k, v in raw_result.items()}

        shutil.rmtree(tempdir)

        self.assertEqual({k: v.strip()
                          for k, v in data.items()},
                         result)
