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

import logging
import devlib.utils.asyn as asyn
from devlib.module import Module
from devlib.utils.misc import ranges_to_list

class Irq(object):
    def __init__(self, target, intid, data_dict, sysfs_root, procfs_root):
        self.target = target
        self.intid = intid
        self.sysfs_path = self.target.path.join(sysfs_root, str(intid))
        self.procfs_path = self.target.path.join(procfs_root, str(intid))

        self.irq_info = self._fix_data_dict(data_dict.copy())

    def _fix_data_dict(self, data_dict):
        clean_dict = data_dict.copy()

        self._fix_sysfs_data(clean_dict)
        self._fix_procfs_data(clean_dict)

        return clean_dict

    def _fix_sysfs_data(self, clean_dict):
        clean_dict['wakeup'] = 0 if clean_dict['wakeup'] == 'disabled' else 1

        if 'hwirq' not in clean_dict:
            clean_dict['hwirq'] = -1
        else:
            clean_dict['hwirq'] = int(clean_dict['hwirq'])

        if 'per_cpu_count' in clean_dict:
            del clean_dict['per_cpu_count']

        if 'name' not in clean_dict:
            clean_dict['name'] = ''

        if 'actions' not in clean_dict:
            clean_dict['actions'] = ''
        else:
            alist = clean_dict['actions'].split(',')
            if alist[0] == '(null)':
                alist = []
            clean_dict['actions'] = alist

    def _fix_procfs_data(self, clean_dict):

        if 'spurious' not in clean_dict:
            clean_dict['spurious'] = ''
        else:
            temp = clean_dict['spurious'].split('\n')
            clean_dict['spurious'] = dict([[i.split(' ')[0], i.split(' ')[1]] for i in temp])

        for alist in ['smp_affinity_list', 'effective_affinity_list']:
            if alist in clean_dict:
                if clean_dict[alist] == '':
                    clean_dict[alist] = []
                    continue
                clean_dict[alist] = ranges_to_list(clean_dict[alist])
 
    @property
    def actions(self):
        return self.irq_info['actions']

    @property
    def chip_name(self):
        return self.irq_info['chip_name']

    @property
    def hwirq(self):
        return self.irq_info['hwirq']

    @property
    def name(self):
        return None if self.irq_info['name'] == '' else self.irq_info['name']

    @property
    def type(self):
        return self.irq_info['type']

    @property
    def wakeup(self):
        return self.irq_info['wakeup']

    @property
    def smp_affinity(self):
        if 'smp_affinity' in self.irq_info.keys():
            return self.irq_info['smp_affinity']
        return -1

    @smp_affinity.setter
    def smp_affinity(self, affinity, verify=True):
        aff = str(affinity)
        aff_path = self.target.path.join(self.procfs_path, 'smp_affinity')
        self.target.write_value(aff_path, aff, verify=verify)

        self.update_affinities()

    @property
    def effective_affinity(self):
        if 'effective_affinity' in self.irq_info.keys():
            return self.irq_info['effective_affinity']
        return -1

    def to_dict(self):
        return self.irq_info.copy()

    @asyn.asyncf
    async def update_affinities(self):
        """Read affinity masks from target."""
        proc_data = await self.target.read_tree_values.asyn(self.procfs_path, depth=2, check_exit_code=False)
        self._fix_procfs_data(proc_data)

        for aff in ['smp_affinity', 'effective_affinity', 'smp_affinity_list', 'effective_affinity_list']:
            self.irq_info[aff] = proc_data[aff]

class IrqModule(Module):
    name = 'irq'
    irq_sysfs_root = '/sys/kernel/irq/'
    irq_procfs_root = '/proc/irq/'

    @staticmethod
    def probe(target):
        if target.file_exists(IrqModule.irq_sysfs_root):
            if target.file_exists(IrqModule.irq_procfs_root):
                return True

    def __init__(self, target):
        self.logger = logging.getLogger(self.name)
        self.logger.debug(f'Initialized {self.name} module')

        self.target = target
        self.irqs = {}

        temp_dict = self._scrape_data(self.target, self.irq_sysfs_root, self.irq_procfs_root)
        for irq, data in temp_dict.items():
            intid = int(irq)
            self.irqs[intid] = Irq(self.target, intid, data, self.irq_sysfs_root, self.irq_procfs_root)

    @asyn.asyncf
    @staticmethod
    async def _scrape_data(cls, target, sysfs_path=None, procfs_path=None):
        if sysfs_path and procfs_path:
            sysfs_dict = await target.read_tree_values.asyn(sysfs_path, depth=2, check_exit_code=False)
            procfs_dict = await target.read_tree_values.asyn(procfs_path, depth=2, check_exit_code=False)

            for irq, data in sysfs_dict.items():
                if irq in procfs_dict.keys():
                    sysfs_dict[irq] = {**data, **procfs_dict[irq]}
            return sysfs_dict

        if sysfs_path:
            sysfs_dict = await target.read_tree_values.asyn(sysfs_path, depth=2, check_exit_code=False)
            return sysfs_dict
        if procfs_path:
            procfs_dict = await target.read_tree_values.asyn(procfs_path, depth=1, check_exit_code=False)
            return procfs_dict

        return None


    def get_all_irqs(self):
        """Returns list of all interrupt IDs (list of integers)."""
        return list(self.irqs.keys())

    def get_all_wakeup_irqs(self):
        """Returns list of all wakeup-enabled interrupt IDs (list of integers)."""
        return [irq.intid for intid, irq in self.irqs.items() if irq.wakeup == 1]

    @asyn.asyncf
    async def get_raw_stats(self):
        """Return raw interrupt stats from procfs on target."""
        raw_stats = await self.target.read_value.asyn('/proc/interrupts')
        return raw_stats

    @asyn.asyncf
    async def get_stats_dict(self):
        """Returns dict of dicts of irq and IPI stats."""
        raw_stats = await self.get_raw_stats.asyn()

        nr_cpus = self.target.number_of_cpus

        d_irq = {
            'intid' : [],
            'chip_name' : [],
            'hwirq' : [],
            'type' : [],
            'actions' : [],
            }

        d_ipi = {
            'id' : [],
            'purpose' : [],
            }

        for cpu in range(0, nr_cpus):
            d_irq[f'cpu{cpu}'] = []
            d_ipi[f'cpu{cpu}'] = []

        for line in self.target.irq.get_raw_stats().splitlines()[1:-1]:
            intid, data = line.split(':', 1)
            data = data.split()

            if 'IPI' in intid:
                d_ipi['id'].append(int(intid[3:]))

                for cpu in range(0, nr_cpus):
                    d_ipi[f'cpu{cpu}'].append(int(data[cpu]))

                d_ipi['purpose'].append(' '.join(data[nr_cpus:]))
            else:
                d_irq['intid'].append(int(intid))
                d_irq['chip_name'].append(data[nr_cpus])

                for cpu in range(0, nr_cpus):
                    d_irq[f'cpu{cpu}'].append(int(data[cpu]))

                if 'Level' in data[nr_cpus+1] or 'Edge' in data[nr_cpus+1]:
                    d_irq['hwirq'].append(None)
                    d_irq['type'].append(data[nr_cpus+1])
                    d_irq['actions'].append(data[nr_cpus+2:])
                else:
                    d_irq['hwirq'].append(int(data[nr_cpus+1]))
                    d_irq['type'].append(data[nr_cpus+2])
                    d_irq['actions'].append(data[nr_cpus+3:])

        return {'irq' : d_irq, 'ipi' : d_ipi}
