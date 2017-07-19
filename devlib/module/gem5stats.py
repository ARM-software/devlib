#    Copyright 2017 ARM Limited
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
import os.path
from collections import defaultdict

import devlib
from devlib.module import Module
from devlib.platform import Platform
from devlib.platform.gem5 import Gem5SimulationPlatform
from devlib.utils.gem5 import iter_statistics_dump, GEM5STATS_ROI_NUMBER, GEM5STATS_DUMP_TAIL
from devlib.utils.zip import BufferedZippedTextReader
from devlib.utils.misc import sync


class Gem5ROI:
    def __init__(self, number, target):
        self.target = target
        self.number = number
        self.running = False
        self.field = 'ROI::{}'.format(number)

    def start(self):
        if self.running:
            return False
        self.target.execute('m5 roistart {}'.format(self.number))
        self.running = True
        return True
    
    def stop(self):
        if not self.running:
            return False
        self.target.execute('m5 roiend {}'.format(self.number))
        self.running = False
        return True

class Gem5StatsModule(Module):
    '''
    Module controlling Region of Interest (ROIs) markers, satistics dump 
    frequency and parsing statistics log file when using gem5 platforms.

    ROIs are identified by user-defined labels and need to be booked prior to
    use. The translation of labels into gem5 ROI numbers will be performed
    internally in order to avoid conflicts between multiple clients.
    '''
    name = 'gem5stats'

    @staticmethod
    def probe(target):
        return isinstance(target.platform, Gem5SimulationPlatform)

    def __init__(self, target):
        super(Gem5StatsModule, self).__init__(target)
        self._current_origin = 0
        self.rois = {}

    def book_roi(self, label):
        if label in self.rois:
            raise KeyError('ROI label {} already used'.format(label))
        if len(self.rois) >= GEM5STATS_ROI_NUMBER:
            raise RuntimeError('Too many ROIs reserved')
        all_rois = set(xrange(GEM5STATS_ROI_NUMBER))
        used_rois = set([roi.number for roi in self.rois.values()])
        avail_rois = all_rois - used_rois
        self.rois[label] = Gem5ROI(list(avail_rois)[0], self.target)

    def free_roi(self, label):
        if label not in self.rois:
            raise KeyError('ROI label {} not reserved yet'.format(label))
        self.rois[label].stop()
        del self.rois[label]

    def roi_start(self, label):
        if label not in self.rois:
            raise KeyError('Incorrect ROI label: {}'.format(label))
        if not self.rois[label].start():
            raise TargetError('ROI {} was already running'.format(label))
    
    def roi_end(self, label):
        if label not in self.rois:
            raise KeyError('Incorrect ROI label: {}'.format(label))
        if not self.rois[label].stop():
            raise TargetError('ROI {} was not running'.format(label))

    def start_periodic_dump(self, delay_ns=0, period_ns=10000000):
        # Default period is 10ms because it's roughly what's needed to have
        # accurate power estimations
        if delay_ns < 0 or period_ns < 0:
            msg = 'Delay ({}) and period ({}) for periodic dumps must be positive'
            raise ValueError(msg.format(delay_ns, period_ns))
        self.target.execute('m5 dumpresetstats {} {}'.format(delay_ns, period_ns))
    
    def match(self, keys, rois_labels):
        '''
        Tries to match the list of keys passed as parameter over the statistics
        dumps covered by selected ROIs since origin. Returns a dict indexed by 
        key parameters containing a dict indexed by ROI labels containing an 
        in-order list of records for the key under consideration during the 
        active intervals of the ROI.

        Keys must match fields in gem5's statistics log file. Key example:
            system.cluster0.cores0.power_model.static_power
        '''
        records = defaultdict(lambda : defaultdict(list))
        for record, active_rois in self.match_iter(keys, rois_labels):
            for key in record:
                for roi_label in active_rois:
                    records[key][roi_label].append(record[key])
        return records

    def match_iter(self, keys, rois_labels):
        '''
        Yields for each dump since origin a pair containing:
        1. a dict storing the values corresponding to each of the specified keys
        2. the list of currently active ROIs among those passed as parameters.

        Keys must match fields in gem5's statistics log file. Key example:
            system.cluster0.cores0.power_model.static_power
        '''
        for label in rois_labels:
            if label not in self.rois:
                raise KeyError('Impossible to match ROI label {}'.format(label))
            if self.rois[label].running:
                self.logger.warning('Trying to match records in statistics file'
                        ' while ROI {} is running'.format(label))
        
        def roi_active(roi_label, dump):
            roi = self.rois[roi_label]
            return (roi.field in dump) and (int(dump[roi.field]) == 1)

        with self.get_input_stream() as stats_file:
            stats_file.seek(self._current_origin)
            for dump in iter_statistics_dump(stats_file):
                active_rois = [l for l in rois_labels if roi_active(l, dump)]
                if active_rois:
                    record = {k: dump[k] for k in keys}
                    yield (record, active_rois)
                sync() # be sure to read the last dumps

    def reset_origin(self):
        '''
        Place origin right after the last full dump in the file
        '''
        last_dump_tail = self._current_origin
        # Dump & reset stats to start from a fresh state
        self.target.execute('m5 dumpresetstats')
        with self.get_input_stream() as stats_file:
            for line in stats_file:
                if GEM5STATS_DUMP_TAIL in line:
                    last_dump_tail = stats_file.tell()
        self._current_origin = last_dump_tail

    def get_input_stream(self):
        '''
        Returns a read-only TextIOBase-like object for the statistics file.
        '''
        stats_filename = self.target.platform.stats_filename
        gem5_out_dir = self.target.platform.gem5_out_dir
        stats_path = os.path.join(gem5_out_dir, stats_filename)
        if stats_path.endswith('.gz'):
            return BufferedZippedTextReader(stats_path)
        return open(stats_path, 'r')
