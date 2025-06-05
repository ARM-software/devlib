#    Copyright 2018-2025 ARM Limited
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

from devlib.module import Module
from devlib.module.hotplug import HotplugModule
from devlib.module.cpufreq import CpufreqModule
from typing import (TYPE_CHECKING, cast, List,
                    Optional, Dict)
if TYPE_CHECKING:
    from devlib.target import Target


class BigLittleModule(Module):

    name = 'bl'

    @staticmethod
    def probe(target: 'Target') -> bool:
        return target.big_core is not None

    @property
    def bigs(self) -> List[int]:
        """
        get the list of big cores
        """
        return [i for i, c in enumerate(self.target.platform.core_names)
                if c == self.target.platform.big_core]

    @property
    def littles(self) -> List[int]:
        """
        get the list of little cores
        """
        return [i for i, c in enumerate(self.target.platform.core_names)
                if c == self.target.platform.little_core]

    @property
    def bigs_online(self) -> List[int]:
        """
        get the list of big cores which are online
        """
        return list(sorted(set(self.bigs).intersection(self.target.list_online_cpus())))

    @property
    def littles_online(self) -> List[int]:
        """
        get the list of little cores which are online
        """
        return list(sorted(set(self.littles).intersection(self.target.list_online_cpus())))

    # hotplug

    def online_all_bigs(self) -> None:
        """
        make all big cores go online
        """
        cast(HotplugModule, self.target.hotplug).online(*self.bigs)

    def offline_all_bigs(self) -> None:
        """
        make all big cores go offline
        """
        cast(HotplugModule, self.target.hotplug).offline(*self.bigs)

    def online_all_littles(self) -> None:
        """
        make all little cores go online
        """
        cast(HotplugModule, self.target.hotplug).online(*self.littles)

    def offline_all_littles(self) -> None:
        """
        make all little cores go offline
        """
        cast(HotplugModule, self.target.hotplug).offline(*self.littles)

    # cpufreq

    def list_bigs_frequencies(self) -> Optional[List[int]]:
        """
        get the big cores frequencies
        """
        bigs_online = self.bigs_online
        if bigs_online:
            return cast(CpufreqModule, self.target.cpufreq).list_frequencies(bigs_online[0])
        return None

    def list_bigs_governors(self) -> Optional[List[str]]:
        """
        get the governors supported for the first big core that is online
        """
        bigs_online = self.bigs_online
        if bigs_online:
            return cast(CpufreqModule, self.target.cpufreq).list_governors(bigs_online[0])
        return None

    def list_bigs_governor_tunables(self) -> Optional[List[str]]:
        """
        get the tunable governors supported for the first big core that is online
        """
        bigs_online = self.bigs_online
        if bigs_online:
            return cast(CpufreqModule, self.target.cpufreq).list_governor_tunables(bigs_online[0])
        return None

    def list_littles_frequencies(self) -> Optional[List[int]]:
        """
        get the little cores frequencies
        """
        littles_online = self.littles_online
        if littles_online:
            return cast(CpufreqModule, self.target.cpufreq).list_frequencies(littles_online[0])
        return None

    def list_littles_governors(self) -> Optional[List[str]]:
        """
        get the governors supported for the first little core that is online
        """
        littles_online = self.littles_online
        if littles_online:
            return cast(CpufreqModule, self.target.cpufreq).list_governors(littles_online[0])
        return None

    def list_littles_governor_tunables(self) -> Optional[List[str]]:
        """
        get the tunable governors supported for the first little core that is online
        """
        littles_online = self.littles_online
        if littles_online:
            return cast(CpufreqModule, self.target.cpufreq).list_governor_tunables(littles_online[0])
        return None

    def get_bigs_governor(self) -> Optional[str]:
        """
        get the current governor set for the first big core that is online
        """
        bigs_online = self.bigs_online
        if bigs_online:
            return cast(CpufreqModule, self.target.cpufreq).get_governor(bigs_online[0])
        return None

    def get_bigs_governor_tunables(self) -> Optional[Dict[str, str]]:
        """
        get the current governor tunables set for the first big core that is online
        """
        bigs_online = self.bigs_online
        if bigs_online:
            return cast(CpufreqModule, self.target.cpufreq).get_governor_tunables(bigs_online[0])
        return None

    def get_bigs_frequency(self) -> Optional[int]:
        """
        get the current frequency that is set for the first big core that is online
        """
        bigs_online = self.bigs_online
        if bigs_online:
            return cast(CpufreqModule, self.target.cpufreq).get_frequency(bigs_online[0])
        return None

    def get_bigs_min_frequency(self) -> Optional[int]:
        """
        get the current minimum frequency that is set for the first big core that is online
        """
        bigs_online = self.bigs_online
        if bigs_online:
            return cast(CpufreqModule, self.target.cpufreq).get_min_frequency(bigs_online[0])
        return None

    def get_bigs_max_frequency(self) -> Optional[int]:
        """
        get the current maximum frequency that is set for the first big core that is online
        """
        bigs_online = self.bigs_online
        if bigs_online:
            return cast(CpufreqModule, self.target.cpufreq).get_max_frequency(bigs_online[0])
        return None

    def get_littles_governor(self) -> Optional[str]:
        """
        get the current governor set for the first little core that is online
        """
        littles_online = self.littles_online
        if littles_online:
            return cast(CpufreqModule, self.target.cpufreq).get_governor(littles_online[0])
        return None

    def get_littles_governor_tunables(self) -> Optional[Dict[str, str]]:
        """
        get the current governor tunables set for the first little core that is online
        """
        littles_online = self.littles_online
        if littles_online:
            return cast(CpufreqModule, self.target.cpufreq).get_governor_tunables(littles_online[0])
        return None

    def get_littles_frequency(self) -> Optional[int]:
        """
        get the current frequency that is set for the first little core that is online
        """
        littles_online = self.littles_online
        if littles_online:
            return cast(CpufreqModule, self.target.cpufreq).get_frequency(littles_online[0])
        return None

    def get_littles_min_frequency(self) -> Optional[int]:
        """
        get the current minimum frequency that is set for the first little core that is online
        """
        littles_online = self.littles_online
        if littles_online:
            return cast(CpufreqModule, self.target.cpufreq).get_min_frequency(littles_online[0])
        return None

    def get_littles_max_frequency(self) -> Optional[int]:
        """
        get the current maximum frequency that is set for the first little core that is online
        """
        littles_online = self.littles_online
        if littles_online:
            return cast(CpufreqModule, self.target.cpufreq).get_max_frequency(littles_online[0])
        return None

    def set_bigs_governor(self, governor: str, **kwargs) -> None:
        """
        set governor for the first online big core
        """
        bigs_online = self.bigs_online
        if bigs_online:
            cast(CpufreqModule, self.target.cpufreq).set_governor(bigs_online[0], governor, **kwargs)
        else:
            raise ValueError("All bigs appear to be offline")

    def set_bigs_governor_tunables(self, governor: str, **kwargs) -> None:
        """
        set governor tunables for the first big core that is online
        """
        bigs_online = self.bigs_online
        if bigs_online:
            cast(CpufreqModule, self.target.cpufreq).set_governor_tunables(bigs_online[0], governor, **kwargs)
        else:
            raise ValueError("All bigs appear to be offline")

    def set_bigs_frequency(self, frequency: int, exact: bool = True) -> None:
        """
        set the frequency for the first big core that is online
        """
        bigs_online = self.bigs_online
        if bigs_online:
            cast(CpufreqModule, self.target.cpufreq).set_frequency(bigs_online[0], frequency, exact)
        else:
            raise ValueError("All bigs appear to be offline")

    def set_bigs_min_frequency(self, frequency: int, exact: bool = True) -> None:
        """
        set the minimum value for the cpu frequency for the first big core that is online
        """
        bigs_online = self.bigs_online
        if bigs_online:
            cast(CpufreqModule, self.target.cpufreq).set_min_frequency(bigs_online[0], frequency, exact)
        else:
            raise ValueError("All bigs appear to be offline")

    def set_bigs_max_frequency(self, frequency: int, exact: bool = True) -> None:
        """
        set the minimum value for the cpu frequency for the first big core that is online
        """
        bigs_online = self.bigs_online
        if bigs_online:
            cast(CpufreqModule, self.target.cpufreq).set_max_frequency(bigs_online[0], frequency, exact)
        else:
            raise ValueError("All bigs appear to be offline")

    def set_littles_governor(self, governor: str, **kwargs) -> None:
        """
        set governor for the first online little core
        """
        littles_online = self.littles_online
        if littles_online:
            cast(CpufreqModule, self.target.cpufreq).set_governor(littles_online[0], governor, **kwargs)
        else:
            raise ValueError("All littles appear to be offline")

    def set_littles_governor_tunables(self, governor: str, **kwargs) -> None:
        """
        set governor tunables for the first little core that is online
        """
        littles_online = self.littles_online
        if littles_online:
            cast(CpufreqModule, self.target.cpufreq).set_governor_tunables(littles_online[0], governor, **kwargs)
        else:
            raise ValueError("All littles appear to be offline")

    def set_littles_frequency(self, frequency: int, exact: bool = True) -> None:
        """
        set the frequency for the first little core that is online
        """
        littles_online = self.littles_online
        if littles_online:
            cast(CpufreqModule, self.target.cpufreq).set_frequency(littles_online[0], frequency, exact)
        else:
            raise ValueError("All littles appear to be offline")

    def set_littles_min_frequency(self, frequency: int, exact: bool = True) -> None:
        """
        set the minimum value for the cpu frequency for the first little core that is online
        """
        littles_online = self.littles_online
        if littles_online:
            cast(CpufreqModule, self.target.cpufreq).set_min_frequency(littles_online[0], frequency, exact)
        else:
            raise ValueError("All littles appear to be offline")

    def set_littles_max_frequency(self, frequency: int, exact: bool = True) -> None:
        """
        set the maximum value for the cpu frequency for the first little core that is online
        """
        littles_online = self.littles_online
        if littles_online:
            cast(CpufreqModule, self.target.cpufreq).set_max_frequency(littles_online[0], frequency, exact)
        else:
            raise ValueError("All littles appear to be offline")
