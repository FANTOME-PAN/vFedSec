from abc import abstractmethod, ABC
from collections.abc import Sequence
import sys
import time


class IProfiler(ABC):
    @abstractmethod
    def tic(self):
        ...

    @abstractmethod
    def toc(self, is_overhead=True, not_in_test=False):
        ...

    @abstractmethod
    def download(self, obj, original_obj=None, not_in_test=False, offset=0):
        ...

    @abstractmethod
    def upload(self, obj, original_obj=None, not_in_test=False, offset=0):
        ...

    @abstractmethod
    def get_num_download_bytes(self, include_overhead=True, test_phase=False):
        ...

    @abstractmethod
    def get_num_upload_bytes(self, include_overhead=True, test_phase=False):
        ...

    @abstractmethod
    def get_cpu_time(self, include_overhead=True, test_phase=False):
        ...


def get_profiler() -> IProfiler:
    return MySimpleProfiler()


def my_getsizeof(obj: object) -> int:
    s = sys.getsizeof(obj) if obj is not None else 0
    if isinstance(obj, list) or isinstance(obj, tuple):
        for o in obj:
            s += my_getsizeof(o)
    elif isinstance(obj, dict):
        for k in obj:
            s += my_getsizeof(obj[k])
    return s


class MySimpleProfiler(IProfiler):

    def __init__(self):
        self.mark = 0
        self.download_bytes = 0
        self.download_overhead_bytes = 0
        self.upload_bytes = 0
        self.upload_overhead_bytes = 0
        self.total_cpu_time = 0
        self.cpu_time_overhead = 0

        self.td_download_bytes = 0
        self.td_download_overhead_bytes = 0
        self.td_upload_bytes = 0
        self.td_upload_overhead_bytes = 0
        self.td_cpu_time_overhead = 0
        self.td_total_cpu_time = 0

    def tic(self):
        self.mark = time.time()

    def toc(self, is_overhead=True, not_in_test=False):
        t = time.time() - self.mark
        self.total_cpu_time += t
        self.cpu_time_overhead += t if is_overhead else 0
        if not_in_test:
            self.td_total_cpu_time += t
            self.td_cpu_time_overhead += t if is_overhead else 0
        return t

    def download(self, obj, original_obj=None, not_in_test=False, offset=0):
        original_size = my_getsizeof(original_obj) if original_obj is not None else 0
        s = my_getsizeof(obj) + offset
        overhead = s - original_size
        self.download_bytes += s
        self.download_overhead_bytes += overhead
        if not_in_test:
            self.td_download_bytes += s
            self.td_download_overhead_bytes += overhead
        return s

    def upload(self, obj, original_obj=None, not_in_test=False, offset=0):
        original_size = my_getsizeof(original_obj) if original_obj is not None else 0
        s = my_getsizeof(obj) + offset
        if original_size > s:
            raise RuntimeError('what?')
        overhead = s - original_size
        self.upload_bytes += s
        self.upload_overhead_bytes += overhead
        if not_in_test:
            self.td_upload_bytes += s
            self.td_upload_overhead_bytes += overhead
        return s

    def get_cpu_time(self, include_overhead=True, test_phase=False):
        overhead_factor = 0 if include_overhead else 1
        ret = self.total_cpu_time - overhead_factor * self.cpu_time_overhead
        if test_phase:
            ret -= self.td_total_cpu_time - overhead_factor * self.td_cpu_time_overhead
        return ret

    def get_num_download_bytes(self, include_overhead=True, test_phase=False):
        overhead_factor = 0 if include_overhead else 1
        ret = self.download_bytes - overhead_factor * self.download_overhead_bytes
        if test_phase:
            ret -= self.td_download_bytes - overhead_factor * self.td_download_overhead_bytes
        return ret

    def get_num_upload_bytes(self, include_overhead=True, test_phase=False):
        overhead_factor = 0 if include_overhead else 1
        ret = self.upload_bytes - overhead_factor * self.upload_overhead_bytes
        if test_phase:
            ret -= self.td_upload_bytes - overhead_factor * self.td_upload_overhead_bytes
        return ret






