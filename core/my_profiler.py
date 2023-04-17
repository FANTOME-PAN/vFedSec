from abc import abstractmethod, ABC
import sys
import time


class IProfiler(ABC):
    @abstractmethod
    def tic(self):
        ...

    @abstractmethod
    def toc(self, is_overhead=True):
        ...

    @abstractmethod
    def download(self, obj, is_overhead=True):
        ...

    @abstractmethod
    def upload(self, obj, is_overhead=True):
        ...

    @abstractmethod
    def get_num_bytes(self, is_download=True, include_overhead=True):
        ...

    @abstractmethod
    def get_cpu_time(self, include_overhead=True):
        ...


def get_profiler() -> IProfiler:
    return MySimpleProfiler()


class MySimpleProfiler(IProfiler):

    def __init__(self):
        self.mark = 0
        self.download_bytes = 0
        self.download_overhead_bytes = 0
        self.upload_bytes = 0
        self.upload_overhead_bytes = 0
        self.total_cpu_time = 0
        self.cpu_time_overhead = 0

    def tic(self):
        self.mark = time.time()

    def toc(self, is_overhead=True):
        t = time.time() - self.mark
        self.total_cpu_time += t
        self.cpu_time_overhead += t if is_overhead else 0
        return t

    def download(self, obj, is_overhead=True):
        s = sys.getsizeof(obj)
        self.download_bytes += s
        self.download_overhead_bytes += s if is_overhead else 0
        return s

    def upload(self, obj, is_overhead=True):
        s = sys.getsizeof(obj)
        self.upload_bytes += s
        self.upload_overhead_bytes += s if is_overhead else 0
        return s

    def get_num_bytes(self, is_download=True, include_overhead=True):
        if is_download:
            return self.download_bytes if include_overhead else self.download_bytes - self.download_overhead_bytes
        return self.upload_bytes if include_overhead else self.upload_bytes - self.upload_overhead_bytes

    def get_cpu_time(self, include_overhead=True):
        return self.total_cpu_time if include_overhead else self.total_cpu_time - self.cpu_time_overhead






