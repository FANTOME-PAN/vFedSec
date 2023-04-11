import pickle
from typing import List

import numpy as np
from flwr.common import bytes_to_ndarray, ndarray_to_bytes, Parameters, ndarrays_to_parameters


def ndarrays_to_bytes(ndarrays: List[np.ndarray]) -> bytes:
    return pyobj2bytes([ndarray_to_bytes(arr) for arr in ndarrays])


def ndarrays_to_param_bytes(ndarrays: List[np.ndarray]) -> bytes:
    return pyobj2bytes(ndarrays_to_parameters(ndarrays))


def bytes_to_ndarrays(long_bytes: bytes) -> List[np.ndarray]:
    bytes_lst = bytes2pyobj(long_bytes)
    return [bytes_to_ndarray(b) for b in bytes_lst]


def pyobj2bytes(obj: object) -> bytes:
    return pickle.dumps(obj)


def bytes2pyobj(b: bytes) -> object:
    return pickle.loads(b)


def empty_parameters() -> Parameters:
    """Utility function that generates empty Flower Parameters dataclass instance."""
    return ndarrays_to_parameters([])
