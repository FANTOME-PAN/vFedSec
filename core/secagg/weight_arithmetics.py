import numpy as np
from typing import List, Tuple


# Weight Manipulation =============================================================

# Combine factor with weights


def factor_weights_combine(weights_factor: int, weights: List[np.ndarray]) -> List[np.ndarray]:
    return [np.array([weights_factor])]+weights

# Extract factor from weights


def factor_weights_extract(weights: List[np.ndarray]) -> Tuple[int, List[np.ndarray]]:
    return weights[0][0], weights[1:]

# Create dimensions list of each element in weights


def weights_shape(weights: List[np.ndarray]) -> List[Tuple]:
    return [arr.shape for arr in weights]

# Generate zero weights based on dimensions list


def weights_zero_generate(dimensions_list: List[Tuple], dtype=np.int64) -> List[np.ndarray]:
    return [np.zeros(dimensions, dtype=dtype) for dimensions in dimensions_list]

# Add two weights together


def weights_addition(a: List[np.ndarray], b: List[np.ndarray]) -> List[np.ndarray]:
    return [a[idx]+b[idx] for idx in range(len(a))]

# Subtract one weight from the other


def weights_subtraction(a: List[np.ndarray], b: List[np.ndarray]) -> List[np.ndarray]:
    return [a[idx]-b[idx] for idx in range(len(a))]

# Take mod of a weights with an integer


def weights_mod(a: List[np.ndarray], b: int) -> List[np.ndarray]:
    if bin(b).count("1") == 1:
        msk = b - 1
        return [a[idx] & msk for idx in range(len(a))]
    return [a[idx] % b for idx in range(len(a))]


# Multiply weight by an integer


def weights_multiply(a: List[np.ndarray], b: int) -> List[np.ndarray]:
    return [a[idx] * b for idx in range(len(a))]

# Divide weight by an integer


def weights_divide(a: List[np.ndarray], b: int) -> List[np.ndarray]:
    return [a[idx] / b for idx in range(len(a))]
