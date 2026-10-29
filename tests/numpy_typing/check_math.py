from typing import assert_type

import numpy as np
from numpy.typing import NDArray

from unitx import Quantity, math
from unitx._numerics import Array


def check_math(values: NDArray[np.float64]) -> None:
    q = Quantity(values, 'm2')
    assert_type(math.sqrt(q), Quantity[NDArray[np.float64] | np.float64])
    assert_type(math.sin(values), Quantity[NDArray[np.float64] | np.float64])
    assert_type(
        math.asin(Quantity(np.float64(0.5), '')), Quantity[NDArray[np.float64] | np.float64]
    )
    assert_type(math.hypot(q, q), Quantity[NDArray[np.float64] | np.float64])
    assert_type(math.atan2(q, q), Quantity[Array])
    assert_type(math.atan2(Quantity(1, 'm2'), q), Quantity[Array])


def check_math_precision(
    single: NDArray[np.float32], complex_single: NDArray[np.complex64]
) -> None:
    assert_type(math.sin(single), Quantity[NDArray[np.float32] | np.float32])
    assert_type(math.cos(single), Quantity[NDArray[np.float32] | np.float32])
    assert_type(math.atan(single), Quantity[NDArray[np.float32] | np.float32])
    assert_type(math.sqrt(single), Quantity[NDArray[np.float32] | np.float32])
    assert_type(math.exp(complex_single), Quantity[NDArray[np.complex64] | np.complex64])
    assert_type(math.conj(complex_single), Quantity[NDArray[np.complex64] | np.complex64])
    real = Quantity(single, 'm')
    complex_q = Quantity(complex_single, 'm')
    assert_type(real + complex_q, Quantity[NDArray[np.complex64] | np.complex64])
    assert_type(real - complex_q, Quantity[NDArray[np.complex64] | np.complex64])
    assert_type(real * complex_q, Quantity[NDArray[np.complex64] | np.complex64])
    assert_type(real / complex_q, Quantity[NDArray[np.complex64] | np.complex64])
    assert_type(abs(complex_q), Quantity[NDArray[np.float32] | np.float32])
