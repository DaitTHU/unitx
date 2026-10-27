"""Concrete dtypes survive supported algebra without claiming a fixed rank."""

from decimal import Decimal
from typing import assert_type

import numpy as np
from numpy.typing import NDArray

from unitx import Quantity
from unitx._numerics.numpy_typing import NumpyValue
from unitx.si import si

type Float32Value = NDArray[np.float32] | np.float32
type Float64Value = NDArray[np.float64] | np.float64
type ComplexValue = NDArray[np.complex128] | np.complex128


def check_precision(
    single: NDArray[np.float32], double: NDArray[np.float64], complex_values: NDArray[np.complex128]
) -> None:
    q = Quantity(single, si.m)
    assert_type(q.to(si.cm), Quantity[Float32Value])
    assert_type(q.value_in(si.cm), Float32Value)
    assert_type(q * q, Quantity[Float32Value])
    assert_type(q / q, Quantity[Float32Value])
    assert_type(q + q, Quantity[Float32Value])
    assert_type(q - q, Quantity[Float32Value])
    assert_type(Decimal('2.5') * q, Quantity[Float32Value])
    assert_type(q / 2, Quantity[Float32Value])
    assert_type(2 / q, Quantity[Float32Value])
    assert_type(q[0], Quantity[Float32Value])
    assert_type(next(iter(q)), Quantity[Float32Value])
    assert_type(q.sum(), Quantity[np.float32])
    assert_type(q.mean(), Quantity[np.float32])
    assert_type(q * double, Quantity[NumpyValue])
    assert_type(q + np.float64(2), Quantity[NumpyValue])
    assert_type(q + Quantity(1, si.cm), Quantity[NumpyValue])
    assert_type(Quantity(np.float64(2), si.m) + q, Quantity[NumpyValue])
    assert_type(q ** np.float64(2), Quantity[NumpyValue])
    # A complete scalar/array formula retains its concrete floating dtype.
    speed = Quantity(2.5, si.m / si.s)
    time = Quantity(double, si.s)
    distance = speed * time
    assert_type(distance, Quantity[Float64Value])
    assert_type((distance / time).to(si.m / si.s), Quantity[Float64Value])
    signal = Quantity(complex_values, si.V)
    assert_type(signal.to(si.mV), Quantity[ComplexValue])
    assert_type(signal * signal, Quantity[ComplexValue])
