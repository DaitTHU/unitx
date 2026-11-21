"""Concrete dtypes survive supported algebra without claiming a fixed rank."""

# pyright: reportUnnecessaryTypeIgnoreComment=true
from decimal import Decimal
from typing import Any, assert_type

import numpy as np
from numpy.typing import NDArray

from unitx import AffineQuantity, LogQuantity, Quantity
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


def check_dtypes_and_shapes(
    values: np.ndarray[tuple[int, int], np.dtype[np.float32]], scalar: np.float32
) -> None:
    q = Quantity(values, si.m)
    assert_type(q.shape, tuple[int, int])
    assert_type(q.T, Quantity[NDArray[np.float32]])
    assert_type((q + q).shape, tuple[int, ...])
    assert_type(Quantity(scalar, si.m).shape, tuple[()])
    assert_type(q.astype(int), Quantity[NDArray[np.intp]])
    assert_type(Quantity(scalar, si.m).astype(int), Quantity[np.intp])
    assert_type((q + q).astype(int), Quantity[NDArray[np.intp] | np.intp])
    assert_type(q.astype(np.dtype(np.float64)), Quantity[NDArray[np.float64]])
    assert_type(q.astype('f4'), Quantity[NDArray[Any]])
    assert_type(Quantity(scalar, si.m).astype('i4'), Quantity[np.number[Any]])
    assert_type(q.astype(int).mean().to(si.cm), Quantity[np.float64])
    assert_type(q.sum(dtype=np.float64), Quantity[np.float64])
    assert_type(q.mean(axis=1, dtype=np.dtype(np.float32)), Quantity[Float32Value])


def check_mixed_indices(values: NDArray[np.float64], mask: NDArray[np.bool_]) -> None:
    q = Quantity(values, si.m)
    p = AffineQuantity(values, si.degC)
    level = LogQuantity(values, si.dBm)
    assert_type(q[0, :], Quantity[NDArray[np.float64]])
    assert_type(q[0, ...], Quantity[NDArray[np.float64]])
    assert_type(p[0, mask, 0], AffineQuantity[NDArray[np.float64]])
    assert_type(level[0, 0, :], LogQuantity[NDArray[np.float64]])
    assert_type(q[0, 0], Quantity[Float64Value])
    q[0, ...] = 1 * si.m

    scalar = np.float64(1.0)
    q_scalar = Quantity(scalar, si.m)
    p_scalar = AffineQuantity(scalar, si.degC)
    level_scalar = LogQuantity(scalar, si.dBm)
    q_scalar[()] = q_scalar  # pyright: ignore[reportIndexIssue]
    p_scalar[()] = p_scalar  # pyright: ignore[reportIndexIssue]
    level_scalar[()] = level_scalar  # pyright: ignore[reportIndexIssue]
