"""LM formulas retain useful dtypes without numerical conversions or casts."""

from typing import assert_type

import numpy as np
from numpy.typing import NDArray

from unitx import Quantity
from unitx._numerics.numpy_typing import NumpyValue
from unitx.si import si

type FloatValue = NDArray[np.float64] | np.float64
type ComplexValue = NDArray[np.complex128] | np.complex128
type SingleComplexValue = NDArray[np.complex64] | np.complex64


def check_selections(values: NDArray[np.complex128], mask: NDArray[np.bool_]) -> None:
    q = Quantity(values, si.V / si.C)
    assert_type(q[:], Quantity[NDArray[np.complex128]])
    assert_type(q[:, 0, 0], Quantity[NDArray[np.complex128]])
    assert_type(q[mask], Quantity[NDArray[np.complex128]])
    assert_type(q[mask, 0], Quantity[NDArray[np.complex128]])
    assert_type(np.real(q[:, 0, 0]), Quantity[NDArray[np.float64]])
    assert_type(q[...], Quantity[NDArray[np.complex128]])
    assert_type(q[None], Quantity[NDArray[np.complex128]])
    # Neither rank nor the rank of an integer-array index is known here.
    assert_type(q[0], Quantity[ComplexValue])
    assert_type(q[()], Quantity[ComplexValue])
    assert_type(q[np.array(0, dtype=np.int64)], Quantity[ComplexValue])
    assert_type(q[0, 0, 0], Quantity[ComplexValue])


def check_phase(single: NDArray[np.float32], double: NDArray[np.float64]) -> None:
    q = Quantity(single, si.m)
    z = Quantity(double, si.m)
    assert_type(q * 1j, Quantity[SingleComplexValue])
    assert_type(1j * q, Quantity[SingleComplexValue])
    assert_type(q / 1j, Quantity[SingleComplexValue])
    assert_type(1j / q, Quantity[SingleComplexValue])
    assert_type(z * 1j, Quantity[ComplexValue])
    assert_type(1j * z, Quantity[ComplexValue])
    assert_type(z / 1j, Quantity[ComplexValue])
    assert_type(1j / z, Quantity[ComplexValue])
    assert_type(z * Quantity(1j, si.s), Quantity[ComplexValue])
    assert_type((z * 1j) * 1j, Quantity[ComplexValue])
    assert_type(Quantity(np.float32(2), si.m) * 1j, Quantity[SingleComplexValue])
    # Strong NumPy scalars must not select the weak Python complex overload.
    assert_type(q * np.complex128(1j), Quantity[NumpyValue])
    assert_type(q * Quantity(np.complex128(1j), si.s), Quantity[NumpyValue])
    assert_type(q * Quantity(double, si.s), Quantity[NumpyValue])


def check_explicit_conversion(values: NDArray[np.float32]) -> None:
    q = Quantity(values, si.cm)
    assert_type(q.astype(float), Quantity[NDArray[np.float64]])
    assert_type(q.astype(complex), Quantity[NDArray[np.complex128]])
    assert_type(q.astype(np.float32), Quantity[NDArray[np.float32]])
    scalar = Quantity(np.float32(2), si.m)
    assert_type(scalar.astype(float), Quantity[np.float64])
    assert_type(scalar.astype(complex), Quantity[np.complex128])
    assert_type(scalar.astype(np.float32), Quantity[np.float32])
    assert_type(Quantity(np.array(2.0), si.m).astype(float), Quantity[NDArray[np.float64]])


def check_energy(values: NDArray[np.float64]) -> None:
    q = Quantity(values, si.J)
    assert_type(q.sum(), Quantity[np.float64])
    assert_type(q.sum().value_in(si.mJ), np.float64)
    assert_type(q.sum().to(si.mJ), Quantity[np.float64])
    assert_type((q.sum() / si.J).n, np.float64)
    assert_type(q.mean(), Quantity[np.float64])
    assert_type(q.sum(axis=None, keepdims=False), Quantity[np.float64])
    assert_type(q.sum(axis=0), Quantity[FloatValue])
    assert_type(q.sum(keepdims=True), Quantity[FloatValue])
    assert_type(q.mean(axis=0), Quantity[FloatValue])
    assert_type(q.mean(keepdims=True), Quantity[FloatValue])


def check_zero_dimensional_exits() -> None:
    # A stored zero-dimensional ndarray may stay an array on identity conversion.
    q = Quantity(np.array(2.0, dtype=np.float64), si.percent)
    assert_type(q.n, FloatValue)
    assert_type(q.value_in(si.one), FloatValue)
    assert_type(q.to(si.one), Quantity[FloatValue])


def check_field_chain(values: NDArray[np.float64]) -> None:
    from unitx import math

    # Ordinary NumPy defaults supply the real dtype; no dtype arguments/casts.
    z = values * si.m
    zR = 0.5 * si.m
    k = 2.0 / si.m
    ratio = z / (z + 1j * zR)
    assert_type(ratio, Quantity[ComplexValue])
    gouy = math.atan(z / zR)
    geometry = math.hypot(1, z / zR)
    assert_type(gouy, Quantity[FloatValue])
    assert_type(geometry, Quantity[FloatValue])
    phase = k * (z - z * ratio / 2)
    conjugate_field = math.conj(math.exp(1j * (phase + gouy)) / geometry)
    assert_type(conjugate_field, Quantity[ComplexValue])
    vertical_phase = math.exp(-1j * gouy * np.arange(3)[:, None])
    assert_type(vertical_phase, Quantity[ComplexValue])
    for _ in range(3):
        conjugate_field *= math.exp(-1j * gouy)
        assert_type(conjugate_field, Quantity[ComplexValue])
    assert_type(abs(conjugate_field), Quantity[FloatValue])
    assert_type(math.sqrt(z / zR), Quantity[FloatValue])
