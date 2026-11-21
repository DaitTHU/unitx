"""Wrapped numerical integration, differentiation and allocation keep their types."""

from decimal import Decimal
from fractions import Fraction
from typing import Any, assert_type

import numpy as np
from numpy.typing import NDArray

from unitx import AffineQuantity, LogQuantity, Quantity, math
from unitx import numpy as unp
from unitx.si import si


def check_calculus(values: NDArray[np.float64]) -> None:
    q = Quantity(values, si.m)
    x = Quantity(values, si.s)
    assert_type(unp.trapezoid(q, x), Quantity[NDArray[np.float64] | np.float64])
    assert_type(unp.gradient(q, axis=0), Quantity[NDArray[np.float64]])
    assert_type(unp.zeros_like(q), Quantity[NDArray[np.float64]])
    assert_type(unp.ones_like(q), Quantity[NDArray[np.float64]])
    assert_type(unp.full_like(q, 2 * si.cm), Quantity[NDArray[np.float64]])
    assert_type(unp.empty_like(Quantity(np.float64(1), si.s)), Quantity[NDArray[np.float64]])
    assert_type(
        unp.zeros_like(AffineQuantity(values, si.degC)), AffineQuantity[NDArray[np.float64]]
    )
    assert_type(unp.ones_like(LogQuantity(values, si.dBm)), LogQuantity[NDArray[np.float64]])
    assert_type(math.sinpi(q / si.m), Quantity[NDArray[np.float64] | np.float64])
    assert_type(math.cospi(q / si.m), Quantity[NDArray[np.float64] | np.float64])


def check_allocations(values: NDArray[np.float32]) -> None:
    q = Quantity(values, si.m)
    assert_type(unp.zeros_like(q, dtype=int), Quantity[NDArray[np.intp]])
    assert_type(unp.ones_like(q, dtype=np.float64), Quantity[NDArray[np.float64]])
    assert_type(unp.empty_like(q, dtype=np.dtype(np.complex64)), Quantity[NDArray[np.complex64]])
    assert_type(unp.full_like(2 * si.m, 30 * si.cm, np.float32), Quantity[NDArray[np.float32]])
    assert_type(unp.ones_like(20 @ si.degC, dtype=float), AffineQuantity[NDArray[np.float64]])
    assert_type(unp.zeros_like(20 @ si.dBm, dtype=np.int32), LogQuantity[NDArray[np.int32]])
    # A Python integer's default native dtype can depend on its value.
    assert_type(unp.zeros_like(2 * si.m), Quantity[NDArray[Any]])


def check_steps(values: NDArray[np.float32], axis: tuple[int, ...]) -> None:
    q = Quantity(values, si.m)
    assert_type(
        unp.trapezoid(q, dx=Quantity(Fraction(1, 2), si.s)),
        Quantity[NDArray[np.float32] | np.float32],
    )
    assert_type(unp.trapezoid(q, dx=Decimal('0.5')), Quantity[NDArray[np.float32] | np.float32])
    assert_type(unp.trapezoid(q, dx=np.float64(0.5)), Quantity[NDArray[np.float64] | np.float64])
    times = AffineQuantity(values, si.s)
    levels = LogQuantity(values, si.dBm)
    assert_type(unp.gradient(q, times, axis=(0,)), Quantity[NDArray[np.float32]])
    assert_type(unp.gradient(levels, times, axis=0), Quantity[NDArray[np.float32]])
    assert_type(unp.gradient(q, axis=(1, 0)), tuple[Quantity[NDArray[np.float32]], ...])
    assert_type(unp.gradient(q, axis=()), tuple[()])
    assert_type(
        unp.gradient(q, axis=axis),
        Quantity[NDArray[np.float32]] | tuple[Quantity[NDArray[np.float32]], ...],
    )
    assert_type(unp.gradient(2 * si.m), tuple[()])
