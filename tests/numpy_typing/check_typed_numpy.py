"""Wrapped numerical integration, differentiation and allocation keep their types."""

from typing import assert_type

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
