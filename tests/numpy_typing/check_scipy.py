"""Adapter annotations do not require installed SciPy stubs."""

from typing import assert_type

import numpy as np

from unitx import Quantity
from unitx._numerics import Array
from unitx.scipy import simpson
from unitx.si import si

samples = Quantity(np.arange(5.0), si.W)
assert_type(simpson(samples, dx=2 * si.s), Quantity[Array])
assert_type(simpson(samples, x=Quantity(np.arange(5.0), si.s)), Quantity[Array])


def check_bessel() -> None:
    from numpy.typing import NDArray

    from unitx.scipy import jv

    x = np.ones(5)
    orders = np.arange(3)[:, None]
    assert_type(jv(orders, x), Quantity[NDArray[np.float64] | np.float64])
    assert_type(jv(orders, x * si.one), Quantity[NDArray[np.float64] | np.float64])
    assert_type(jv(orders, 1j * (x * si.one)), Quantity[NDArray[np.complex128] | np.complex128])
    # Dtypes outside the documented double-precision cases retain a broad domain.
    assert_type(jv(0, np.ones(3, dtype=np.float32)), Quantity[Array])
