"""Installed optional SciPy runtime and adapter annotations."""

from importlib import import_module
from typing import assert_type

import numpy as np
from numpy.typing import NDArray

from unitx import Quantity
from unitx.scipy import jv, simpson
from unitx.si import si
from unitx.typing import Array

special = import_module('scipy.special')
# Native SciPy stubs do not describe the Quantity wrapper. Validate this
# declared boundary at runtime; simpson's inferred type is checked below.
value: Quantity[Array] = special.jv(0, 2.0 * si.one)
assert isinstance(value, Quantity) and value.unit == si.one
np.testing.assert_allclose(value.value, special.jv(0, 2.0))
y = Quantity(np.arange(5.0) ** 2, si.W)
x = Quantity(np.arange(5.0), si.s)
area = simpson(y, x=x)
assert_type(area, Quantity[Array])
assert area.unit == si.W * si.s
np.testing.assert_allclose(area.value_in(si.J), 64 / 3)

# A real Bessel sum needs no local Quantity annotation, cast or dtype argument.
arguments = np.linspace(0.1, 1.0, 5) * si.one
orders = np.arange(3)
bessel_values = jv(orders[None, :], arguments[:, None])
overlap = bessel_values @ np.ones(3)
assert_type(overlap, Quantity[NDArray[np.float64] | np.float64])
np.testing.assert_allclose(
    overlap.value, special.jv(orders[None, :], arguments.value[:, None]) @ np.ones(3)
)
