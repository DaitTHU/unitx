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
