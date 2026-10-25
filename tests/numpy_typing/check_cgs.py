"""Gaussian conversion exposes its NumPy result domain."""

from typing import assert_type

import numpy as np

from unitx import Quantity
from unitx._numerics import Array
from unitx.cgs import gaussian
from unitx.si import si

values = np.array([1.0, 2.0])
assert_type(gaussian.convert(Quantity(values, si.C), gaussian.statC), Quantity[Array])
assert_type(
    gaussian.convert(Quantity(np.float32(1), si.C), gaussian.statC),
    Quantity[Array],
)
