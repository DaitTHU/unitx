"""Gaussian conversion exposes its NumPy result domain."""

from typing import assert_type

import numpy as np

from unitx import Quantity
from unitx._numerics import Array
from unitx.cgs import emu, esu, gaussian, hl
from unitx.si import si

values = np.array([1.0, 2.0])
assert_type(gaussian.convert(Quantity(values, si.C), gaussian.statC), Quantity[Array])
assert_type(
    gaussian.convert(Quantity(np.float32(1), si.C), gaussian.statC),
    Quantity[Array],
)
assert_type(esu.convert(Quantity(values, si.T), esu.magnetic_flux_density), Quantity[Array])
assert_type(emu.convert(Quantity(values, si.C), emu.abC), Quantity[Array])
assert_type(hl.convert(Quantity(values, si.C), hl.charge), Quantity[Array])
