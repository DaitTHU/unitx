"""Logarithmic arrays preserve concrete family and array result types."""

from typing import Any, assert_type

import numpy as np
import numpy.typing as npt

from unitx import LogQuantity, Quantity
from unitx._numerics import Array
from unitx.si import si

NumpyValue = npt.NDArray[Any] | np.number[Any]


def contracts(value: npt.NDArray[np.float64]) -> None:
    level = LogQuantity(value, si.dBm)
    assert_type(level, LogQuantity[npt.NDArray[np.float64]])
    assert_type(level.to(si.dBW), LogQuantity[NumpyValue])
    assert_type(level.to_quantity(si.W), Quantity[NumpyValue])
    assert_type(level.T, LogQuantity[npt.NDArray[np.float64]])
    assert_type(level[:2], LogQuantity[NumpyValue])
    assert_type(LogQuantity.from_quantity(Quantity(value, si.W), si.dBm), LogQuantity[Array])
