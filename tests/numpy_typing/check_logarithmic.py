"""Logarithmic arrays preserve concrete family and array result types."""

from typing import assert_type

import numpy as np
import numpy.typing as npt

from unitx import LogQuantity, Quantity
from unitx.si import si
from unitx.typing import Array

type FloatValue = npt.NDArray[np.float64] | np.float64


def contracts(value: npt.NDArray[np.float64]) -> None:
    level = LogQuantity(value, si.dBm)
    assert_type(level, LogQuantity[npt.NDArray[np.float64]])
    assert_type(level.to(si.dBW), LogQuantity[FloatValue])
    assert_type(level.to_quantity(si.W), Quantity[FloatValue])
    assert_type(level.T, LogQuantity[npt.NDArray[np.float64]])
    assert_type(level[:2], LogQuantity[npt.NDArray[np.float64]])
    assert_type(LogQuantity.from_quantity(Quantity(value, si.W), si.dBm), LogQuantity[Array])

    assert_type(level - level.mean(), Quantity[FloatValue])
    delta = Quantity(value, si.dB)
    assert_type(level + delta, LogQuantity[FloatValue])
    assert_type(delta + level, LogQuantity[FloatValue])
    reference = 0 @ si.dBm
    assert_type(reference.isclose(level), npt.NDArray[np.bool_] | np.bool_)
    assert_type(
        reference.isclose(reference, rel_tol=np.zeros_like(value)), npt.NDArray[np.bool_] | np.bool_
    )
    assert_type(reference.isclose(reference, abs_tol=delta), npt.NDArray[np.bool_] | np.bool_)
    selected = level[(level > 0 @ si.dBm) & (level < 30 @ si.dBm)]
    assert_type(selected.mean().value_in(si.dBW), np.float64)
    assert_type(selected.mean().to_quantity(si.W), Quantity[np.float64])
