"""Array exits and representation changes retain truthful magnitude domains."""

from typing import assert_type

import numpy as np
from numpy.typing import NDArray

from unitx import AffineQuantity, AffineUnit, LogQuantity, LogUnit, Quantity, Unit
from unitx._numerics.numpy_typing import NumpyValue
from unitx.si import si


def check_representation(values: NDArray[np.float64]) -> None:
    quantity = Quantity(values, si.percent)
    assert_type(quantity.n, NumpyValue)
    assert_type(quantity.to_base(), Quantity[NumpyValue])
    assert_type(quantity.simplify(), Quantity[NumpyValue])
    assert_type(quantity.compact(), Quantity[NumpyValue])
    assert_type(quantity.as_pair(), tuple[NDArray[np.float64], Unit])
    assert_type(AffineQuantity(values, si.degC).as_pair(), tuple[NDArray[np.float64], AffineUnit])
    assert_type(LogQuantity(values, si.dBm).as_pair(), tuple[NDArray[np.float64], LogUnit])
