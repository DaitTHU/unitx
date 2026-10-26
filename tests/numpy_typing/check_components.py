"""NumPy component protocols preserve the Quantity wrapper and native dtype."""

from typing import assert_type

import numpy as np
from numpy.typing import NDArray

from unitx import Quantity


def components(
    doubles: Quantity[NDArray[np.complex128]],
    singles: Quantity[NDArray[np.complex64]],
    real: Quantity[NDArray[np.float64]],
    scalar: Quantity[np.complex128],
) -> None:
    assert_type(doubles.real, Quantity[NDArray[np.float64]])
    assert_type(doubles.imag, Quantity[NDArray[np.float64]])
    assert_type(np.real(doubles), Quantity[NDArray[np.float64]])
    assert_type(np.imag(doubles), Quantity[NDArray[np.float64]])
    assert_type(np.real(singles), Quantity[NDArray[np.float32]])
    assert_type(np.imag(singles), Quantity[NDArray[np.float32]])
    assert_type(np.real(real), Quantity[NDArray[np.float64]])
    assert_type(np.imag(real), Quantity[NDArray[np.float64]])
    assert_type(np.real(scalar), Quantity[np.float64])
    assert_type(np.imag(scalar), Quantity[np.float64])
