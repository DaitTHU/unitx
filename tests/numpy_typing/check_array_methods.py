"""Direct methods preserve array wrappers, known shape dtypes and out types."""

from typing import assert_type

import numpy as np
from numpy.typing import NDArray

from unitx import AffineQuantity, LogQuantity, Quantity, Unit
from unitx._numerics.numpy_typing import NumpyArray, NumpyValue
from unitx.si import si


def check_inferred_shapes() -> None:
    # Newer NumPy stubs infer a one-axis shape here. Reshape and ravel must
    # retain the dtype without promising that the source shape is unchanged.
    q = Quantity(np.arange(6, dtype=np.float32), si.m)
    assert_type(q.reshape(2, 3), Quantity[NDArray[np.float32]])
    assert_type(q.ravel(), Quantity[NDArray[np.float32]])
    assert_type(q.copy(), Quantity[NDArray[np.float32]])


def check_array_methods(values: NDArray[np.complex64], target: NDArray[np.float32]) -> None:
    q = Quantity(values, si.m)
    assert_type(q.reshape(-1, 1), Quantity[NDArray[np.complex64]])
    assert_type(q.reshape((1, -1), copy=True), Quantity[NDArray[np.complex64]])
    assert_type(q.ravel(), Quantity[NDArray[np.complex64]])
    assert_type(q.copy(), Quantity[NDArray[np.complex64]])
    assert_type(q.reshape(-1, 1).mean(axis=0).unit, Unit)
    assert_type(q.sum(), Quantity[NumpyValue])
    assert_type(q.mean(), Quantity[NumpyValue])
    assert_type(q.std(), Quantity[NumpyValue])
    out = Quantity(target, si.cm)
    assert_type(q.sum(out=out), Quantity[NDArray[np.float32]])
    assert_type(q.mean(None, None, out), Quantity[NDArray[np.float32]])
    assert_type(q.std(out=out), Quantity[NDArray[np.float32]])
    temperatures = AffineQuantity(target, si.degC)
    levels = LogQuantity(target, si.dBm)
    assert_type(temperatures.reshape(-1), AffineQuantity[NDArray[np.float32]])
    assert_type(temperatures.ravel(), AffineQuantity[NDArray[np.float32]])
    assert_type(temperatures.copy(), AffineQuantity[NDArray[np.float32]])
    assert_type(temperatures.mean(), AffineQuantity[NumpyValue])
    assert_type(temperatures.mean(out=temperatures), AffineQuantity[NDArray[np.float32]])
    assert_type(temperatures.std(out=out), Quantity[NDArray[np.float32]])
    assert_type(levels.reshape(-1), LogQuantity[NDArray[np.float32]])
    assert_type(levels.ravel(), LogQuantity[NDArray[np.float32]])
    assert_type(levels.copy(), LogQuantity[NDArray[np.float32]])
    assert_type(levels.mean(), LogQuantity[NumpyValue])
    assert_type(levels.mean(out=levels), LogQuantity[NDArray[np.float32]])
    assert_type(levels.std(), Quantity[NumpyValue])
    assert_type(Quantity(np.float32(2), si.m).reshape(()), Quantity[NumpyValue])
    assert_type(Quantity(np.float32(2), si.m).ravel(), Quantity[NumpyArray])
    assert_type(Quantity(np.float32(2), si.m).copy(), Quantity[NumpyArray])
