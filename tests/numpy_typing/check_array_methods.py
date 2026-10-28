"""Direct methods preserve array wrappers, known shape dtypes and out types."""

from typing import assert_type

import numpy as np
from numpy.typing import NDArray

from unitx import AffineQuantity, LogQuantity, Quantity, Unit
from unitx.si import si
from unitx.typing import NumpyArray, NumpyValue


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
    assert_type(q.sum(), Quantity[np.complex64])
    assert_type(q.mean(), Quantity[np.complex64])
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


def check_axes(values: NDArray[np.complex64], real: NDArray[np.float32]) -> None:
    q = Quantity(values, si.m)
    assert_type(q.transpose(), Quantity[NDArray[np.complex64]])
    assert_type(q.transpose(2, 0, 1), Quantity[NDArray[np.complex64]])
    assert_type(q.transpose((2, 0, 1)), Quantity[NDArray[np.complex64]])
    assert_type(q.squeeze(axis=(1,)), Quantity[NDArray[np.complex64]])
    assert_type(q.transpose().squeeze().mean(), Quantity[np.complex64])
    assert_type(Quantity(np.float32(2), si.m).transpose(), Quantity[np.float32])
    assert_type(Quantity(np.float32(2), si.m).squeeze(), Quantity[np.float32])
    assert_type(AffineQuantity(real, si.degC).transpose(), AffineQuantity[NDArray[np.float32]])
    assert_type(AffineQuantity(real, si.degC).squeeze(), AffineQuantity[NDArray[np.float32]])
    assert_type(LogQuantity(real, si.dBm).transpose(), LogQuantity[NDArray[np.float32]])
    assert_type(LogQuantity(real, si.dBm).squeeze(), LogQuantity[NDArray[np.float32]])
    assert_type(AffineQuantity(np.float32(2), si.degC).squeeze(), AffineQuantity[np.float32])
    assert_type(LogQuantity(np.float32(2), si.dBm).transpose(), LogQuantity[np.float32])


def check_remaining_reductions(values: NDArray[np.complex64], target: NDArray[np.float32]) -> None:
    q = Quantity(values, si.m)
    out = Quantity(target, si.cm)
    variance_out = Quantity(target, si.cm**2)
    assert_type(q.min(), Quantity[NumpyValue])
    assert_type(q.max(), Quantity[NumpyValue])
    assert_type(q.var(), Quantity[NumpyValue])
    assert_type(q.min(out=out), Quantity[NDArray[np.float32]])
    assert_type(q.max(0, out), Quantity[NDArray[np.float32]])
    assert_type(q.var(out=variance_out), Quantity[NDArray[np.float32]])
    assert_type(q.var(0, None, variance_out), Quantity[NDArray[np.float32]])
    temperatures = AffineQuantity(target, si.degC)
    levels = LogQuantity(target, si.dBm)
    assert_type(temperatures.min(), AffineQuantity[NumpyValue])
    assert_type(temperatures.max(out=temperatures), AffineQuantity[NDArray[np.float32]])
    assert_type(temperatures.var(), Quantity[NumpyValue])
    assert_type(temperatures.var(out=variance_out), Quantity[NDArray[np.float32]])
    assert_type(levels.min(0, levels), LogQuantity[NDArray[np.float32]])
    assert_type(levels.max(), LogQuantity[NumpyValue])
    assert_type(levels.var(), Quantity[NumpyValue])
    assert_type(levels.var(0, None, variance_out), Quantity[NDArray[np.float32]])
