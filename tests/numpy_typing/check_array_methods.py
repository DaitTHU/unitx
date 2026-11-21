"""Direct methods preserve array wrappers, known shape dtypes and out types."""

from decimal import Decimal
from typing import assert_type

import numpy as np
from numpy.typing import NDArray

from unitx import AffineQuantity, LogQuantity, Quantity, Unit
from unitx.si import si
from unitx.typing import NumpyValue


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
    assert_type(q.std(), Quantity[np.float32])
    out = Quantity(target, si.cm)
    assert_type(q.sum(out=out), Quantity[NDArray[np.float32]])
    assert_type(q.mean(None, None, out), Quantity[NDArray[np.float32]])
    assert_type(q.std(out=out), Quantity[NDArray[np.float32]])
    temperatures = AffineQuantity(target, si.degC)
    levels = LogQuantity(target, si.dBm)
    assert_type(temperatures.reshape(-1), AffineQuantity[NDArray[np.float32]])
    assert_type(temperatures.ravel(), AffineQuantity[NDArray[np.float32]])
    assert_type(temperatures.copy(), AffineQuantity[NDArray[np.float32]])
    assert_type(temperatures.mean(), AffineQuantity[np.float32])
    assert_type(temperatures.mean(out=temperatures), AffineQuantity[NDArray[np.float32]])
    assert_type(temperatures.std(out=out), Quantity[NDArray[np.float32]])
    assert_type(levels.reshape(-1), LogQuantity[NDArray[np.float32]])
    assert_type(levels.ravel(), LogQuantity[NDArray[np.float32]])
    assert_type(levels.copy(), LogQuantity[NDArray[np.float32]])
    assert_type(levels.mean(), LogQuantity[np.float32])
    assert_type(levels.mean(out=levels), LogQuantity[NDArray[np.float32]])
    assert_type(levels.std(), Quantity[np.float32])
    assert_type(Quantity(np.float32(2), si.m).reshape(()), Quantity[NumpyValue])
    assert_type(Quantity(np.float32(2), si.m).ravel(), Quantity[NDArray[np.float32]])
    assert_type(Quantity(np.float32(2), si.m).copy(), Quantity[NDArray[np.float32]])


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
    assert_type(q.min(), Quantity[np.complex64])
    assert_type(q.max(), Quantity[np.complex64])
    assert_type(q.var(), Quantity[np.float32])
    assert_type(q.min(out=out), Quantity[NDArray[np.float32]])
    assert_type(q.max(0, out), Quantity[NDArray[np.float32]])
    assert_type(q.var(out=variance_out), Quantity[NDArray[np.float32]])
    assert_type(q.var(0, None, variance_out), Quantity[NDArray[np.float32]])
    temperatures = AffineQuantity(target, si.degC)
    levels = LogQuantity(target, si.dBm)
    assert_type(temperatures.min(), AffineQuantity[np.float32])
    assert_type(temperatures.max(out=temperatures), AffineQuantity[NDArray[np.float32]])
    assert_type(temperatures.var(), Quantity[np.float32])
    assert_type(temperatures.var(out=variance_out), Quantity[NDArray[np.float32]])
    assert_type(levels.min(0, levels), LogQuantity[NDArray[np.float32]])
    assert_type(levels.max(), LogQuantity[np.float32])
    assert_type(levels.var(), Quantity[np.float32])
    assert_type(levels.var(0, None, variance_out), Quantity[NDArray[np.float32]])


def check_native_workflow() -> None:
    # Ordinary NumPy initialization and ndarray-style methods; no typed adapters.
    samples = np.ones((3, 4)) * si.m
    centered = samples - samples.mean()
    spread = centered.std()
    variance = centered.var()
    lower, upper = samples.min(), samples.max()
    flat = centered.flatten()
    assert_type(centered, Quantity[NDArray[np.float64] | np.float64])
    assert_type(spread, Quantity[np.float64])
    assert_type(variance, Quantity[np.float64])
    assert_type(lower, Quantity[np.float64])
    assert_type(upper, Quantity[np.float64])
    assert_type(flat, Quantity[NDArray[np.float64]])
    assert_type(flat.dtype, np.dtype[np.float64])
    assert_type(np.real(flat), Quantity[NDArray[np.float64]])
    assert_type(np.isrealobj(flat), bool)
    assert_type(spread.value_in(si.cm), np.float64)


def check_metadata(
    values: np.ndarray[tuple[int, int], np.dtype[np.float32]],
) -> None:
    q = values * si.m
    assert_type(q.shape, tuple[int, int])
    assert_type(q.dtype, np.dtype[np.float32])
    point = AffineQuantity(values, si.degC)
    assert_type(point.shape, tuple[int, int])
    assert_type(point.dtype, np.dtype[np.float32])
    assert_type(point.flatten(), AffineQuantity[NDArray[np.float32]])
    level = LogQuantity(values, si.dBm)
    assert_type(level.shape, tuple[int, int])
    assert_type(level.dtype, np.dtype[np.float32])
    assert_type(level.flatten(), LogQuantity[NDArray[np.float32]])


def check_statistical_boundaries(values: NDArray[np.complex128], counts: NDArray[np.int16]) -> None:
    q = values * si.m
    assert_type(q.std(), Quantity[np.float64])
    assert_type(q.std(axis=0), Quantity[NDArray[np.float64] | np.float64])
    assert_type(q.var(keepdims=True), Quantity[NDArray[np.float64] | np.float64])
    assert_type(q.std(mean=np.zeros(1)), Quantity[NumpyValue])
    assert_type(q.var(dtype=np.complex128), Quantity[NumpyValue])
    count = counts * si.m
    assert_type(count.sum(), Quantity[np.intp])
    assert_type(count.sum(axis=0), Quantity[NDArray[np.intp] | np.intp])
    assert_type((np.uint16(1) * si.m).sum(), Quantity[np.uintp])
    assert_type((np.int64(1) * si.m).sum(), Quantity[np.int64])
    assert_type((np.uint64(1) * si.m).sum(), Quantity[np.uint64])
    assert_type(count.min(), Quantity[np.int16])
    assert_type(count.max(axis=0), Quantity[NDArray[np.int16] | np.int16])
    assert_type(count.var(), Quantity[np.float64])
    assert_type(count.std(), Quantity[np.float64])
    zero = np.float32(1) * si.m
    assert_type(zero.flatten(), Quantity[NDArray[np.float32]])
    assert_type(zero.std(), Quantity[np.float32])
    assert_type(zero.min(), Quantity[np.float32])


def check_point_readings(values: NDArray[np.float32], counts: NDArray[np.int16]) -> None:
    readings = AffineQuantity(values, si.degC)
    mask = (0 @ si.degC < readings) & (readings < 100 @ si.degC)
    mask.all()
    selected = readings[mask]
    assert_type(selected, AffineQuantity[NDArray[np.float32]])
    assert_type(next(iter(selected)), AffineQuantity[NDArray[np.float32] | np.float32])
    assert_type(selected.mean().to(si.K), AffineQuantity[np.float32])
    assert_type(selected.mean().to_quantity(si.K), Quantity[np.float32])
    assert_type(selected.mean().value_in(si.K), np.float32)
    assert_type(readings - readings.mean(), Quantity[NDArray[np.float32] | np.float32])
    delta = Quantity(values, si.K)
    assert_type(readings + delta, AffineQuantity[NDArray[np.float32] | np.float32])
    assert_type(delta + readings, AffineQuantity[NDArray[np.float32] | np.float32])
    reference = 0 @ si.degC
    assert_type(reference.isclose(readings), NDArray[np.bool_] | np.bool_)
    assert_type(
        reference.isclose(reference, rel_tol=np.zeros_like(values)), NDArray[np.bool_] | np.bool_
    )
    assert_type(reference.isclose(reference, abs_tol=delta), NDArray[np.bool_] | np.bool_)
    assert_type(readings.isclose(readings), NDArray[np.bool_] | np.bool_)
    scalar = np.float32(1) @ si.K
    assert_type(scalar < (Decimal(2) @ si.K), bool | NDArray[np.bool_] | np.bool_)
    assert_type(AffineQuantity(counts, si.degC).mean(), AffineQuantity[np.float64])
    assert_type(AffineQuantity(counts, si.degC).value_in(si.K), NumpyValue)
    assert_type(readings.mean(dtype=np.float64), AffineQuantity[np.float64])
