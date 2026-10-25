"""Check array runtime and inferred types using the installed wheel."""

from typing import assert_type, cast

import numpy as np
from numpy.typing import NDArray

from unitx import AffineQuantity, LogQuantity, LogUnit, Quantity, Unit
from unitx._numerics import Array
from unitx.cgs import emu, esu, gaussian, hl
from unitx.exceptions import DimensionError
from unitx.si import si

type ArrayQuantity = Quantity[Array] | AffineQuantity[Array] | LogQuantity[Array]


def check_arrays(values: NDArray[np.float64]) -> None:
    charge = gaussian.convert(Quantity(values, si.C), gaussian.statC)
    assert_type(charge, Quantity[Array])
    np.testing.assert_allclose(gaussian.convert(charge, si.C).value, values)
    for catalogue, charge_unit in ((esu, esu.statC), (emu, emu.abC), (hl, hl.charge)):
        converted = catalogue.convert(Quantity(values, si.C), charge_unit)
        assert_type(converted, Quantity[Array])
        np.testing.assert_allclose(catalogue.convert(converted, si.C).value, values)
    level = LogQuantity(values, si.dBm)
    assert_type(level, LogQuantity[NDArray[np.float64]])
    assert_type(level.unit, LogUnit)
    assert level.value is values
    np.testing.assert_allclose(level.to_quantity(si.W).value, [0.001, 0.1, 1.0])

    voltage_level = LogQuantity(values, si.dB(si.mV))
    assert_type(voltage_level, LogQuantity[NDArray[np.float64]])
    assert voltage_level.value is values
    np.testing.assert_allclose(voltage_level.to_quantity(si.mV).value, 10 ** (values / 20))

    distance = Quantity(values.copy(), si.m)
    assert_type(distance, Quantity[NDArray[np.float64]])
    assert_type(distance.unit, Unit)
    np.testing.assert_allclose(distance.to(si.cm).value, values * 100)
    saved = distance.value.copy()
    try:
        distance[:] = Quantity(values, si.s)
    except DimensionError:
        pass
    else:
        error_message = 'An incompatible assignment must fail.'
        raise AssertionError(error_message)
    np.testing.assert_array_equal(distance.value, saved)

    # NumPy's stubs lose the wrapper on these returns. State the supported
    # array result domain explicitly and verify the wrapper at runtime.
    for sample in (distance, AffineQuantity(values, si.degC), level):
        expanded = cast(ArrayQuantity, np.expand_dims(sample, (0, -1)))
        # moveaxis/rollaxis stubs require ndarray input despite runtime dispatch.
        moved = cast(
            ArrayQuantity,
            expanded.__array_function__(np.moveaxis, (type(expanded),), (expanded, 0, -1), {}),
        )
        swapped = cast(ArrayQuantity, np.swapaxes(moved, 0, 1))
        rolled = cast(
            ArrayQuantity,
            swapped.__array_function__(np.rollaxis, (type(swapped),), (swapped, 2, 0), {}),
        )
        squeezed = np.squeeze(rolled)
        result = cast(ArrayQuantity, np.ravel(squeezed))
        assert type(result) is type(sample) and result.unit == sample.unit
        np.testing.assert_array_equal(result.value, values)
        assert np.shares_memory(result.value, sample.value)

        for actual, expected in (
            (np.atleast_1d(sample), np.atleast_1d(values)),
            (np.atleast_2d(sample), np.atleast_2d(values)),
            (np.atleast_3d(sample), np.atleast_3d(values)),
            (np.broadcast_to(sample, (2, 3)), np.broadcast_to(values, (2, 3))),
            (np.flip(sample), np.flip(values)),
            (np.flipud(sample), np.flipud(values)),
            (np.roll(sample, 1), np.roll(values, 1)),
            (np.repeat(sample, 2), np.repeat(values, 2)),
            (np.tile(sample, (2, 1)), np.tile(values, (2, 1))),
            (np.copy(sample), np.copy(values)),
        ):
            manipulated = cast(ArrayQuantity, actual)
            assert type(manipulated) is type(sample) and manipulated.unit == sample.unit
            np.testing.assert_array_equal(manipulated.value, expected)
        assert_type(np.shape(sample), tuple[int, ...])
        assert_type(np.ndim(sample), int)
        assert_type(np.size(sample), int)
        assert np.shape(sample) == (3,) and np.ndim(sample) == 1 and np.size(sample) == 3
        for operation in (np.matrix_transpose, np.linalg.matrix_transpose, np.fliplr, np.rot90):
            manipulated = cast(ArrayQuantity, operation(np.atleast_2d(sample)))
            assert type(manipulated) is type(sample) and manipulated.unit == sample.unit
            np.testing.assert_array_equal(manipulated.value, operation(np.atleast_2d(values)))
        broadcasted, bare = cast(
            tuple[ArrayQuantity, NDArray[np.float64]],
            np.broadcast_arrays(sample, np.ones((2, 1))),
        )
        assert type(broadcasted) is type(sample) and broadcasted.unit == sample.unit
        np.testing.assert_array_equal(broadcasted.value, np.broadcast_to(values, (2, 3)))
        np.testing.assert_array_equal(bare, np.ones((2, 3)))

    bounded = cast(Quantity[Array], np.clip(distance, Quantity(500, si.cm), Quantity(25, si.m)))
    assert isinstance(bounded, Quantity) and bounded.unit == si.m
    np.testing.assert_allclose(bounded.value, [5, 20, 25])
    cleaned = cast(
        Quantity[Array], np.where(distance > Quantity(0, si.m), bounded, Quantity(0, si.m))
    )
    assert isinstance(cleaned, Quantity) and cleaned.unit == si.m
    np.testing.assert_allclose(cleaned.value, [0, 20, 25])
    selected = cast(Quantity[Array], np.minimum(distance, Quantity(2500, si.cm)))
    assert isinstance(selected, Quantity) and selected.unit == si.m
    np.testing.assert_allclose(selected.value, [0, 20, 25])
    target = Quantity(np.empty(values.shape), si.cm)
    # NumPy's out annotations only admit ndarrays; exercise the supported
    # protocol directly and retain the target's concrete storage type.
    distance.__array_function__(
        np.clip, (Quantity,), (distance, Quantity(5, si.m), Quantity(25, si.m)), {'out': target}
    )
    assert_type(target.value, NDArray[np.float64])
    np.testing.assert_allclose(target.value, [500, 2000, 2500])

    weights = Quantity(np.array([1.0, 2, 1]), Unit('m/cm'))
    average, weight_sum = cast(
        tuple[Quantity[Array], Quantity[Array]],
        np.average(distance, weights=weights, returned=True),
    )
    assert isinstance(average, Quantity) and average.unit == si.m
    assert isinstance(weight_sum, Quantity) and weight_sum.unit == Unit()
    np.testing.assert_allclose(average.value, 17.5)
    np.testing.assert_allclose(weight_sum.value, 400)
    quartiles = cast(Quantity[Array], np.quantile(distance, [0.25, 0.75]))
    percentiles = cast(
        Quantity[Array], np.percentile(distance, Quantity(np.array([0.25, 0.75]), Unit('m/cm')))
    )
    assert quartiles.unit == percentiles.unit == si.m
    np.testing.assert_allclose(quartiles.value, [10, 25])
    np.testing.assert_allclose(percentiles.value, quartiles.value)
    median_target = Quantity(np.empty(()), si.cm)
    distance.__array_function__(
        np.quantile,
        (Quantity,),
        (distance, 0.5),
        {'weights': weights, 'method': 'inverted_cdf', 'out': median_target},
    )
    assert_type(median_target.value.dtype, np.dtype[np.float64])
    np.testing.assert_allclose(median_target.value, 2000)

    gapped = Quantity(np.array([1.0, np.nan, 3]), si.m)
    for operation in (
        np.nanmean,
        np.nanmedian,
        np.nanstd,
        np.nanvar,
        np.nanmin,
        np.nanmax,
        np.nansum,
    ):
        result = cast(Quantity[Array], operation(gapped))
        assert isinstance(result, Quantity)
        assert result.unit == (si.m**2 if operation is np.nanvar else si.m)
        np.testing.assert_allclose(result.value, operation(gapped.value))
    product = cast(Quantity[Array], np.nanprod(gapped))
    assert isinstance(product, Quantity) and product.unit == si.m**2
    np.testing.assert_allclose(product.value, 3)
    for operation in (np.nanargmin, np.nanargmax):
        indices = np.empty((), dtype=np.intp)
        assert operation(gapped, out=indices) is indices
        assert_type(indices.dtype, np.dtype[np.intp])
        np.testing.assert_array_equal(indices, operation(gapped.value))
    for operation, probability in ((np.nanquantile, 0.5), (np.nanpercentile, 50)):
        result = cast(Quantity[Array], operation(gapped, probability))
        assert isinstance(result, Quantity) and result.unit == si.m
        np.testing.assert_allclose(result.value, 2)
        gapped.__array_function__(
            operation, (Quantity,), (gapped, probability), {'out': median_target}
        )
        np.testing.assert_allclose(median_target.value, 200)
    assert_type(median_target.value.dtype, np.dtype[np.float64])

    sample_time = Quantity(np.array([0.0, 1, 3]), si.s)
    sample_current = Quantity(np.array([0.0, 2, 6]), si.A)
    grid = Quantity(np.array([0.0, 500, 1000, 2000, 3000]), si.ms)
    interpolated = cast(Quantity[Array], np.interp(grid, sample_time, sample_current))
    assert isinstance(interpolated, Quantity) and interpolated.unit == si.A
    np.testing.assert_allclose(interpolated.value, [0, 1, 2, 4, 6])
    charge = cast(Quantity[Array], np.trapezoid(interpolated, grid))
    np.testing.assert_allclose(charge.value_in(si.C), 9)


check_arrays(np.asarray([0.0, 20.0, 30.0], dtype=np.float64))
