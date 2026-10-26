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


def check_components(signal: Quantity[NDArray[np.complex128]]) -> None:
    assert_type(signal.real, Quantity[NDArray[np.float64]])
    assert_type(signal.imag, Quantity[NDArray[np.float64]])
    real = np.real(signal)
    imag = np.imag(signal)
    assert_type(real, Quantity[NDArray[np.float64]])
    assert_type(imag, Quantity[NDArray[np.float64]])
    assert real.unit == imag.unit == signal.unit
    np.testing.assert_array_equal(real.value, signal.value.real)
    np.testing.assert_array_equal(imag.value, signal.value.imag)


def check_arrays(values: NDArray[np.float64]) -> None:
    signal = Quantity(np.asarray([3 + 4j], dtype=np.complex128), si.V)
    check_components(signal)
    norm = cast(Quantity[Array], np.linalg.norm(signal))
    assert norm.unit == si.V
    np.testing.assert_allclose(norm.value, 5)
    count = cast(Quantity[Array], np.linalg.norm(signal, ord=0))
    assert count.unit == si.one
    np.testing.assert_allclose(count.value, 1)
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

    phase = Quantity(np.array([170.0, -170.0, -160.0]), si.deg)
    unwrapped = cast(Quantity[Array], np.unwrap(phase))
    assert unwrapped.unit == si.deg
    np.testing.assert_allclose(unwrapped.value, [170, 190, 200])
    ones = cast(Quantity[Array], np.ones_like(phase))
    assert ones.unit == si.deg
    np.testing.assert_array_equal(ones.value, [1, 1, 1])
    normalized = cast(Quantity[Array], np.sinh(Quantity(values, si.one)))
    assert normalized.unit == si.one
    np.testing.assert_allclose(normalized.value, np.sinh(values))
    force = Quantity(np.eye(3), si.N)
    displacement = Quantity(values, si.m)
    work = cast(Quantity[Array], np.tensordot(force, displacement, axes=1))
    assert work.unit == si.N * si.m
    np.testing.assert_array_equal(work.value, values)
    # NumPy 2.1's native einsum overloads contain unknown dtype parameters.
    contracted = cast(
        Quantity[Array],
        np.einsum('ij,j->i', force, displacement),  # pyright: ignore[reportUnknownMemberType]
    )
    np.testing.assert_array_equal(contracted.value, values)
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

        for actual, expected in (
            (np.hstack((sample, sample)), np.hstack((values, values))),
            (np.vstack((sample, sample)), np.vstack((values, values))),
            (np.dstack((sample, sample)), np.dstack((values, values))),
            (np.column_stack((sample, sample)), np.column_stack((values, values))),
            (np.append(sample, sample), np.append(values, values)),
            (np.insert(sample, 1, sample), np.insert(values, 1, values)),
            (np.take(sample, [2, 0]), np.take(values, [2, 0])),
            (
                sample.__array_function__(
                    np.take_along_axis, (type(sample),), (sample, np.array([2, 0]), 0), {}
                ),
                np.take_along_axis(values, np.array([2, 0]), axis=0),
            ),
            (np.compress([True, False, True], sample), values[[0, 2]]),
            (np.extract([True, False, True], sample), values[[0, 2]]),
            (np.delete(sample, 1), np.delete(values, 1)),
            (np.diag(sample), np.diag(values)),
            (np.diagflat(sample, k=1), np.diagflat(values, k=1)),
            (np.diagonal(np.atleast_2d(sample)), np.diagonal(np.atleast_2d(values))),
            (np.tril(sample), np.tril(values)),
            (np.triu(sample), np.triu(values)),
        ):
            selected = cast(ArrayQuantity, actual)
            assert type(selected) is type(sample) and selected.unit == sample.unit
            np.testing.assert_array_equal(selected.value, expected)
        for actual, expected in (
            (np.split(sample, 3), np.split(values, 3)),
            (np.array_split(sample, 2), np.array_split(values, 2)),
            (np.hsplit(sample, 3), np.hsplit(values, 3)),
            (np.vsplit(np.atleast_2d(sample), 1), np.vsplit(np.atleast_2d(values), 1)),
            (np.dsplit(np.atleast_3d(sample), 1), np.dsplit(np.atleast_3d(values), 1)),
        ):
            for selected, raw in zip(cast(list[ArrayQuantity], actual), expected):
                assert type(selected) is type(sample) and selected.unit == sample.unit
                np.testing.assert_array_equal(selected.value, raw)
        selection_target = cast(ArrayQuantity, np.take(sample, [0, 1]))
        sample.__array_function__(
            np.take, (type(sample),), (sample, [2, 0]), {'out': selection_target}
        )
        np.testing.assert_array_equal(selection_target.value, values[[2, 0]])
        sample.__array_function__(
            np.compress,
            (type(sample),),
            ([True, False, True], sample),
            {'out': selection_target},
        )
        np.testing.assert_array_equal(selection_target.value, values[[0, 2]])

    bounded = cast(Quantity[Array], np.clip(distance, Quantity(500, si.cm), Quantity(25, si.m)))
    sorted_distance = cast(Quantity[Array], np.sort(distance))
    np.testing.assert_array_equal(sorted_distance.value, np.sort(values))
    np.testing.assert_array_equal(np.argsort(distance), np.argsort(values))
    np.testing.assert_array_equal(np.searchsorted(sorted_distance, Quantity(2000, si.cm)), 1)
    unique_values, counts = np.unique(distance, return_counts=True)
    unique_distance = cast(Quantity[Array], unique_values)
    assert unique_distance.unit == distance.unit
    np.testing.assert_array_equal(unique_distance.value, values)
    np.testing.assert_array_equal(counts, [1, 1, 1])
    np.testing.assert_array_equal(
        np.isin(distance, Quantity(np.array([2000.0, 3000]), si.cm)), [False, True, True]
    )
    empty = cast(Quantity[Array], np.empty_like(distance))
    zero = cast(Quantity[Array], np.zeros_like(distance))
    assert empty.unit == zero.unit == distance.unit
    np.testing.assert_array_equal(zero.value, np.zeros_like(values))
    complex_distance = distance.astype(complex)
    real = cast(Quantity[Array], np.real(complex_distance))
    imag = cast(Quantity[Array], np.imag(complex_distance))
    assert real.unit == imag.unit == distance.unit
    np.testing.assert_array_equal(real.value, values)
    np.testing.assert_array_equal(imag.value, np.zeros_like(values))

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
    # NumPy versions differ in the shape inferred for empty(values.shape).
    assert isinstance(target.value, np.ndarray)
    assert_type(target.value.dtype, np.dtype[np.float64])
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


def check_linalg() -> None:
    a = Quantity(np.array([[3.0, 1.0], [1.0, 2.0]]), si.ohm)
    b = Quantity(np.array([9.0, 8.0]), si.V)
    x = cast(Quantity[Array], np.linalg.solve(a, b))
    np.testing.assert_allclose(x.value_in(si.A), [2, 3])
    inverse = cast(Quantity[Array], np.linalg.inv(a))
    pseudo = cast(Quantity[Array], np.linalg.pinv(a, rcond=1e-12))
    np.testing.assert_allclose((a @ inverse).value_in(si.one), np.eye(2), atol=1e-14)
    np.testing.assert_allclose(pseudo.value, inverse.value)
    squared = cast(Quantity[Array], np.linalg.matrix_power(a, 2))
    determinant = cast(Quantity[Array], np.linalg.det(a))
    assert squared.unit == determinant.unit == a.unit**2
    solution, residuals, rank, singular = cast(
        tuple[Quantity[Array], Quantity[Array], object, Quantity[Array]], np.linalg.lstsq(a, b)
    )
    assert solution.unit == b.unit / a.unit and residuals.unit == b.unit**2
    assert rank == 2 and singular.unit == a.unit
    for operation in (np.linalg.vector_norm, np.linalg.matrix_norm):
        norm = cast(Quantity[Array], operation(a))
        assert norm.unit == a.unit
    condition = cast(Quantity[Array], np.linalg.cond(a))
    assert condition.unit == si.one and np.linalg.matrix_rank(a) == 2
    tensor = Quantity(np.eye(4).reshape(2, 2, 2, 2), si.ohm)
    tensor_inverse = cast(Quantity[Array], np.linalg.tensorinv(tensor, ind=2))
    result = cast(Quantity[Array], np.linalg.tensorsolve(tensor, Quantity(np.ones((2, 2)), si.V)))
    assert tensor_inverse.unit == tensor.unit**-1 and result.unit == si.V / si.ohm


def check_factorizations() -> None:
    a = Quantity(np.array([[3.0, 1.0], [1.0, 2.0]]), si.ohm)
    q, r = cast(tuple[Quantity[Array], Quantity[Array]], np.linalg.qr(a))
    assert q.unit == si.one and r.unit == a.unit
    np.testing.assert_allclose((q @ r).value_in(si.ohm), a.value)
    lower = cast(Quantity[Array], np.linalg.cholesky(a))
    np.testing.assert_allclose((lower @ lower.T).value_in(si.ohm), a.value)
    u, s, vh = cast(tuple[Quantity[Array], Quantity[Array], Quantity[Array]], np.linalg.svd(a))
    assert u.unit == vh.unit == si.one and s.unit == a.unit
    np.testing.assert_allclose(((u * s) @ vh).value_in(si.ohm), a.value)
    singular = cast(Quantity[Array], np.linalg.svdvals(a))
    assert singular.unit == a.unit
    np.testing.assert_allclose(singular.value, s.value)
    for operation in (np.linalg.eig, np.linalg.eigh):
        w, v = cast(tuple[Quantity[Array], Quantity[Array]], operation(a))
        assert w.unit == a.unit and v.unit == si.one
        np.testing.assert_allclose((a @ v).value_in(si.ohm), (v * w).value_in(si.ohm))
    for operation in (np.linalg.eigvals, np.linalg.eigvalsh):
        w = cast(Quantity[Array], operation(a))
        assert w.unit == a.unit
        np.testing.assert_allclose(np.sort(w.value), np.sort(s.value))


def check_sinc() -> None:
    x = Quantity(np.array([0.0, 50.0, 100.0]), Unit('cm/m'))
    result = cast(Quantity[Array], np.sinc(x))
    assert result.unit == si.one
    np.testing.assert_allclose(result.value, np.sinc(x.value_in(si.one)))


check_arrays(np.asarray([0.0, 20.0, 30.0], dtype=np.float64))
check_linalg()
check_factorizations()
check_sinc()
