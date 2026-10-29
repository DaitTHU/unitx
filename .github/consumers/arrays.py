"""Check array runtime and inferred types using the installed wheel."""

from typing import Any, assert_type, cast

import numpy as np
from numpy.typing import NDArray

from unitx import AffineQuantity, LogQuantity, LogUnit, Quantity, Unit
from unitx import math as unitx_math
from unitx import numpy as unp
from unitx.cgs import emu, esu, gaussian, hl
from unitx.exceptions import DimensionError, ImplicitConversionError, UnitOperationError
from unitx.si import si
from unitx.typing import Array, NumpyValue

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
    ratio = Quantity(values, si.percent)
    assert_type(ratio.n, NDArray[np.float64] | np.float64)
    np.testing.assert_allclose(ratio.n, values / 100)
    assert_type(ratio.as_pair(), tuple[NDArray[np.float64], Unit])
    assert ratio.as_pair()[0] is values
    assert_type(ratio.to_base(), Quantity[NDArray[np.float64] | np.float64])
    assert_type(ratio.simplify(), Quantity[NDArray[np.float64] | np.float64])
    assert_type(Quantity(values, si.m).compact(), Quantity[NDArray[np.float64] | np.float64])
    assert Quantity(values, si.cm * si.m).simplify().unit == si.m**2
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
    np.testing.assert_allclose(result.value, np.sinc(np.asarray(x.value_in(si.one))))


def check_fft() -> None:
    x = Quantity(np.arange(6.0).reshape(2, 3), si.m)
    for name in (
        'fft',
        'ifft',
        'rfft',
        'irfft',
        'hfft',
        'ihfft',
        'fft2',
        'ifft2',
        'fftn',
        'ifftn',
        'rfft2',
        'irfft2',
        'rfftn',
        'irfftn',
    ):
        operation = getattr(np.fft, name)
        result = cast(Quantity[Array], operation(x))
        assert result.unit == x.unit
        np.testing.assert_allclose(result.value, operation(x.value))
    signal = Quantity(np.array([1.0, 2, 3, 4, 5]), si.m)
    spectrum = cast(Quantity[Array], np.fft.rfft(signal))
    restored = cast(Quantity[Array], np.fft.irfft(spectrum, n=signal.size))
    np.testing.assert_allclose(restored.value_in(si.m), signal.value)
    spacing = 2 * si.ms
    # Some NumPy versions' frequency-helper stubs reject Quantity spacing.
    frequency = cast(Quantity[Array], cast(Any, np.fft.rfftfreq)(signal.size, d=spacing))
    assert frequency.unit == si.ms**-1
    np.testing.assert_allclose(frequency.value_in(si.Hz), [0, 100, 200])
    full = cast(Quantity[Array], cast(Any, np.fft.fftfreq)(signal.size, d=spacing))
    centered = cast(Quantity[Array], np.fft.fftshift(full))
    np.testing.assert_allclose(centered.value_in(si.Hz), [-200, -100, 0, 100, 200])
    unshifted = cast(Quantity[Array], np.fft.ifftshift(centered))
    np.testing.assert_allclose(unshifted.value, full.value)
    output = Quantity(np.empty(3, dtype=complex), si.cm)
    assert signal.__array_function__(np.fft.rfft, (Quantity,), (signal,), {'out': output}) is output
    np.testing.assert_allclose(output.value_in(si.m), spectrum.value)
    assert (spectrum * spacing).unit == si.m * si.ms


def check_statistics() -> None:
    q = Quantity(np.array([0.0, 1, 2, 3]), si.m)
    span = cast(Quantity[Array], np.ptp(q))
    assert span == 3 * si.m
    covariance = cast(Quantity[Array], np.cov(q, q.to(si.cm)))
    assert covariance.unit == si.m**2
    coefficient = cast(Quantity[Array], np.corrcoef(q, Quantity(q.value, si.s)))
    assert coefficient.unit == si.one
    np.testing.assert_allclose(coefficient.value, np.ones((2, 2)))
    density, edges = cast(
        tuple[Quantity[Array], Quantity[Array]], np.histogram(q, bins=2, density=True)
    )
    assert density.unit == si.m**-1 and edges.unit == si.m
    integral = cast(Quantity[Array], np.sum(density * np.diff(edges)))
    np.testing.assert_allclose(integral.value, 1)
    time = Quantity(q.value, si.s)
    h2, xe, ye = cast(
        tuple[Quantity[Array], Quantity[Array], Quantity[Array]],
        np.histogram2d(q, time, bins=2, density=True),
    )
    hd, axes = cast(
        tuple[Quantity[Array], list[Quantity[Array]]],
        np.histogramdd((q, time), bins=2, density=True),
    )
    assert h2.unit == hd.unit == (si.m * si.s) ** -1
    assert xe.unit == axes[0].unit == si.m and ye.unit == axes[1].unit == si.s
    np.testing.assert_allclose(h2.value, hd.value)
    generated = cast(Quantity[Array], np.histogram_bin_edges(q, bins=2))
    np.testing.assert_array_equal(generated.value, edges.value)
    indices = np.digitize(q, edges)
    assert indices.dtype.kind in 'iu'
    weighted = cast(
        Quantity[Array], np.bincount(np.array([0, 1, 0, 1]), weights=Quantity(np.ones(4), si.kg))
    )
    assert weighted.unit == si.kg
    np.testing.assert_array_equal(weighted.value, [2, 2])
    gapped = Quantity(np.array([1.0, np.nan, 3]), si.m)
    accumulated = cast(Quantity[Array], np.nancumsum(gapped))
    np.testing.assert_array_equal(accumulated.value, [1, 1, 4])
    for operation in (np.cumprod, np.nancumprod):
        result = cast(Quantity[Array], operation(Quantity(np.array([1.0, 2, 3]), Unit('cm/m'))))
        assert result.unit == si.one
        np.testing.assert_allclose(result.value, [0.01, 0.0002, 0.000006])
    avg = cast(Quantity[Array], np.mean(q))
    result = cast(
        Quantity[Array],
        q.__array_function__(np.var, (Quantity,), (q,), {'mean': avg, 'correction': 1}),
    )
    assert result.unit == si.m**2
    np.testing.assert_allclose(result.value, np.var(q.value, ddof=1))


check_arrays(np.asarray([0.0, 20.0, 30.0], dtype=np.float64))
check_linalg()
check_factorizations()
check_sinc()
check_fft()
check_statistics()


def check_creation() -> None:
    a = Quantity(0.0, si.m)
    b = Quantity(100.0, si.cm)
    samples, spacing = cast(
        tuple[Quantity[Array], Quantity[Array]], np.linspace(a, b, 5, retstep=True)
    )
    assert samples.unit == spacing.unit == si.m
    np.testing.assert_allclose(spacing.value, 0.25)
    filled = cast(Quantity[Array], np.full_like(samples, Quantity(50.0, si.cm)))
    np.testing.assert_allclose(filled.value, 0.5)
    logarithmic = cast(Quantity[Array], np.geomspace(Quantity(1.0, si.m), Quantity(100.0, si.m), 3))
    np.testing.assert_allclose(logarithmic.value, [1, 10, 100])
    exponent = Quantity(2.0, si.one)
    loggrid = cast(Quantity[Array], np.logspace(0.0, exponent, 3))
    assert loggrid.unit == si.one
    np.testing.assert_allclose(loggrid.value, [1, 10, 100])
    grids = cast(
        tuple[Quantity[Array], Quantity[Array]],
        np.meshgrid(samples, Quantity(np.arange(2.0), si.s)),
    )
    assert grids[0].unit == si.m and grids[1].unit == si.s
    padded = cast(Quantity[Array], np.pad(samples, 1, mode='edge'))
    windows = cast(Quantity[Array], np.lib.stride_tricks.sliding_window_view(padded, 3))
    assert windows.unit == si.m
    assert not cast(NDArray[np.generic], windows.value).flags.writeable


check_creation()


def check_writes_and_methods() -> None:
    q = Quantity(np.zeros(3), si.m)
    # NumPy output/destination stubs do not model Quantity buffers.
    q.__array_function__(np.copyto, (Quantity,), (q, Quantity(100.0, si.cm)), {})
    q.__array_ufunc__(np.add, 'at', q, [0, 0, 2], Quantity(50.0, si.cm))
    np.testing.assert_allclose(q.value, [2, 1, 1.5])
    segmented = cast(Quantity[Array], np.add.reduceat(q, [0, 2]))
    np.testing.assert_allclose(segmented.value, [3, 1.5])
    maximum = cast(Quantity[Array], np.maximum.accumulate(q))
    assert maximum.unit == si.m
    product = cast(Quantity[Array], np.multiply.outer(q, Quantity(np.arange(2.0), si.s)))
    assert product.unit == si.m * si.s
    q.__array_function__(np.put, (Quantity,), (q, [1], Quantity(250.0, si.cm)), {})
    np.testing.assert_allclose(q.value, [2, 2.5, 1.5])


check_writes_and_methods()


def check_compatibility() -> None:
    q = Quantity(np.array([1.0, np.nan, 3.0]), si.m)
    clean = cast(
        Quantity[Array],
        q.__array_function__(np.nan_to_num, (Quantity,), (q,), {'nan': Quantity(200.0, si.cm)}),
    )
    prefix = cast(Quantity[Array], np.cumulative_sum(clean, include_initial=True))
    np.testing.assert_array_equal(prefix.value, [0, 1, 3, 6])
    assert np.array_equal(clean, clean.to(si.cm))
    blocks = cast(Quantity[Array], np.block([[clean], [clean.to(si.cm)]]))
    assert blocks.unit == si.m and blocks.shape == (2, 3)
    mantissa, exponent = cast(tuple[Quantity[Array], NDArray[np.int32]], np.frexp(clean))
    restored = cast(Quantity[Array], np.ldexp(mantissa, exponent))
    np.testing.assert_array_equal(restored.value, clean.value)
    assert np.shares_memory(clean, clean[1:])
    selected = cast(Quantity[Array], np.choose(np.array([0, 1, 0]), [clean, clean.to(si.cm)]))
    np.testing.assert_array_equal(selected.value, clean.value)
    unstacked = cast(tuple[Quantity[Array], ...], np.unstack(blocks))
    assert unstacked[0].unit == si.m
    identity = Quantity(np.eye(2), si.m)
    product = cast(Quantity[Array], np.linalg.matmul(identity, identity))
    diagonal = cast(Quantity[Array], np.linalg.diagonal(product))
    assert diagonal.unit == si.m**2


check_compatibility()


def check_array_methods(target: NDArray[np.float64]) -> None:
    values = np.arange(6, dtype=np.float32)
    q = Quantity(values, si.m)
    columns = q.reshape(2, 3)
    assert_type(columns, Quantity[NDArray[np.float32]])
    assert_type(columns.ravel(), Quantity[NDArray[np.float32]])
    assert_type(columns.copy(), Quantity[NDArray[np.float32]])
    assert_type(columns.mean(axis=0), Quantity[NDArray[np.float32] | np.float32])
    assert np.shares_memory(columns.value, values)
    assert not np.shares_memory(columns.copy().value, values)
    out = Quantity(target, si.cm)
    assert_type(columns.sum(axis=1, out=out), Quantity[NDArray[np.float64]])
    assert columns.sum(axis=1, out=out) is out
    np.testing.assert_allclose(out.value, [300, 1200])
    for point in (AffineQuantity(values, si.degC), LogQuantity(values, si.dBm)):
        assert type(point.reshape(2, 3).mean(axis=1)) is type(point)
        assert isinstance(point.std(), Quantity)
    signal = Quantity(np.asarray([1 + 2j], dtype=np.complex64), si.V)
    assert_type(signal.reshape(-1), Quantity[NDArray[np.complex64]])
    assert signal.copy().value.dtype == np.complex64
    assert_type(Quantity(np.float32(2), si.m).reshape(()), Quantity[NumpyValue])
    # NumPy's dx annotation remains numeric despite this native runtime path.
    integral = cast(
        Quantity[Array],
        np.trapezoid(np.ones(3), dx=2 * si.s),  # pyright: ignore[reportCallIssue, reportArgumentType]
    )
    assert integral == 4 * si.s


check_array_methods(np.zeros(2, dtype=np.float64))


def check_calculus_types(values: NDArray[np.float64]) -> None:
    coordinate = Quantity(values, si.s)
    signal = 2 * Quantity(values, si.A)
    assert_type(coordinate.to(si.ms), Quantity[NDArray[np.float64] | np.float64])
    assert_type(signal * coordinate, Quantity[NDArray[np.float64] | np.float64])
    weights = cast(Quantity[Array], np.gradient(coordinate, axis=0))
    weights[[0, -1]] /= 2
    integral = cast(Quantity[Array], np.trapezoid(signal, coordinate))
    assert np.sum(signal * weights).isclose(integral)
    points = AffineQuantity(values, si.degC)
    increments = cast(Quantity[Array], np.diff(points))
    assert increments.unit == si.degC.delta_unit
    assert np.diff(points, n=0) is points
    rate = cast(Quantity[Array], np.gradient(points, coordinate, axis=0))
    np.testing.assert_allclose(rate.value, 1)
    levels = LogQuantity(values, si.dBm)
    differences = cast(Quantity[Array], np.diff(levels))
    assert differences.unit == si.dBm.delta_unit
    try:
        np.asarray(coordinate)
    except ImplicitConversionError as error:
        assert isinstance(error, TypeError)
    else:
        error_message = 'Array coercion must preserve the explicit numerical boundary.'
        raise AssertionError(error_message)
    try:
        np.add(points, points)
    except UnitOperationError as error:
        assert isinstance(error, TypeError)
    else:
        error_message = 'Adding two points must fail.'
        raise AssertionError(error_message)


check_calculus_types(np.array([0.0, 1, 3, 6]))


def check_scientific_inference(values: NDArray[np.float64]) -> None:
    spectrum = Quantity(values.reshape(2, 2, 2), si.J)
    mask = np.array([True, False])
    selected = spectrum[mask, 0, 0]
    assert_type(selected, Quantity[NDArray[np.float64]])
    assert_type(spectrum[:, 0, 0], Quantity[NDArray[np.float64]])
    phase = 1j * selected
    assert_type(phase, Quantity[NDArray[np.complex128] | np.complex128])
    assert_type(spectrum.sum(), Quantity[np.float64])
    assert_type(spectrum.sum().value_in(si.J), np.float64)
    assert_type(selected.astype(float), Quantity[NDArray[np.float64]])
    assert_type(Quantity(np.float32(2), si.percent).n, np.float32)
    np.testing.assert_array_equal(selected.value, [values[0]])
    np.testing.assert_array_equal(phase.value, 1j * selected.value)
    assert spectrum.sum().value_in(si.J) == np.sum(values)


check_scientific_inference(np.arange(8, dtype=np.float64))


def check_complex_field(values: NDArray[np.float64]) -> None:
    z = values * si.m
    gouy = unitx_math.atan(z / (0.5 * si.m))
    geometry = unitx_math.hypot(1, z / (0.5 * si.m))
    field = unitx_math.conj(unitx_math.exp(1j * gouy) / geometry)
    assert_type(field, Quantity[NDArray[np.complex128] | np.complex128])
    for _ in range(3):
        field *= unitx_math.exp(-1j * gouy)
        assert_type(field, Quantity[NDArray[np.complex128] | np.complex128])
    angle = np.arctan(np.linspace(-1, 1, 5) / 0.5)
    expected = np.exp(-4j * angle) / np.hypot(1, np.linspace(-1, 1, 5) / 0.5)
    np.testing.assert_allclose(field.n, expected, rtol=1e-14)


check_complex_field(np.linspace(-1, 1, 5))


def check_shape_axes(values: NDArray[np.float64]) -> None:
    q = Quantity(values.reshape(2, 1, 4), si.m)
    permuted = q.transpose(2, 0, 1).squeeze(-1)
    assert_type(permuted, Quantity[NDArray[np.float64]])
    assert np.shares_memory(permuted.value, values)
    np.testing.assert_array_equal(permuted.value, values.reshape(2, 4).T)
    assert_type(Quantity(np.float32(2), si.m).squeeze(), Quantity[np.float32])
    for point in (
        AffineQuantity(values.reshape(2, 1, 4), si.degC),
        LogQuantity(values.reshape(2, 1, 4), si.dBm),
    ):
        transformed = point.transpose((2, 0, 1)).squeeze(axis=-1)
        assert type(transformed) is type(point) and transformed.unit is point.unit
        np.testing.assert_array_equal(transformed.value, values.reshape(2, 4).T)


check_shape_axes(np.arange(8.0))


def check_reduction_methods(values: NDArray[np.float64], target: NDArray[np.float64]) -> None:
    samples = Quantity(values, si.m)
    assert samples.min() == 1 * si.m
    assert samples.max() == 3 * si.m
    variance_out = Quantity(target, si.cm**2)
    assert_type(samples.var(out=variance_out), Quantity[NDArray[np.float64]])
    assert samples.var(ddof=1, out=variance_out) is variance_out
    np.testing.assert_allclose(variance_out.value, 10000)
    assert np.power(samples[:, None], np.full((1, 2), 2)).shape == (3, 2)
    assert AffineQuantity(np.array([0.0, 10.0]), si.degC).max() == 10 @ si.degC
    assert LogQuantity(np.array([0.0, 20.0]), si.dBm).var().unit == si.dB**2


check_reduction_methods(np.array([1.0, 2.0, 3.0]), np.array(0.0))

# Explicit typed entries retain units and concrete allocation/derivative arrays.

_typed_samples = Quantity(np.linspace(0.0, 1.0, 5), si.m)
_typed_coordinates = Quantity(np.linspace(0.0, 2.0, 5), si.s)
_typed_integral = unp.trapezoid(_typed_samples, _typed_coordinates)
assert_type(_typed_integral, Quantity[NDArray[np.float64] | np.float64])
assert_type(unp.zeros_like(_typed_samples), Quantity[NDArray[np.float64]])
assert_type(unp.gradient(_typed_samples, _typed_coordinates, axis=0), Quantity[NDArray[np.float64]])
assert _typed_integral.unit == si.m * si.s
np.testing.assert_allclose(unp.gradient(_typed_samples, _typed_coordinates, axis=0).value, 0.5)

assert_type(unitx_math.sinpi(_typed_coordinates / si.s), Quantity[NDArray[np.float64] | np.float64])
np.testing.assert_array_equal(unitx_math.sinpi(_typed_coordinates / si.s).value, [0, 1, 0, -1, 0])
