"""Masked reductions and physical initial values retain a single result unit."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING

import pytest

from unitx._numerics import Magnitude

np = pytest.importorskip('numpy')

from unitx import Quantity, Unit  # noqa: E402
from unitx.exceptions import DimensionError  # noqa: E402
from unitx.factor import Factor  # noqa: E402

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray


REDUCTIONS = [np.sum, np.add.reduce, np.mean, np.std, np.var]
SUMS = [np.sum, np.add.reduce]


@pytest.mark.parametrize('operation', REDUCTIONS)
@pytest.mark.parametrize('axis', [None, 0, -1, (), (0, -1)])
def test_numpy_statistics_where(
    operation: Callable[..., object], axis: None | int | tuple[int, ...]
) -> None:
    values = np.arange(1.0, 49).reshape(2, 3, 8)[..., ::2]
    values.flags.writeable = False
    q = Quantity(values, Factor(2) * Unit('cm'))
    # Unequal selected counts are legitimate for sums and statistics.
    mask = np.array([[[True, False, True, False]], [[False, True, True, True]]])
    # axis=() with a false mask has empty samples: tested separately below.
    if axis == ():
        mask = np.array(True)
    for keepdims in (False, True):
        result = operation(q, axis=axis, keepdims=keepdims, where=mask)
        expected = operation(values, axis=axis, keepdims=keepdims, where=mask)
        assert result.unit == (q.unit**2 if operation is np.var else q.unit)
        np.testing.assert_allclose(result.value, expected)
        assert np.asarray(result.value).dtype == np.asarray(expected).dtype


@pytest.mark.parametrize('operation', REDUCTIONS)
@pytest.mark.parametrize('value', [2, np.float32(2), np.array(2.0)])
def test_numpy_statistics_scalar(
    operation: Callable[..., object], value: int | np.float32 | NDArray[np.generic]
) -> None:
    result = operation(Quantity(value, 'km/m'), where=np.bool_(True))
    assert result.unit == Unit('km2/m2' if operation is np.var else 'km/m')
    assert result.value == operation(value, where=True)
    assert isinstance(result.value, np.number)


@pytest.mark.parametrize('operation', [np.mean, np.std, np.var])
@pytest.mark.parametrize('shape,mask', [((0, 3), True), ((2, 3), False), ((2, 3), np.array(False))])
def test_numpy_statistics_empty(
    operation: Callable[..., object], shape: tuple[int, ...], mask: bool | NDArray[np.generic]
) -> None:
    q = Quantity(np.ones(shape), 'm')
    with pytest.warns(RuntimeWarning):
        result = operation(q, axis=0, where=mask)
    assert result.unit == Unit('m2' if operation is np.var else 'm')
    np.testing.assert_array_equal(np.isnan(result.value), np.ones(3, dtype=bool))


@pytest.mark.parametrize('operation', REDUCTIONS)
def test_numpy_statistics_masked_nan(operation: Callable[..., object]) -> None:
    q = Quantity(np.array([[1.0, np.nan, 3], [np.inf, 5, 7]]), 'm')
    mask = np.array([[True, False, True], [False, True, True]])
    with np.errstate(all='raise'):
        result = operation(q, axis=1, where=mask)
    expected = [2, 6] if operation is np.mean else [4, 12] if operation in SUMS else [1, 1]
    np.testing.assert_allclose(result.value, expected)


@pytest.mark.parametrize('operation', [np.std, np.var])
def test_numpy_statistics_ddof(operation: Callable[..., object]) -> None:
    q = Quantity(np.array([[1 + 2j, 3j, 5], [2j, 4, 6j]]), 'm')
    mask = np.array([[True, True, False], [True, True, True]])
    for ddof in (0, 0.5, 1):
        result = operation(q, axis=1, where=mask, ddof=ddof, keepdims=True)
        np.testing.assert_allclose(
            result.value, operation(q.value, axis=1, where=mask, ddof=ddof, keepdims=True)
        )
        assert not np.iscomplexobj(result.value)


@pytest.mark.parametrize('operation', SUMS)
@pytest.mark.parametrize('initial', [2, np.float32(2), np.array(2.0), Fraction(2), Decimal(2)])
def test_numpy_sum_initial(
    operation: Callable[..., object],
    initial: int | np.float32 | NDArray[np.generic] | Fraction | Decimal,
) -> None:
    q = Quantity(np.array([[2.0, 3, 4], [5, 6, 7]]), Factor(2) * Unit('cm'))
    result = operation(
        q, axis=1, initial=Quantity(initial, 'm'), where=np.array([True, False, True])
    )
    assert result.unit == q.unit
    np.testing.assert_array_equal(result.value, [106, 112])
    # Bare values are coherent dimensionless coordinates, not stored coordinates.
    scaled = operation(Quantity(np.array([1.0, 2.0]), 'km/m'), initial=initial)
    assert scaled.unit == Unit('km/m')
    assert scaled.value == pytest.approx(3.002)


@pytest.mark.parametrize('operation', SUMS)
def test_numpy_sum_identity(operation: Callable[..., object]) -> None:
    q = Quantity(np.array([2.0, 3.0]), 'm')
    for zero in (0, 0.0, Fraction(0), Decimal(0), np.int64(0), np.array(0.0)):
        assert operation(q, initial=zero).value == 5
    assert operation(q, initial=0j, dtype=complex).value == 5
    with pytest.raises(TypeError):
        operation(q, initial=0j)
    result = operation(q, where=False, initial=Quantity(25, 'cm'))
    assert result.unit == q.unit and result.value == 0.25
    result = operation(Quantity(np.empty((0, 3)), 'm'), axis=0)
    assert result.unit == q.unit
    np.testing.assert_array_equal(result.value, np.zeros(3))
    assert operation(q, initial=None).value == 5
    with pytest.raises(ValueError, match='no identity'):
        operation(Quantity(np.array([]), 'm'), initial=None)


@pytest.mark.parametrize('operation', SUMS)
def test_numpy_sum_defaults(operation: Callable[..., object]) -> None:
    q = Quantity(np.arange(6).reshape(2, 3), 'm')
    result = operation(q)
    np.testing.assert_array_equal(result.value, operation(q.value))
    # Explicit dtype selects the accumulator; the converted initial is cast to it.
    result = operation(q, axis=1, dtype=np.int32, initial=Quantity(50, 'cm'))
    assert result.value.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(result.value, [3, 12])
    # np.sum's existing positional signature includes initial and where.
    if operation is np.sum:
        result = operation(q, 1, None, None, True, Quantity(2, 'm'), True)
        np.testing.assert_array_equal(result.value, [[5], [14]])


@pytest.mark.parametrize('operation', REDUCTIONS)
def test_numpy_statistics_out(operation: Callable[..., object]) -> None:
    storage = np.arange(12.0).reshape(2, 6)
    q = Quantity(storage, Factor(2) * Unit('m'))
    # The output overlaps the input and is noncontiguous.
    target = Quantity(storage[:, 0], 'cm2' if operation is np.var else 'cm')
    mask = np.array([True, False, True, False, True, False])
    unit = q.unit**2 if operation is np.var else q.unit
    expected = Quantity(operation(storage.copy(), axis=1, where=mask), unit).value_in(target.unit)
    magnitude, dtype, target_unit = target.value, target.value.dtype, target.unit
    assert operation(q, axis=1, out=target, where=mask) is target
    assert target.value is magnitude and target.value.dtype == dtype and target.unit is target_unit
    np.testing.assert_allclose(target.value, expected)


@pytest.mark.parametrize('operation', REDUCTIONS)
def test_numpy_statistics_mask_alias(operation: Callable[..., object]) -> None:
    storage = np.arange(1.0, 7)
    q = Quantity(storage, 'm')
    mask = storage.view(np.bool_)[:: storage.itemsize]
    mask[:] = [True, False, True, False, True, True]
    target = Quantity(storage[:1].reshape(()), 'm2' if operation is np.var else 'm')
    expected = operation(storage.copy(), where=mask.copy())
    assert operation(q, axis=None, out=target, where=mask) is target
    np.testing.assert_allclose(target.value, expected)


@pytest.mark.parametrize('operation', REDUCTIONS)
@pytest.mark.parametrize('dtype', [np.int32, np.float32, np.complex128])
def test_numpy_statistics_dtype(operation: Callable[..., object], dtype: DTypeLike) -> None:
    q = Quantity(np.arange(6, dtype=dtype).reshape(2, 3), 'm')
    if operation is np.std and dtype is np.int32:
        # NumPy's integer variance buffer cannot hold the floating square root.
        with pytest.raises(TypeError):
            operation(q.value, axis=1, dtype=dtype, where=np.array([True, False, True]))
        with pytest.raises(TypeError):
            operation(q, axis=1, dtype=dtype, where=np.array([True, False, True]))
        return
    result = operation(q, axis=1, dtype=dtype, where=np.array([True, False, True]))
    expected = operation(q.value, axis=1, dtype=dtype, where=np.array([True, False, True]))
    assert result.value.dtype == expected.dtype
    np.testing.assert_allclose(result.value, expected)


@pytest.mark.parametrize('operation', REDUCTIONS)
@pytest.mark.parametrize('where', [1, [True, False], np.array([1, 0]), Quantity(1, '')])
def test_numpy_statistics_invalid_where(
    operation: Callable[..., object],
    where: int | list[bool] | NDArray[np.generic] | Quantity[Magnitude],
) -> None:
    target = Quantity(np.array(-123.0), 'm2' if operation is np.var else 'm')
    with pytest.raises(TypeError, match='unitless boolean mask'):
        operation(Quantity(np.array([2.0, 3]), 'm'), out=target, where=where)
    assert target.value == -123


@pytest.mark.parametrize('operation', SUMS)
@pytest.mark.parametrize(
    'initial',
    [
        1,
        Fraction(1, 10**400),
        Decimal('1e-1000'),
        Quantity(0, 's'),
        Quantity(2, ''),
        Quantity(1, 's'),
    ],
)
def test_numpy_sum_invalid_dimension(
    operation: Callable[..., object], initial: int | Fraction | Decimal | Quantity[Magnitude]
) -> None:
    target = Quantity(np.array(-123.0), 'm')
    with pytest.raises(DimensionError, match='initial has an incompatible dimension'):
        operation(Quantity(np.array([2.0, 3]), 'm'), out=target, initial=initial)
    assert target.value == -123


@pytest.mark.parametrize('operation', SUMS)
@pytest.mark.parametrize('initial', [True, '1', [1], np.array([1]), Quantity(np.array([1]), 'm')])
def test_numpy_sum_invalid_initial(
    operation: Callable[..., object],
    initial: bool | str | list[int] | NDArray[np.generic] | Quantity[Magnitude],
) -> None:
    with pytest.raises(TypeError, match='initial must be a numeric scalar'):
        operation(Quantity(np.array([2.0, 3]), 'm'), initial=initial)


@pytest.mark.parametrize('operation', REDUCTIONS)
@pytest.mark.parametrize(
    'options', [{'where': np.ones(3, dtype=bool)}, {'axis': 2}, {'dtype': object}]
)
def test_numpy_statistics_invalid_kernel(
    operation: Callable[..., object], options: dict[str, object]
) -> None:
    target = Quantity(np.array(-123.0), 'm2' if operation is np.var else 'm')
    with pytest.raises((ValueError, TypeError)):
        operation(Quantity(np.array([2.0, 3]), 'm'), out=target, **options)
    assert target.value == -123


@pytest.mark.parametrize('operation', REDUCTIONS)
def test_numpy_statistics_invalid_output(operation: Callable[..., object]) -> None:
    q = Quantity(np.array([[2.0, 3], [5, 7]]), 'm')
    unit = 'm2' if operation is np.var else 'm'
    for target, error in (
        (Quantity(np.array(-123.0), 's'), DimensionError),
        (Quantity(np.full(3, -123.0), unit), ValueError),
        (Quantity(np.full(2, -123), 'cm2' if operation is np.var else 'cm'), TypeError),
    ):
        with pytest.raises(error):
            operation(q, axis=1, where=np.array([True, False]), out=target)
        np.testing.assert_array_equal(target.value, -123)
    target = Quantity(np.full(2, -123.0), unit)
    target.value.flags.writeable = False
    with pytest.raises(ValueError, match='read-only'):
        operation(q, axis=1, where=True, out=target)
    np.testing.assert_array_equal(target.value, -123)


@pytest.mark.parametrize('operation', SUMS)
def test_numpy_sum_invalid_conversion(operation: Callable[..., object]) -> None:
    q = Quantity(np.array([1.0, 2]), 'm')
    target = Quantity(np.array(-123.0), 'm')
    with np.errstate(over='raise'):
        with pytest.raises(FloatingPointError):
            operation(q, out=target, initial=Quantity(1e308, 'km'))
    assert target.value == -123


def test_numpy_statistics_invalid_options() -> None:
    q = Quantity(np.arange(3.0), 'm')
    for operation in (np.mean, np.std, np.var, np.add.accumulate):
        with pytest.raises(TypeError):
            operation(q, initial=Quantity(1, 'm'))
    with pytest.raises(TypeError):
        np.add.accumulate(q, where=True)
    for option in ({'mean': Quantity(1, 'm')}, {'correction': 1}):
        with pytest.raises(TypeError):
            np.var(q, **option)
