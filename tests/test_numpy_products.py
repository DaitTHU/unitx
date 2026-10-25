"""Product reductions must produce one physically meaningful unit."""

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


@pytest.mark.parametrize('operation', [np.prod, np.multiply.reduce, np.nanprod])
@pytest.mark.parametrize('axis,count', [(None, 24), (0, 2), (-1, 4), ((), 1), ((0, -1), 8)])
def test_numpy_product_axes(
    operation: Callable[..., object], axis: None | int | tuple[int, ...], count: int
) -> None:
    q = Quantity(np.linspace(1, 2, 24).reshape(2, 3, 4), Factor(2) * Unit('cm'))
    q.value.flags.writeable = False
    for keepdims in (False, True):
        result = operation(q, axis=axis, keepdims=keepdims)
        assert result.unit == q.unit**count
        np.testing.assert_allclose(result.value, operation(q.value, axis=axis, keepdims=keepdims))


def test_numpy_product_defaults() -> None:
    q = Quantity(np.array([[2, 3], [4, 5]]), 'm')
    result = np.prod(q)
    assert result.unit == Unit('m4') and result.value == 120
    result = np.multiply.reduce(q)
    assert result.unit == Unit('m2')
    np.testing.assert_array_equal(result.value, [8, 15])


@pytest.mark.parametrize('operation', [np.prod, np.multiply.reduce, np.nanprod])
@pytest.mark.parametrize('value', [2, np.float32(2), np.array(2.0)])
@pytest.mark.parametrize('axis', [None, 0, -1, ()])
def test_numpy_product_scalar(
    operation: Callable[..., object],
    value: int | np.float32 | NDArray[np.generic],
    axis: None | int | tuple[int, ...],
) -> None:
    result = operation(Quantity(value, 'm'), axis=axis)
    assert result.unit == Unit('m') and result.value == 2
    assert isinstance(result.value, np.number)


@pytest.mark.parametrize('operation', [np.prod, np.multiply.reduce, np.nanprod])
@pytest.mark.parametrize(
    'shape,axis,count', [((0,), None, 0), ((0, 3), 0, 0), ((0, 3), 1, 3), ((2, 0), 0, 2)]
)
def test_numpy_product_empty(
    operation: Callable[..., object], shape: tuple[int, ...], axis: None | int, count: int
) -> None:
    q = Quantity(np.empty(shape), Factor(2) * Unit('m'))
    for where in (True, np.ones(shape, dtype=bool)):
        result = operation(q, axis=axis, where=where)
        assert result.unit == q.unit**count
        np.testing.assert_array_equal(result.value, operation(q.value, axis=axis, where=where))
    result = operation(q, axis=axis, where=False, initial=Quantity(2, 's'))
    assert result.unit == Unit('s')
    np.testing.assert_array_equal(result.value, np.full(np.shape(result.value), 2))


@pytest.mark.parametrize('operation', [np.prod, np.multiply.reduce, np.nanprod])
@pytest.mark.parametrize(
    'where,count',
    [
        (True, 3),
        (False, 0),
        (np.bool_(False), 0),
        (np.array(False), 0),
        (np.array([True, False, True]), 2),
        (np.array([[True, False, True], [False, True, True]]), 2),
    ],
)
def test_numpy_product_where(
    operation: Callable[..., object], where: bool | np.bool | NDArray[np.generic], count: int
) -> None:
    q = Quantity(np.array([[2, 3, 4], [5, 6, 7]]), 'm')
    result = operation(q, axis=1, where=where)
    assert result.unit == q.unit**count
    np.testing.assert_array_equal(result.value, operation(q.value, axis=1, where=where))


@pytest.mark.parametrize('operation', [np.prod, np.multiply.reduce, np.nanprod])
@pytest.mark.parametrize(
    'initial', [0, 2, 2j, Decimal('0.5'), Fraction(1, 2), np.float32(2), np.array(2)]
)
def test_numpy_product_initial(
    operation: Callable[..., object],
    initial: int | complex | Decimal | Fraction | np.float32 | NDArray[np.generic],
) -> None:
    q = Quantity(np.array([2.0, 3.0], dtype=complex if np.iscomplexobj(initial) else float), 'm')
    seed = float(initial) if isinstance(initial, (Decimal, Fraction)) else initial
    for initial_q in (initial, Quantity(initial, Factor(3) * Unit('s'))):
        result = operation(q, initial=initial_q)
        expected_unit = q.unit**2
        if isinstance(initial_q, Quantity):
            expected_unit *= initial_q.unit
        assert result.unit == expected_unit
        assert result.value == operation(q.value, initial=seed)


@pytest.mark.parametrize('operation', [np.prod, np.multiply.reduce, np.nanprod])
def test_numpy_product_initial_none(operation: Callable[..., object]) -> None:
    assert operation(Quantity(np.array([2, 3]), 'm'), initial=None).value == 6
    with pytest.raises(ValueError, match='zero-size'):
        operation(Quantity(np.array([]), 'm'), initial=None)


@pytest.mark.parametrize('operation', [np.prod, np.multiply.reduce, np.nanprod])
@pytest.mark.parametrize('unit', ['km/m', 'deg', ''])
def test_numpy_product_dimensionless(operation: Callable[..., object], unit: str) -> None:
    q = Quantity(np.array([[2.0, 3, 4], [5, 6, 7]]), unit)
    mask = np.array([[True, False, False], [True, True, False]])
    result = operation(q, axis=1, where=mask, initial=Quantity(2, 's'))
    assert result.unit == Unit('s')
    np.testing.assert_allclose(
        result.value, operation(q.value_in(''), axis=1, where=mask, initial=2)
    )
    empty = operation(q, axis=1, where=False)
    assert empty.unit == Unit()
    np.testing.assert_array_equal(empty.value, [1, 1])


@pytest.mark.parametrize('operation', [np.prod, np.multiply.reduce, np.nanprod])
def test_numpy_product_masked_scale(operation: Callable[..., object]) -> None:
    q = Quantity(np.array([1e300, 2.0]), 'km/m')
    with np.errstate(over='raise'):
        result = operation(q, where=np.array([False, True]))
    assert result.unit == Unit() and result.value == 2000


@pytest.mark.parametrize('operation', [np.prod, np.multiply.reduce, np.nanprod])
def test_numpy_product_out_conversion(operation: Callable[..., object]) -> None:
    q = Quantity(np.array([[2.0, 3, 4], [5, 6, 7]])[:, ::-1], Factor(2) * Unit('cm'))
    mask = np.array([True, False, True])
    storage = np.full((2, 2), -123.0)
    target = Quantity(storage[:, 0], 'm2 s')
    buffer = target.value
    expected = operation(q.value, axis=1, where=mask) * 0.02**2 * 120
    assert operation(q, axis=1, where=mask, initial=Quantity(2, 'min'), out=target) is target
    assert target.value is buffer and target.unit == Unit('m2 s')
    np.testing.assert_allclose(buffer, expected)
    np.testing.assert_array_equal(storage[:, 1], -123)
    assert operation(q, axis=1, where=False, out=Quantity(buffer, '')) is not None
    np.testing.assert_array_equal(buffer, 1)


@pytest.mark.parametrize('operation', [np.prod, np.multiply.reduce, np.nanprod])
def test_numpy_product_out_alias(operation: Callable[..., object]) -> None:
    storage = np.array([[2.0, 3], [4, 5]])
    q = Quantity(storage, 'm')
    target = Quantity(storage[0], 'cm2')
    assert operation(q, axis=0, out=target) is target
    np.testing.assert_array_equal(storage, [[80000, 150000], [4, 5]])


def test_numpy_product_out_positional() -> None:
    q = Quantity(np.array([2.0, 3.0]), 'm')
    target = Quantity(np.array(-123.0), 'm2')
    assert np.prod(q, None, None, target, False) is target
    assert np.multiply.reduce(q, None, None, target, False) is target
    assert np.multiply.reduce(q, out=(target,)) is target
    assert np.multiply.reduce(q, out=(None,)).value == 6
    assert target.value == 6


@pytest.mark.parametrize('operation', [np.prod, np.multiply.reduce, np.nanprod])
def test_numpy_product_dtype(operation: Callable[..., object]) -> None:
    for dtype in (np.uint8, np.float32, np.complex128):
        q = Quantity(np.array([20, 20], dtype=dtype), 'm')
        expected = operation(q.value, dtype=dtype)
        result = operation(q, dtype=dtype)
        assert result.value.dtype == expected.dtype
        assert result.value == expected and result.unit == Unit('m2')


@pytest.mark.parametrize('operation', [np.prod, np.multiply.reduce, np.nanprod])
@pytest.mark.parametrize(
    'axis,mask',
    [(1, np.array([[True, False, False], [True, True, False]])), ((), np.array([[True], [False]]))],
)
def test_numpy_product_invalid_counts(
    operation: Callable[..., object], axis: int | tuple[int, ...], mask: NDArray[np.generic]
) -> None:
    # Equal numerical outputs (even zero) cannot hide heterogeneous units.
    q = Quantity(np.zeros((2, 3)), 'm')
    target = Quantity(np.full(2, -123.0), 'm2')
    with pytest.raises(DimensionError, match='equal element counts'):
        operation(q, axis=axis, where=mask, out=target)
    np.testing.assert_array_equal(target.value, -123)


@pytest.mark.parametrize('operation', [np.prod, np.multiply.reduce, np.nanprod])
@pytest.mark.parametrize('initial', [True, '1', [1], np.array([1]), Quantity(np.array([1]), 's')])
def test_numpy_product_invalid_initial(
    operation: Callable[..., object],
    initial: bool | str | list[int] | NDArray[np.generic] | Quantity[Magnitude],
) -> None:
    target = Quantity(np.array(-123.0), 'm2')
    with pytest.raises(TypeError, match='initial must be'):
        operation(Quantity(np.array([2, 3]), 'm'), initial=initial, out=target)
    assert target.value == -123


@pytest.mark.parametrize('operation', [np.prod, np.multiply.reduce, np.nanprod])
@pytest.mark.parametrize('where', [[True, False], np.array([1, 0]), Quantity(np.array([1, 0]), '')])
def test_numpy_product_invalid_where(
    operation: Callable[..., object], where: list[bool] | NDArray[np.generic] | Quantity[Magnitude]
) -> None:
    with pytest.raises(TypeError, match='unitless boolean mask'):
        operation(Quantity(np.array([2, 3]), 'm'), where=where)


@pytest.mark.parametrize('operation', [np.prod, np.multiply.reduce, np.nanprod])
def test_numpy_product_invalid_dimension(operation: Callable[..., object]) -> None:
    q = Quantity(np.array([2.0, 3.0]), 'm')
    for unit in ('m', 's'):
        target = Quantity(np.array(-123.0), unit)
        with pytest.raises(DimensionError, match='incompatible dimension'):
            operation(q, out=target)
        assert target.value == -123


@pytest.mark.parametrize('operation', [np.prod, np.multiply.reduce, np.nanprod])
def test_numpy_product_invalid_buffer(operation: Callable[..., object]) -> None:
    q = Quantity(np.array([2.0, 3.0]), 'm')
    for target in (np.array(-123.0), Quantity(1, 'm2')):
        with pytest.raises(TypeError):
            operation(q, out=target)


@pytest.mark.parametrize('operation', [np.prod, np.multiply.reduce, np.nanprod])
def test_numpy_product_invalid_shape(operation: Callable[..., object]) -> None:
    q = Quantity(np.array([2.0, 3.0]), 'm')
    target = Quantity(np.full(2, -123.0), 'm2')
    with pytest.raises(ValueError, match='result shape'):
        operation(q, out=target)
    np.testing.assert_array_equal(target.value, -123)


@pytest.mark.parametrize('operation', [np.prod, np.multiply.reduce, np.nanprod])
def test_numpy_product_invalid_cast(operation: Callable[..., object]) -> None:
    q = Quantity(np.array([2.0, 3.0]), 'cm')
    target = Quantity(np.array(-123), 'm2')
    with pytest.raises(TypeError):
        operation(q, out=target)
    assert target.value == -123


@pytest.mark.parametrize('operation', [np.prod, np.multiply.reduce, np.nanprod])
def test_numpy_product_invalid_readonly(operation: Callable[..., object]) -> None:
    q = Quantity(np.array([2.0, 3.0]), 'm')
    target = Quantity(np.array(-123.0), 'm2')
    target.value.flags.writeable = False
    with pytest.raises(ValueError, match='read-only'):
        operation(q, out=target)
    assert target.value == -123


@pytest.mark.parametrize('operation', [np.prod, np.multiply.reduce, np.nanprod])
@pytest.mark.parametrize('unit', ['m', 'km/m'])
def test_numpy_product_invalid_broadcast(operation: Callable[..., object], unit: str) -> None:
    q = Quantity(np.array(2.0), unit)
    target = Quantity(np.array(-123.0), '')
    with pytest.raises(ValueError):
        operation(q, where=np.array([True, True]), out=target)
    assert target.value == -123


@pytest.mark.parametrize('operation', [np.prod, np.multiply.reduce, np.nanprod])
@pytest.mark.parametrize('axis', [2, (0, 0), (0, 1)])
def test_numpy_product_invalid_axis(
    operation: Callable[..., object], axis: int | tuple[int, ...]
) -> None:
    target = Quantity(np.array(-123.0), 'm')
    with pytest.raises(ValueError):
        operation(Quantity(np.array([2, 3]), 'm'), axis=axis, out=target)
    assert target.value == -123


@pytest.mark.parametrize('operation', [np.prod, np.multiply.reduce, np.nanprod])
@pytest.mark.parametrize('dtype', [object, str, bool])
def test_numpy_product_invalid_dtype(operation: Callable[..., object], dtype: DTypeLike) -> None:
    target = Quantity(np.array(-123.0), 'm2')
    with pytest.raises(TypeError, match='numeric output dtypes'):
        operation(Quantity(np.array([2, 3]), 'm'), dtype=dtype, out=target)
    assert target.value == -123


@pytest.mark.parametrize('operation', [np.prod, np.multiply.reduce, np.nanprod])
def test_numpy_product_invalid_kernel(operation: Callable[..., object]) -> None:
    q = Quantity(np.array([1e300, 1e300]), 'm')
    target = Quantity(np.array(-123.0), 'm2')
    with np.errstate(over='raise'):
        with pytest.raises(FloatingPointError):
            operation(q, out=target)
    assert target.value == -123


def test_numpy_product_foreign_protocol() -> None:
    class Foreign:
        def __array_function__(
            self,
            func: Callable[..., object],
            types: tuple[type[object], ...],
            args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> str:
            return 'foreign dispatched'

        def __array_ufunc__(
            self, ufunc: np.ufunc, method: str, *inputs: object, **kwargs: object
        ) -> str:
            return 'foreign dispatched'

    q = Quantity(np.array([2, 3]), 'm')
    assert np.prod(q, out=Foreign()) == 'foreign dispatched'
    assert np.multiply.reduce(q, out=Foreign()) == 'foreign dispatched'
    for operation in (np.cumprod, np.multiply.accumulate):
        with pytest.raises(TypeError):
            operation(q)
