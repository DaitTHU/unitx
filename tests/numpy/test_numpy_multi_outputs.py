"""Multi-output decompositions preserve units and stage all writes together."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

np = pytest.importorskip('numpy')

from unitx import Quantity, Unit  # noqa: E402
from unitx.exceptions import DimensionError  # noqa: E402
from unitx.factor import Factor  # noqa: E402

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray


@pytest.mark.parametrize('operation', [np.modf, np.frexp])
@pytest.mark.parametrize('unit', ['m', 'km/m', Factor(2) * Unit('cm')])
@pytest.mark.parametrize(
    'value',
    [
        2,
        np.float32(-2.5),
        np.array(-2.5),
        np.empty(0),
        np.array([[0.0, -0.0, -2.5], [3.75, np.inf, np.nan]])[:, ::-1],
    ],
)
def test_numpy_multi_output_values(
    operation: np.ufunc, unit: str | Unit, value: int | np.float32 | NDArray[np.generic]
) -> None:
    q = Quantity(value, unit)
    result = operation(q)
    expected = operation(value)
    assert isinstance(result, tuple) and len(result) == 2
    for i, (actual, raw) in enumerate(zip(result, expected)):
        if operation is np.modf or i == 0:
            assert isinstance(actual, Quantity) and actual.unit == q.unit
            actual = actual.value
        else:
            assert not isinstance(actual, Quantity) and actual.dtype.kind in 'iu'
        assert type(actual) is type(raw) and actual.dtype == raw.dtype
        np.testing.assert_equal(actual, raw)


@pytest.mark.parametrize('operation', [np.modf, np.frexp])
def test_numpy_multi_output_reconstruction(operation: np.ufunc) -> None:
    q = Quantity(np.array([-3.75, -0.0, 0.0, 1.25]), Factor(3) * Unit('cm'))
    first, second = operation(q)
    rebuilt = (
        first + second
        if operation is np.modf
        else Quantity(np.ldexp(first.value, second), first.unit)
    )
    np.testing.assert_array_equal(rebuilt.value_in('m'), q.value_in('m'))


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longdouble])
def test_numpy_frexp_extremes(dtype: DTypeLike) -> None:
    info = np.finfo(dtype)
    values = np.array([info.max, -info.max, info.tiny, info.smallest_subnormal], dtype=dtype)
    q = Quantity(values, Factor(2) * Unit('m'))
    mantissa, exponent = np.frexp(q)
    with np.errstate(over='raise', invalid='raise'):
        rebuilt = Quantity(np.ldexp(mantissa.value, exponent), mantissa.unit)
    assert rebuilt.unit == q.unit
    np.testing.assert_array_equal(rebuilt.value, q.value)


@pytest.mark.parametrize('operation', [np.modf, np.frexp])
def test_numpy_multi_output_buffers(operation: np.ufunc) -> None:
    q = Quantity(np.array([-2.5, 3.75]), Factor(2) * Unit('m'))
    storage = np.full((2, 2), -123.0)
    first = Quantity(storage[:, 0], 'cm')
    second = Quantity(storage[:, 1], 'mm') if operation is np.modf else np.full(2, -123)
    expected = operation(q.value)
    a, b = operation(q, out=(first, second))
    assert a is first and b is second
    assert a.value is not q.value and a.unit == Unit('cm')
    np.testing.assert_allclose(a.value, expected[0] * 200)
    np.testing.assert_allclose(
        b.value if isinstance(b, Quantity) else b,
        expected[1] * (2000 if operation is np.modf else 1),
    )
    a, b = operation(q, first, second)
    assert a is first and b is second


@pytest.mark.parametrize('operation', [np.modf, np.frexp])
@pytest.mark.parametrize('missing', [0, 1, 2])
def test_numpy_multi_output_partial(operation: np.ufunc, missing: int) -> None:
    q = Quantity(np.array([1.25, -2.5]), 'km/m')
    first = None if missing in (0, 2) else Quantity(np.empty((3, 2)), '')
    second = (
        None
        if missing in (1, 2)
        else (
            Quantity(np.empty((3, 2)), '') if operation is np.modf else np.empty((3, 2), dtype=int)
        )
    )
    a, b = operation(q, out=(first, second))
    assert first is None or a is first
    assert second is None or b is second
    expected = operation(q.value)
    shape = (2,) if missing == 2 else (3, 2)
    np.testing.assert_equal(
        a.value, np.broadcast_to(expected[0] * (1 if first is None else 1000), shape)
    )
    raw_b = b.value if isinstance(b, Quantity) else b
    scale = 1000 if operation is np.modf and second is not None else 1
    np.testing.assert_equal(raw_b, np.broadcast_to(expected[1] * scale, shape))


@pytest.mark.parametrize('operation', [np.modf, np.frexp])
def test_numpy_multi_output_mask(operation: np.ufunc) -> None:
    q = Quantity(np.array([1.25, -2.5]), 'm')
    first = Quantity(np.full((3, 2), -123.0), 'cm')
    second = (
        Quantity(np.full((3, 2), -456.0), 'm') if operation is np.modf else np.full((3, 2), -456)
    )
    mask = np.array([[True, False], [False, True], [True, True]])
    assert operation(q, out=(first, second), where=mask)[0] is first
    expected = operation(q.value)
    np.testing.assert_equal(first.value, np.where(mask, expected[0] * 100, -123))
    np.testing.assert_equal(
        second.value if isinstance(second, Quantity) else second, np.where(mask, expected[1], -456)
    )


def test_numpy_multi_output_mask_alias() -> None:
    q = Quantity(np.array([1.25, -2.5]), 'm')
    exponent = np.array([1, 0], dtype=np.int32)
    mask = exponent.view(np.bool_)[:2]
    original_mask = mask.copy()
    first = Quantity(np.full(2, -123.0), 'm')
    np.frexp(q, out=(first, exponent), where=mask)
    np.testing.assert_equal(first.value, np.where(original_mask, np.frexp(q.value)[0], -123))
    np.testing.assert_equal(exponent, np.where(original_mask, np.frexp(q.value)[1], [1, 0]))


@pytest.mark.parametrize('operation', [np.modf, np.frexp])
def test_numpy_multi_output_input_alias(operation: np.ufunc) -> None:
    q = Quantity(np.array([1.25, -2.5]), 'm')
    expected = operation(q.value.copy())
    second = Quantity(np.empty(2), 'm') if operation is np.modf else np.empty(2, dtype=int)
    assert operation(q, out=(q, second))[0] is q
    np.testing.assert_equal(q.value, expected[0])
    np.testing.assert_equal(second.value if isinstance(second, Quantity) else second, expected[1])


@pytest.mark.parametrize('operation', [np.modf, np.frexp])
@pytest.mark.parametrize('failure', ['dimension', 'cast', 'shape', 'readonly', 'scalar', 'kind'])
def test_numpy_multi_output_invalid_second(operation: np.ufunc, failure: str) -> None:
    q = Quantity(np.array([1.25, -2.5]), 'm')
    first = Quantity(np.full(2, -123.0), 'cm')
    raw = np.full(2, -456.0) if operation is np.modf else np.full(2, -456)
    second = Quantity(raw, 'm') if operation is np.modf else raw
    error = TypeError
    if failure == 'dimension':
        second = Quantity(raw, 's')
        error = DimensionError if operation is np.modf else TypeError
    elif failure == 'cast':
        second = (
            Quantity(np.full(2, -456), 'm')
            if operation is np.modf
            else np.full(2, 123, dtype=np.uint8)
        )
    elif failure == 'shape':
        second = Quantity(np.full(3, -456.0), 'm') if operation is np.modf else np.full(3, -456)
        error = ValueError
    elif failure == 'readonly':
        raw.flags.writeable = False
        error = ValueError
    elif failure == 'scalar':
        second = Quantity(-456, 'm') if operation is np.modf else -456
    else:
        second = raw if operation is np.modf else Quantity(raw, '')
    snapshot = second.value if isinstance(second, Quantity) else second
    snapshot = snapshot.copy() if isinstance(snapshot, np.ndarray) else snapshot
    with pytest.raises(error):
        operation(q, out=(first, second))
    np.testing.assert_equal(first.value, -123)
    np.testing.assert_equal(second.value if isinstance(second, Quantity) else second, snapshot)


@pytest.mark.parametrize('operation', [np.modf, np.frexp])
def test_numpy_multi_output_invalid_overlap(operation: np.ufunc) -> None:
    storage = np.full(3, -123.0)
    first = Quantity(storage[:2], 'm')
    second = Quantity(storage[1:], 'm') if operation is np.modf else storage.view(np.int64)[1:]
    with pytest.raises(ValueError, match='must not overlap'):
        operation(Quantity(np.array([1.25, 2.5]), 'm'), out=(first, second))
    np.testing.assert_equal(storage, -123)


@pytest.mark.parametrize('operation', [np.modf, np.frexp])
def test_numpy_multi_output_invalid_shape(operation: np.ufunc) -> None:
    first = Quantity(np.full(2, -123.0), 'm')
    second = (
        Quantity(np.full((3, 2), -456.0), 'm') if operation is np.modf else np.full((3, 2), -456)
    )
    with pytest.raises(ValueError, match='broadcast shape'):
        operation(Quantity(np.array([1.25, 2.5]), 'm'), out=(first, second))
    np.testing.assert_equal(first.value, -123)


@pytest.mark.parametrize('operation', [np.modf, np.frexp])
def test_numpy_multi_output_invalid_kernel(operation: np.ufunc) -> None:
    first = Quantity(np.full(2, -123.0), 'm')
    second = Quantity(np.full(2, -456.0), 'm') if operation is np.modf else np.full(2, -456)
    with pytest.raises(TypeError):
        operation(Quantity(np.array([1j, 2j]), 'm'), out=(first, second))
    np.testing.assert_equal(first.value, -123)
    with pytest.raises(TypeError):
        operation(Quantity(np.array([1.25, 2.5]), 'm'), out=(first, second), where=[True, False])
    np.testing.assert_equal(first.value, -123)


@pytest.mark.parametrize('operation', [np.modf, np.frexp])
def test_numpy_multi_output_foreign(operation: np.ufunc) -> None:
    class Foreign:
        def __array_ufunc__(
            self, ufunc: np.ufunc, method: str, *inputs: object, **kwargs: object
        ) -> str:
            return 'foreign dispatched'

    q = Quantity(np.array([1.25, -2.5]), 'm')
    assert operation(q, out=(None, Foreign())) == 'foreign dispatched'
    assert q.__array_ufunc__(operation, 'reduce', q) is NotImplemented
    assert q.__array_ufunc__(operation, '__call__', q, out=(None,)) is NotImplemented


@pytest.mark.parametrize('operation', [np.modf, np.frexp])
@pytest.mark.parametrize('shape', [(), (0, 2)])
def test_numpy_multi_output_shapes(operation: np.ufunc, shape: tuple[int, ...]) -> None:
    q = Quantity(np.full(shape, 1.25), 'm')
    first = Quantity(np.empty(shape), 'cm')
    second = Quantity(np.empty(shape), 'm') if operation is np.modf else np.empty(shape, dtype=int)
    a, b = operation(q, out=(first, second))
    assert a is first and b is second
    np.testing.assert_equal(
        a.value, np.modf(q.value)[0] * 100 if operation is np.modf else np.frexp(q.value)[0] * 100
    )


@pytest.mark.parametrize('operation', [np.modf, np.frexp])
@pytest.mark.parametrize('dtype', [np.int16, np.float32, np.float64, np.longdouble])
def test_numpy_multi_output_dtype(operation: np.ufunc, dtype: DTypeLike) -> None:
    q = Quantity(np.array([1, -2], dtype=dtype), 'm')
    a, b = operation(q, order='F', subok=False)
    expected = operation(q.value, order='F', subok=False)
    assert a.value.dtype == expected[0].dtype
    assert (b.value if isinstance(b, Quantity) else b).dtype == expected[1].dtype


@pytest.mark.parametrize('operation', [np.modf, np.frexp])
def test_numpy_multi_output_casting(operation: np.ufunc) -> None:
    q = Quantity(np.array([1.25, -2.5]), 'm')
    first = Quantity(np.full(2, -123.0, dtype=np.float32), 'm')
    second = Quantity(np.full(2, -456.0), 'm') if operation is np.modf else np.full(2, -456)
    with pytest.raises(TypeError):
        operation(q, out=(first, second), casting='safe')
    np.testing.assert_equal(first.value, -123)
    np.testing.assert_equal(second.value if isinstance(second, Quantity) else second, -456)
    assert operation(q, out=(first, second), casting='same_kind')[0] is first
    assert first.value.dtype == np.dtype(np.float32)


def test_numpy_multi_output_invalid_conversion() -> None:
    q = Quantity(np.array([1e300]), 'm')
    first = Quantity(np.full(1, -123.0), 'm')
    second = Quantity(np.full(1, -456.0), 'ym')
    with np.errstate(over='raise'):
        with pytest.raises(FloatingPointError):
            np.modf(q, out=(first, second))
        np.testing.assert_equal(first.value, -123)
        np.testing.assert_equal(second.value, -456)
        # Masked coordinates are skipped both by the kernel and unit conversion.
        np.modf(q, out=(first, second), where=False)
    np.testing.assert_equal(first.value, -123)
    np.testing.assert_equal(second.value, -456)


def test_numpy_multi_output_first_mask() -> None:
    # Writing the mantissa changes the mask bytes before the exponent is copied.
    storage = np.ones(2, dtype=np.uint64)
    first = Quantity(storage.view(np.float64), 'm')
    mask = storage.view(np.bool_).reshape(2, 8)[:, 0]
    assert np.all(mask)
    exponent = np.full(2, -123)
    np.frexp(Quantity(np.array([1.25, -2.5]), 'm'), out=(first, exponent), where=mask)
    np.testing.assert_equal(first.value, [0.625, -0.625])
    np.testing.assert_equal(exponent, [1, 2])
