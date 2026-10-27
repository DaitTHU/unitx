"""Runtime evidence for the dtype-specific numerical annotations."""

from decimal import Decimal
from typing import Any

import pytest

from unitx import Quantity, math
from unitx.si import si

np = pytest.importorskip('numpy')


@pytest.mark.parametrize(
    'dtype',
    ['float16', 'float32', 'float64', 'longdouble', 'complex64', 'complex128', 'clongdouble'],
)
@pytest.mark.parametrize('shape', [(), (3,)])
def test_quantity_inexact_dtype(dtype: str, shape: tuple[int, ...]) -> None:
    values = np.full(shape, 2, dtype=dtype)
    q = Quantity(values, si.m)
    results = (
        q.to(si.cm),
        q.to_base(),
        q.simplify(),
        q + q,
        q - q,
        q * q,
        q / q,
        q**2,
        q**0.5,
        +q,
        -q,
        q * Decimal('1.5'),
        Decimal('1.5') / q,
        q.sum(),
        q.mean(),
    )
    for result in results:
        assert result.value.dtype == values.dtype
        assert isinstance(result.value, (np.ndarray, np.number))
    assert q.value_in(si.cm).dtype == values.dtype
    if values.dtype.kind != 'c':
        assert q.compact().value.dtype == values.dtype


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'longdouble'])
@pytest.mark.parametrize('shape', [(), (3,)])
def test_math_floating_dtype(dtype: str, shape: tuple[int, ...]) -> None:
    q = Quantity(np.full(shape, 0.5, dtype=dtype), si.one)
    for function in (
        math.sqrt,
        math.cbrt,
        math.sin,
        math.cos,
        math.tan,
        math.asin,
        math.acos,
        math.atan,
        math.exp,
        math.expm1,
        math.log,
        math.log1p,
        math.log2,
        math.log10,
    ):
        assert function(q).value.dtype == q.value.dtype


@pytest.mark.parametrize('right', [np.float64(2), np.array([2.0], dtype=np.float64)])
def test_quantity_mixed_dtype(right: Any) -> None:
    q = Quantity(np.ones(3, dtype=np.float32), si.m)
    assert (q * right).value.dtype == np.float64
    assert (q + Quantity(right, si.m)).value.dtype == np.float64
