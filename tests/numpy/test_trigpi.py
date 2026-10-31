"""Array half-turn evaluation keeps precision, exact nodes and NumPy failures."""

from typing import Any

import pytest

from unitx import Quantity, math
from unitx.si import si

np = pytest.importorskip('numpy')


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
@pytest.mark.parametrize('shape', [(), (7,)])
def test_math_trigpi_numpy_nodes(dtype: Any, shape: tuple[int, ...]) -> None:
    values = np.full(shape, -0.5, dtype=dtype) if not shape else np.arange(-3, 4, dtype=dtype) / 2
    for function, expected in [
        (math.sinpi, [1, 0, -1, 0, 1, 0, -1]),
        (math.cospi, [0, -1, 0, 1, 0, -1, 0]),
    ]:
        result = function(Quantity(values, si.one))
        assert result.value.dtype == values.dtype
        assert result.unit == si.one
        np.testing.assert_array_equal(
            result.value, expected if shape else (-1 if function is math.sinpi else 0)
        )
        assert not np.shares_memory(result.value, values)


def test_math_trigpi_numpy_large_integer() -> None:
    values = np.array([2**63 + 1, 2**63 + 2], dtype=np.uint64)
    np.testing.assert_array_equal(math.sinpi(values).value, [0, 0])
    np.testing.assert_array_equal(math.cospi(values).value, [-1, 1])


def test_math_trigpi_numpy_neighbors() -> None:
    values = np.array(
        [np.nextafter(1.0, 0.0), np.nextafter(1.0, 2.0), -np.nextafter(0.0, 1.0), -0.0]
    )
    for function in (math.sinpi, math.cospi):
        np.testing.assert_allclose(
            function(values).value, [function(float(x)).value for x in values], rtol=2e-15, atol=0
        )
    assert np.signbit(math.sinpi(values).value[-1])
    np.testing.assert_array_equal(
        math.sinpi(Quantity(np.array([50.0, 150.0]), si.percent)).value, [1, -1]
    )
    with np.errstate(invalid='raise'):
        with pytest.raises(FloatingPointError):
            math.sinpi(np.array([np.inf]))
    assert math.cospi(np.empty((0, 2))).value.shape == (0, 2)
    assert isinstance(math.sinpi(np.float32(0.5)).value, np.float32)


def test_math_trigpi_numpy_warning() -> None:
    with np.errstate(invalid='warn'):
        with pytest.warns(RuntimeWarning) as caught:
            result = math.cospi(np.array([np.inf]))
    assert np.isnan(result.value[0])
    assert len(caught) == 1 and caught[0].filename == __file__


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, np.longdouble])
@pytest.mark.parametrize('scalar', [False, True])
def test_math_trigpi_numpy_precision(dtype: Any, scalar: bool) -> None:
    # At odd quarter turns both functions have magnitude sqrt(1/2), so this
    # reference does not inherit a float64 approximation of pi.
    values = np.array([0.25, 0.75, 1.25, 1.75], dtype=dtype)
    if scalar:
        values = values[0]
    root = np.sqrt(dtype(0.5))
    for function, signs in [
        (math.sinpi, [1, 1, -1, -1]),
        (math.cospi, [1, -1, -1, 1]),
    ]:
        expected = root if scalar else root * np.array(signs, dtype=dtype)
        result = function(values).value
        assert result.dtype == values.dtype
        np.testing.assert_allclose(result, expected, rtol=2 * np.finfo(dtype).eps, atol=0)
