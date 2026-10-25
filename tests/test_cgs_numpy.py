"""Native NumPy promotion and explicit electromagnetic conversion."""

import pytest

np = pytest.importorskip('numpy')

from unitx import Quantity, Unit  # noqa: E402
from unitx.cgs import cgs, emu, esu, gaussian, hl  # noqa: E402
from unitx.exceptions import DimensionError  # noqa: E402
from unitx.si import si  # noqa: E402

_Catalogue = type[esu] | type[emu] | type[hl]


@pytest.mark.parametrize(
    ('catalogue', 'source', 'target'),
    [
        (esu, si.C, esu.statC),
        (esu, si.T, esu.magnetic_flux_density),
        (emu, si.C, emu.abC),
        (hl, si.C, hl.charge),
        (hl, si.V / si.m, hl.electric_field),
    ],
)
@pytest.mark.parametrize('dtype', ['int32', 'float32', 'float64', 'complex64', 'complex128'])
def test_cgs_system_numpy_scaling(
    catalogue: _Catalogue, source: Unit, target: Unit, dtype: str
) -> None:
    values = np.array([0, 1, -2], dtype=dtype)
    values.flags.writeable = False
    saved = values.copy()
    result = catalogue.convert(Quantity(values, source), target)
    multiplier = catalogue.convert(1 * source, target).value
    expected = values * multiplier
    assert result.value.dtype == expected.dtype
    np.testing.assert_array_equal(result.value, expected)
    np.testing.assert_array_equal(values, saved)
    assert not np.shares_memory(result.value, values)
    np.testing.assert_allclose(catalogue.convert(result, source).value, saved, rtol=2e-7)


@pytest.mark.parametrize('dtype', ['int32', 'float32', 'float64', 'complex64', 'complex128'])
def test_gaussian_numpy_scaling(dtype: str) -> None:
    values = np.array([0, 1, -2], dtype=dtype)
    values.flags.writeable = False
    original = values.copy()
    result = gaussian.convert(Quantity(values, si.C), gaussian.statC)
    multiplier = gaussian.convert(1 * si.C, gaussian.statC).value
    expected = values * multiplier
    assert result.value.dtype == expected.dtype
    np.testing.assert_array_equal(result.value, expected)
    np.testing.assert_array_equal(values, original)
    assert not np.shares_memory(values, result.value)
    restored = gaussian.convert(result, si.C)
    np.testing.assert_allclose(restored.value, original, rtol=2e-7)


@pytest.mark.parametrize('value', [np.float32(2), np.int64(3), np.complex64(2j)])
def test_gaussian_numpy_scalar(value: object) -> None:
    result = gaussian.convert(Quantity(value, gaussian.statC), si.C)
    assert isinstance(result.value, np.number)
    assert result.value.dtype == (value * gaussian.convert(1 * gaussian.statC, si.C).value).dtype


def test_gaussian_numpy_broadcasting() -> None:
    q = gaussian.convert(Quantity(np.array([[1.0], [2.0]]), si.C), gaussian.statC)
    e = gaussian.convert(
        Quantity(np.array([1.0, 2.0, 3.0]), si.V / si.m),
        gaussian.statV / cgs.cm,
        kind='electric_field',
    )
    force = q * e
    np.testing.assert_allclose(force.value_in(si.N), [[1, 2, 3], [2, 4, 6]])


def test_gaussian_numpy_write_boundary() -> None:
    target = Quantity(np.array([1.0, 2.0]), si.C)
    source = Quantity(np.array([3.0, 4.0]), gaussian.statC)
    saved = target.value.copy()
    with pytest.raises(DimensionError):
        target[:] = source
    np.testing.assert_array_equal(target.value, saved)
    with pytest.raises(DimensionError):
        np.add(target, source, out=target)
    np.testing.assert_array_equal(target.value, saved)
    with pytest.raises(DimensionError):
        target += source
    np.testing.assert_array_equal(target.value, saved)
    target[:] = gaussian.convert(source, si.C)
    np.testing.assert_allclose(target.value, gaussian.convert(source, si.C).value)


@pytest.mark.parametrize('dtype', ['object', 'bool', 'str'])
def test_gaussian_numpy_invalid(dtype: str) -> None:
    with pytest.raises(TypeError, match='plain numeric'):
        gaussian.convert(Quantity(np.array([1], dtype=dtype), si.C), gaussian.statC)
