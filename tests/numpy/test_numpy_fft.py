"""Discrete transforms preserve units; sampling intervals set frequency units."""

import warnings
from decimal import Decimal
from fractions import Fraction
from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity, Unit
from unitx.exceptions import DimensionError
from unitx.factor import Factor, pi
from unitx.si import si

np = pytest.importorskip('numpy')

_ONE_DIMENSIONAL = ('fft', 'ifft', 'rfft', 'irfft', 'hfft', 'ihfft')
_MULTIDIMENSIONAL = ('fft2', 'ifft2', 'fftn', 'ifftn', 'rfft2', 'irfft2', 'rfftn', 'irfftn')
_TRANSFORMS = _ONE_DIMENSIONAL + _MULTIDIMENSIONAL
_COMPLEX_TRANSFORMS = tuple(
    name for name in _TRANSFORMS if name not in ('rfft', 'rfft2', 'rfftn', 'ihfft')
)


@pytest.mark.parametrize('name', _TRANSFORMS)
@pytest.mark.parametrize('norm', [None, 'backward', 'forward', 'ortho'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_numpy_fft_transform(name: str, norm: Any, dtype: Any) -> None:
    values = np.arange(30, dtype=dtype).reshape(2, 3, 5)[:, ::-1]
    q = Quantity(values, Factor(2) * pi * si.cm)
    q.value.flags.writeable = False
    operation = getattr(np.fft, name)
    result = operation(a=q, norm=norm)
    expected = operation(values, norm=norm)
    assert result.unit is q.unit and result.value.dtype == expected.dtype
    np.testing.assert_allclose(result.value, expected)


@pytest.mark.parametrize('name', _COMPLEX_TRANSFORMS)
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_numpy_fft_complex(name: str, dtype: Any) -> None:
    values = (np.arange(30).reshape(2, 3, 5) * (1 + 2j)).astype(dtype)
    q = Quantity(values, 'V')
    operation = getattr(np.fft, name)
    result = operation(q, norm='ortho')
    expected = operation(values, norm='ortho')
    assert result.unit is q.unit and result.value.dtype == expected.dtype
    np.testing.assert_allclose(result.value, expected)


@pytest.mark.parametrize('name', _TRANSFORMS)
def test_numpy_fft_options(name: str) -> None:
    q = Quantity(np.arange(30.0).reshape(2, 3, 5), 'm')
    options = dict(n=7, axis=1) if name in _ONE_DIMENSIONAL else dict(s=(4, 7), axes=(0, 2))
    operation = getattr(np.fft, name)
    result = operation(q, norm='forward', **options)
    assert result.unit is q.unit
    np.testing.assert_allclose(result.value, operation(q.value, norm='forward', **options))


@pytest.mark.parametrize('name', _TRANSFORMS)
@pytest.mark.parametrize('positional', [False, True])
def test_numpy_fft_output(name: str, positional: bool) -> None:
    q = Quantity(np.arange(30.0).reshape(2, 3, 5), Factor(2) * si.m)
    operation = getattr(np.fft, name)
    options = (None, -1, 'ortho') if name in _ONE_DIMENSIONAL else (None, (0, 2), 'ortho')
    expected = operation(q.value, *options)
    target = Quantity(np.full_like(expected, -7), 'cm')
    buffer = target.value
    result = operation(q, *options, target) if positional else operation(q, *options, out=target)
    assert result is target and result.value is buffer and result.unit == si.cm
    np.testing.assert_allclose(result.value, expected * 200)


@pytest.mark.parametrize('name', ['fft', 'ifft', 'rfft', 'irfft', 'hfft', 'ihfft'])
def test_numpy_fft_round_trip(name: str) -> None:
    n = 7
    values = np.arange(n, dtype=np.float64)
    inverse = {
        'fft': 'ifft',
        'ifft': 'fft',
        'rfft': 'irfft',
        'irfft': 'rfft',
        'hfft': 'ihfft',
        'ihfft': 'hfft',
    }
    if name in ('irfft', 'hfft'):
        values = values[: n // 2 + 1]
    q = Quantity(values, Factor(2) * pi * si.m)
    transformed = getattr(np.fft, name)(q, n=n)
    recovered = getattr(np.fft, inverse[name])(transformed, n=n)
    assert recovered.unit is q.unit
    np.testing.assert_allclose(recovered.value, q.value, atol=1e-14)


def test_numpy_fft_output_alias() -> None:
    storage = np.arange(6, dtype=np.complex128)
    q = Quantity(storage[:4], 'm')
    target = Quantity(storage[2:], 'cm')
    expected = np.fft.fft(q.value.copy()) * 100
    assert np.fft.fft(q, out=target) is target
    np.testing.assert_allclose(target.value, expected)
    expected = np.fft.ifft(target.value.copy())
    assert np.fft.ifft(target, out=target) is target
    np.testing.assert_allclose(target.value, expected)


def test_numpy_fft_output_dtype() -> None:
    q = Quantity(np.array([1.123456789, 2.345678912, 3.456789123, 4.567891234]), 'm')
    target = Quantity(np.empty(4, dtype=np.complex64), 'cm')
    assert np.fft.fft(q, out=target) is target
    expected = (np.fft.fft(q.value) * 100).astype(np.complex64)
    assert target.value.dtype == np.complex64
    np.testing.assert_array_equal(target.value, expected)


@pytest.mark.parametrize('name', _TRANSFORMS)
@pytest.mark.parametrize('constructor,unit', [(AffineQuantity, 'degC'), (LogQuantity, 'dBm')])
def test_numpy_fft_invalid_point(name: str, constructor: Any, unit: str) -> None:
    point = constructor(np.ones((2, 3)), unit)
    with pytest.raises(TypeError, match='linear'):
        getattr(np.fft, name)(point)


@pytest.mark.parametrize('invalid', ['unit', 'shape', 'dtype', 'readonly', 'bare', 'point'])
def test_numpy_fft_invalid_output(invalid: str) -> None:
    q = Quantity(np.arange(4.0), 'm')
    values = np.full(
        (1, 4) if invalid == 'shape' else (4,), 7, dtype=float if invalid == 'dtype' else complex
    )
    if invalid == 'readonly':
        values.flags.writeable = False
    target = Quantity(values, 's' if invalid == 'unit' else 'cm')
    if invalid == 'bare':
        target = values
    elif invalid == 'point':
        target = AffineQuantity(values.real.copy(), 'K')
    buffer = target if type(target) is np.ndarray else target.value
    saved = buffer.copy()
    error = (
        DimensionError
        if invalid in ('unit', 'point')
        else ValueError
        if invalid in ('shape', 'readonly')
        else TypeError
    )
    with pytest.raises(error):
        np.fft.fft(q, out=target)
    np.testing.assert_array_equal(buffer, saved)


def test_numpy_fft_invalid_options() -> None:
    q = Quantity(np.arange(4.0), 'm')
    target = Quantity(np.full(4, 7 + 0j), 'm')
    with pytest.raises(ValueError):
        np.fft.fft(q, n=0, out=target)
    np.testing.assert_array_equal(target.value, 7)


def test_numpy_fft_invalid_duplicate_output() -> None:
    q = Quantity(np.arange(4.0), 'm')
    target = Quantity(np.full(4, 7 + 0j), 'm')
    with pytest.raises(TypeError):
        np.fft.fft(q, None, -1, None, target, out=target)
    np.testing.assert_array_equal(target.value, 7)


def test_numpy_fft_invalid_output_conversion() -> None:
    q = Quantity(np.full(4, 1e300), 'm')
    target = Quantity(np.full(4, 7 + 0j), Unit('m', scale=Factor(Fraction(1, 10**20))))
    with np.errstate(over='raise', invalid='raise'):
        with pytest.raises(FloatingPointError):
            np.fft.fft(q, out=target)
    np.testing.assert_array_equal(target.value, 7)


def test_numpy_fft_deprecation() -> None:
    q = Quantity(np.arange(6.0).reshape(2, 3), 'm')
    with warnings.catch_warnings(record=True) as native:
        warnings.simplefilter('always', DeprecationWarning)
        expected = np.fft.fftn(q.value, s=(2,))
    if not native:
        pytest.skip('This NumPy version does not warn about s without axes')
    with pytest.warns(DeprecationWarning) as caught:
        result = np.fft.fftn(q, s=(2,))
    assert caught[0].filename == __file__
    np.testing.assert_allclose(result.value, expected)
    target = Quantity(np.full(expected.shape, 7 + 0j), 'm')
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        with pytest.raises(DeprecationWarning):
            np.fft.fftn(q, s=(2,), out=target)
    np.testing.assert_array_equal(target.value, 7)


@pytest.mark.parametrize('axes', [None, 0, -1, (0, 2)])
@pytest.mark.parametrize('length', [4, 5])
@pytest.mark.parametrize(
    'constructor,unit', [(Quantity, 'm'), (AffineQuantity, 'degC'), (LogQuantity, 'dBm')]
)
def test_numpy_fft_shift(axes: Any, length: int, constructor: Any, unit: str) -> None:
    q = constructor(np.arange(6 * length).reshape(2, 3, length), unit)
    q.value.flags.writeable = False
    for operation, inverse in (
        (np.fft.fftshift, np.fft.ifftshift),
        (np.fft.ifftshift, np.fft.fftshift),
    ):
        result = operation(q, axes=axes)
        assert type(result) is type(q) and result.unit is q.unit
        np.testing.assert_array_equal(result.value, operation(q.value, axes=axes))
        np.testing.assert_array_equal(inverse(result, axes=axes).value, q.value)


@pytest.mark.parametrize('name', ['fftfreq', 'rfftfreq'])
@pytest.mark.parametrize('n', [5, 6])
@pytest.mark.parametrize(
    'value', [2, 2.5, np.float32(2.5), np.array(2.5), Fraction(5, 2), Decimal('2.5')]
)
@pytest.mark.parametrize('unit', [si.ms, si.cm, Factor(2) * pi * si.s, Unit('cm/m')])
def test_numpy_fft_frequency(name: str, n: int, value: Any, unit: Unit) -> None:
    spacing = Quantity(value, unit)
    operation = getattr(np.fft, name)
    result = operation(n, d=spacing, device='cpu')
    assert result.unit == spacing.unit**-1
    np.testing.assert_allclose(result.value, operation(n, d=float(value)), rtol=2e-6)


@pytest.mark.parametrize('name', ['fftfreq', 'rfftfreq'])
def test_numpy_fft_frequency_unit_invariance(name: str) -> None:
    operation = getattr(np.fft, name)
    result = operation(6, d=2 * si.ms)
    equivalent = operation(6, d=0.002 * si.s)
    np.testing.assert_allclose(result.value_in(si.Hz), equivalent.value_in(si.Hz))
    assert type(operation(6, d=2)) is np.ndarray


@pytest.mark.parametrize('name', ['fftfreq', 'rfftfreq'])
@pytest.mark.parametrize('point', [AffineQuantity(2.0, 'degC'), LogQuantity(2.0, 'dBm')])
def test_numpy_fft_frequency_invalid_point(name: str, point: Any) -> None:
    with pytest.raises(TypeError):
        getattr(np.fft, name)(6, d=point)


def test_numpy_fft_sampling_workflow() -> None:
    spacing = 0.125 * si.s
    time = np.arange(8) * spacing
    signal = 2 * si.m * np.cos(2 * np.pi * (2 * si.Hz) * time)
    spectrum = np.fft.rfft(signal)
    frequency = np.fft.rfftfreq(signal.size, d=spacing)
    assert spectrum.unit == si.m
    assert frequency[np.argmax(np.abs(spectrum))] == 2 * si.Hz
    recovered = np.fft.irfft(spectrum, n=signal.size)
    np.testing.assert_allclose(recovered.value_in(si.m), signal.value_in(si.m), atol=1e-14)
    integral = spectrum * spacing
    assert integral.unit == si.m * si.s
    energy = np.sum(np.abs(np.fft.fft(signal)) ** 2) / signal.size
    np.testing.assert_allclose(energy.value_in(si.m**2), np.sum((signal**2).value_in(si.m**2)))
