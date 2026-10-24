from __future__ import annotations

import math as stdmath
import subprocess
import sys
from decimal import Decimal
from fractions import Fraction

import pytest

from unitx import AffineQuantity, Quantity, math, pi
from unitx.exceptions import DimensionError
from unitx.factor import Factor
from unitx.unit import Unit


def test_math_root_units() -> None:
    result = math.sqrt(Quantity(9, 'cm2'))
    assert result.unit == Unit('cm')
    assert result.value == 3.0
    assert type(result.value) is float
    assert math.cbrt(Quantity(-8, 'm3')) == Quantity(-2, 'm')
    assert math.sqrt(Quantity(4, 'm')).unit ** 2 == Unit('m')
    assert math.cbrt(Quantity(8, 'm')).unit ** 3 == Unit('m')
    assert math.sqrt(Quantity(4, Unit('m2', scale=Factor(9)))).value_in('m') == 6


@pytest.mark.parametrize(
    'name, value',
    [
        ('sin', 0.3),
        ('cos', 0.3),
        ('tan', 0.3),
        ('exp', 0.3),
        ('expm1', 1e-12),
        ('log', 2),
        ('log1p', 1e-12),
        ('log2', 2),
        ('log10', 2),
    ],
)
def test_math_dimensionless_scaling(name: str, value: float | int) -> None:
    function = getattr(math, name)
    for argument in [value, Quantity(value / 1000, 'km/m')]:
        result = function(argument)
        assert result.unit == Unit()
        assert result.value == pytest.approx(getattr(stdmath, name)(value))
    with pytest.raises(DimensionError):
        function(Quantity(1, 'm'))


def test_math_angles() -> None:
    assert math.sin(pi / 2).value == pytest.approx(1)
    assert math.cos(pi).value == pytest.approx(-1)
    assert math.sin(Quantity(90, 'deg')).value == pytest.approx(1)
    assert math.cos(Quantity(180, 'deg')).value == pytest.approx(-1)
    for name in ('asin', 'acos', 'atan'):
        result = getattr(math, name)(Quantity(0.0005, 'km/m'))
        assert result.unit == Unit('rad')
        assert result.value == pytest.approx(getattr(stdmath, name)(0.5))


def test_math_scalar_domains() -> None:
    assert math.sqrt(Decimal('2')).value == stdmath.sqrt(2)
    assert math.log(Fraction(1, 3)).value == stdmath.log(Fraction(1, 3))
    assert math.log(10**1000).value == stdmath.log(10**1000)
    assert stdmath.isnan(math.sqrt(float('nan')).value)
    assert math.exp(float('-inf')).value == 0
    for function, value in [(math.sqrt, -1), (math.log, 0), (math.asin, 2)]:
        with pytest.raises(ValueError):
            function(value)


def test_math_hypot_alignment() -> None:
    result = math.hypot(Quantity(3, 'm'), Quantity(400, 'cm'))
    assert result == Quantity(5, 'm')
    assert result.unit == Unit('m')
    assert math.hypot() == Quantity(0.0, '')
    assert math.hypot(Quantity(-3, 'm')) == Quantity(3, 'm')
    assert math.hypot(2, Quantity(3, 'km/m')).value == stdmath.hypot(2, 3000)
    with pytest.raises(DimensionError):
        math.hypot(Quantity(1, 'm'), 1)


def test_math_atan2_alignment() -> None:
    result = math.atan2(Quantity(100, 'cm'), Quantity(1, 'm'))
    assert result.unit == Unit('rad')
    assert result.value == pytest.approx(stdmath.pi / 4)
    assert math.atan2(-0.0, -1).value == -stdmath.pi
    with pytest.raises(DimensionError):
        math.atan2(Quantity(1, 'm'), Quantity(1, 's'))


@pytest.mark.parametrize('value', [True, 1j, [1], (1,), object(), AffineQuantity(1, 'K')])
def test_math_invalid_input(value: object) -> None:
    with pytest.raises(TypeError):
        math.sin(value)


def test_math_invalid_logarithmic() -> None:
    with pytest.raises(DimensionError, match='physical ratios'):
        math.exp(Quantity(1, 'dB'))


def test_math_time_optional_imports() -> None:
    # -S removes all installed optional backends, including NumPy and SymPy.
    code = """
import sys
sys.path.insert(0, 'src')
import unitx
assert unitx.math.sqrt(4).value == 2
assert isinstance(unitx.time.time().value, int)
assert 'numpy' not in sys.modules
assert 'sympy' not in sys.modules
"""
    subprocess.run([sys.executable, '-S', '-c', code], check=True)
