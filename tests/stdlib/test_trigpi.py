"""Half-turn arguments retain their periods before numerical evaluation."""

import math as stdmath
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from unitx import AffineQuantity, math
from unitx.exceptions import DimensionError
from unitx.si import si


@pytest.mark.parametrize('n', [-(10**100) - 1, -3, -1, 0, 1, 4, 10**100 + 1])
def test_math_trigpi_integer(n: int) -> None:
    assert math.sinpi(n).value == 0
    assert math.cospi(n).value == (-1 if n % 2 else 1)
    half = Fraction(2 * n + 1, 2)
    assert math.sinpi(half).value == (-1 if n % 2 else 1)
    assert math.cospi(half).value == 0


def _sine_reference(x: float) -> float:
    # Independent high-precision series, evaluated before conversion to float.
    with localcontext() as context:
        context.prec = 100
        pi = Decimal(
            '3.141592653589793238462643383279502884197169399375105820974944592307816406286'
        )
        z = Decimal.from_float(x) * pi
        term = total = z
        for n in range(1, 150):
            term *= -z * z / ((2 * n) * (2 * n + 1))
            total += term
        return float(total)


@pytest.mark.parametrize('x', [stdmath.nextafter(1, 0), stdmath.nextafter(1, 2), -0.125, 0.3])
def test_math_trigpi_neighbors(x: float) -> None:
    assert math.sinpi(x).value == pytest.approx(_sine_reference(x), rel=3e-15, abs=0)
    assert math.cospi(x).value == pytest.approx(_sine_reference(x + 0.5), rel=3e-15, abs=2e-16)


def test_math_trigpi_scaled() -> None:
    assert math.sinpi(50 * si.percent).value == 1
    assert math.cospi(100 * si.percent).value == -1
    assert math.sinpi(Fraction(1, 6)).value == pytest.approx(0.5)
    with localcontext() as context:
        context.prec = 2
        assert math.sinpi(Decimal('1000000000000000000000000000000.5')).value == 1
        assert math.cospi(Decimal('1E1000')).value == 1
    assert stdmath.copysign(1, math.sinpi(-0.0).value) == -1


@pytest.mark.parametrize('function', [math.sinpi, math.cospi])
def test_math_trigpi_invalid(function: object) -> None:
    with pytest.raises(TypeError):
        function(AffineQuantity(2, 'K'))
    with pytest.raises(DimensionError):
        function(1 * si.m)
    with pytest.raises(DimensionError):
        function(1 * si.dB)
    with pytest.raises(ValueError):
        function(float('inf'))
    assert stdmath.isnan(function(float('nan')).value)
