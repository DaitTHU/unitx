"""Check the extension's rational networks and complex AC behavior."""

from __future__ import annotations

import cmath
from decimal import Decimal
from fractions import Fraction

import pytest

from examples.impedance import Impedance, main
from unitx import Quantity
from unitx._numerics import Magnitude
from unitx.exceptions import DimensionError
from unitx.si import si


def test_impedance_network(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    assert 'RMS: I =' in capsys.readouterr().out
    first, second, third = (
        Impedance(Fraction(100)),
        Impedance(Fraction(1, 5), si.kohm),
        Impedance(300),
    )
    result = (first + second).to(si.kohm) // third
    assert type(result) is Impedance
    assert result == 150 * si.ohm
    assert result.unit == si.kohm
    assert first // second // third == Fraction(600, 11) * si.ohm


def test_impedance_complex_network() -> None:
    first = Impedance(100 + 100j)
    second = Impedance(0.1 - 0.1j, si.kohm)
    series, parallel = first + second, first // second
    assert type(series) is type(parallel) is Impedance
    assert series == 200 * si.ohm
    assert parallel == 100 * si.ohm
    assert first.to(si.kohm).value == 0.1 + 0.1j
    voltage = 12 * si.V
    current = voltage / first
    assert type(current) is Quantity
    assert cmath.isclose(current.value_in(si.A), 0.06 - 0.06j)


@pytest.mark.parametrize('value', [100, 100.0, Fraction(100), 10**400, 100j, -100j, -100])
def test_impedance_magnitude(value: int | float | Fraction | complex) -> None:
    assert Impedance(value).value is value


@pytest.mark.parametrize('left, right', [(0, 0), (0, 100j), (100j, 0)])
def test_impedance_short_circuit(left: int | complex, right: int | complex) -> None:
    assert Impedance(left) // Impedance(right) == 0 * si.ohm


def test_impedance_parallel_large_float() -> None:
    assert (Impedance(1e300) // Impedance(1e300)).value == 5e299


def test_impedance_parallel_resonance_invalid() -> None:
    with pytest.raises(ZeroDivisionError, match='equivalent impedance is infinite'):
        Impedance(100j) // Impedance(-100j)


def test_impedance_physical_results() -> None:
    impedance = Impedance(Fraction(100))
    voltage = impedance * (2 * si.A)
    admittance = 1 / impedance
    assert type(voltage) is type(admittance) is Quantity
    assert voltage == 200 * si.V
    assert admittance == Fraction(1, 100) * si.S


@pytest.mark.parametrize('value', [True, '100', Decimal(100), [100]])
def test_impedance_magnitude_invalid(value: bool | str | Decimal | list[int]) -> None:
    with pytest.raises(TypeError, match='int, float, Fraction or complex scalar'):
        Impedance(value)


@pytest.mark.parametrize('value', [float('inf'), float('nan'), complex(1, float('inf'))])
def test_impedance_value_invalid(value: float | complex) -> None:
    with pytest.raises(ValueError, match='must be finite'):
        Impedance(value)


def test_impedance_unit_invalid() -> None:
    with pytest.raises(DimensionError, match='impedance unit'):
        Impedance(100, si.V)


@pytest.mark.parametrize('other', [100, 100 * si.ohm])
def test_impedance_parallel_invalid(other: int | Quantity[Magnitude]) -> None:
    with pytest.raises(TypeError):
        Impedance(100) // other
