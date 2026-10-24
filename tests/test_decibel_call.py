"""The dB call is a zero-reference convenience, with explicit identity boundaries."""

from __future__ import annotations

import copy
import pickle
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import NoReturn

import pytest

from unitx import LogQuantity, LogUnit, Quantity, Unit
from unitx._numerics import Magnitude
from unitx.exceptions import DimensionError
from unitx.factor import Factor
from unitx.logunit import LogKind
from unitx.serialization import from_dict, to_dict
from unitx.si import si
from unitx.unitregistry import default_registry


@pytest.mark.parametrize('coordinate', [si.dBm, si.dBW, si.dBV])
def test_unit_call_named_reference(coordinate: LogUnit) -> None:
    assert si.dB(coordinate.reference.unit) is coordinate
    assert si.dB(coordinate.reference, kind=coordinate.kind) is coordinate


@pytest.mark.parametrize(
    ('unit', 'kind'), [(si.kW, 'power'), (si.uV, 'amplitude'), (si.mV, 'amplitude')]
)
def test_unit_call_prefixed_reference(unit: Unit, kind: LogKind) -> None:
    coordinate = si.dB(unit)
    assert coordinate.kind == kind
    assert coordinate.reference == 1 * unit
    assert coordinate.reference.unit == unit
    assert (0 @ coordinate).to_quantity(unit) == 1 * unit
    assert (20 @ coordinate).to_quantity(unit) == (100 if kind == 'power' else 10) * unit


def test_unit_call_millivolt_reference() -> None:
    coordinate = si.dB(si.mV)
    assert (0 @ coordinate).to(si.dBV).value == -60
    assert coordinate.dimension != si.dBm.dimension


def test_unit_call_scaled_reference() -> None:
    unit = Unit(si.V, scale=Factor(Fraction(3, 2)))
    coordinate = si.dB(unit)
    assert coordinate.kind == 'amplitude'
    assert coordinate.reference.unit == unit
    assert (0 @ coordinate).to_quantity(si.V) == Fraction(3, 2) * si.V


def test_unit_call_explicit_kind() -> None:
    coordinate = si.dB(si.V, kind='power')
    assert coordinate.kind == 'power'
    assert (20 @ coordinate).to_quantity(si.V) == 100 * si.V
    assert coordinate != si.dBV


def test_unit_call_quantity_reference() -> None:
    reference = 20 * si.uPa
    coordinate = si.dB(reference, kind='amplitude')
    assert coordinate.reference == reference
    assert (20 @ coordinate).to_quantity(si.uPa) == 200 * si.uPa


def test_unit_call_reference_unit_identity() -> None:
    coordinate = si.dB(1000 * si.mW)
    assert coordinate.reference.unit == si.mW
    assert coordinate.reference.value == 1000
    assert coordinate != si.dBW
    assert (0 @ coordinate).to(si.dBW).value == 0


def test_unit_call_decimal_reference() -> None:
    with localcontext() as context:
        context.prec = 2
        coordinate = si.dB(Decimal('1.2345') * si.mV)
        assert coordinate.reference.value == Fraction(2469, 2000)
        assert (Decimal(0) @ coordinate).to_quantity(si.mV).value == Decimal('1.2')


@pytest.mark.parametrize('unit', [si.Pa, si.A, si.W / si.A, si.V.define('privateVoltage')])
def test_unit_call_kind_invalid(unit: Unit) -> None:
    with pytest.raises(TypeError, match='Specify kind'):
        si.dB(unit)
    assert si.dB(unit, kind='amplitude').reference.unit == unit


@pytest.mark.parametrize(
    'unit',
    [
        si.m,
        si.B,
        si.Np,
        Factor(2) * si.dB,
        si.dB / si.m,
        Factor(Fraction(1, 10)) * si.B,
        si.dB.define('dB'),
    ],
)
def test_unit_call_unit_invalid(unit: Unit) -> None:
    with pytest.raises(TypeError, match='Only dB supports'):
        unit(si.V)


@pytest.mark.parametrize('reference', ['V', 1, None, 0 @ si.dBm])
def test_unit_call_reference_invalid(reference: str | int | None | LogQuantity[Magnitude]) -> None:
    with pytest.raises(TypeError, match='reference must be a Unit or physical Quantity'):
        si.dB(reference)


def test_unit_call_convention_invalid() -> None:
    with pytest.raises(ValueError, match='kind'):
        si.dB(si.V, kind='invalid')


@pytest.mark.parametrize('value', [0, -1, float('inf'), float('nan')])
def test_unit_call_value_invalid(value: int | float) -> None:
    with pytest.raises(ValueError):
        si.dB(value * si.V)


def test_unit_call_physical_reference_invalid() -> None:
    with pytest.raises(DimensionError, match='physical quantity'):
        si.dB(1 * si.dB, kind='amplitude')


def test_unit_call_reconstruction() -> None:
    for decibel in (
        Unit('decibel'),
        Unit(si.dB),
        pickle.loads(pickle.dumps(si.dB)),
        copy.copy(si.dB),
        from_dict(to_dict(si.dB)),
    ):
        assert decibel(si.V) is si.dBV
    coordinate = si.dB(si.mV)
    for restored in (
        pickle.loads(pickle.dumps(coordinate)),
        from_dict(to_dict(coordinate)),
        eval(
            repr(coordinate), dict(LogUnit=LogUnit, Quantity=Quantity, Unit=Unit, Fraction=Fraction)
        ),
    ):
        assert restored == coordinate
        assert (20 @ restored).to_quantity(si.mV) == 10 * si.mV


def test_unit_call_bound_definition(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError('Bound units must not resolve spellings.')

    monkeypatch.setattr(type(default_registry), '_resolve', fail)
    assert si.dB(si.V) is si.dBV
    assert si.dB(si.mV).reference.unit == si.mV
