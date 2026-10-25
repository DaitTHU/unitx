"""Exact Imperial/US conventions and independent catalogue spellings."""

from __future__ import annotations

import copy
import importlib.util
import pickle
from decimal import Inexact, Rounded, localcontext
from fractions import Fraction

import pytest

from unitx import Quantity, Unit, UnitCatalog
from unitx.exceptions import UnitSymbolError
from unitx.factor import pi
from unitx.imperial import imperial, imperial_historical, units, us
from unitx.serialization import from_dict, to_dict
from unitx.si import si


@pytest.mark.parametrize(
    ('unit', 'target', 'expected'),
    [
        (us.us_survey_foot, si.m, Fraction(1200, 3937)),
        (us.us_survey_mile, si.m, Fraction(6336000, 3937)),
        (us.us_cable, si.m, Fraction('219.456')),
        (imperial.btu_IT, si.J, Fraction('1055.05585262')),
        (imperial.btu_th, si.J, Fraction(23722880951, 22500000)),
        (us.inHg, si.Pa, Fraction('3386.388640341')),
        (us.inH2O, si.Pa, Fraction('249.08891')),
        (us.us_dry_bushel, si.L, Fraction('35.23907016688')),
        (us.us_standard_dry_barrel, imperial.inch**3, 7056),
        (us.us_cranberry_barrel, imperial.inch**3, 5826),
        (us.us_petroleum_barrel, us.us_gal, 42),
        (imperial_historical.ramsden_chain, imperial.ft, 100),
    ],
)
def test_imperial_exact_conversion(unit: Unit, target: Unit, expected: Fraction | int) -> None:
    assert Quantity(1, unit).value_in(target) == expected


def test_us_survey_foot_distinct_convention() -> None:
    ratio = (1 * us.us_survey_foot).value_in(imperial.ft)
    assert ratio == Fraction(500000, 499999)
    assert ratio != 1
    assert us.unit('[US cable length]') == us.us_cable
    assert (1 * us.us_cable).value_in(us.us_survey_foot) != 720


def test_imperial_btu_thermochemical_relation() -> None:
    # The NIST REFPROP convention uses 4.184 J/cal, 453.59237 g/lb and 5/9 K/°R.
    assert (1 * imperial.btu_th / imperial.btu_IT).value_in(Unit()) == Fraction(10460, 10467)
    specific_heat = imperial.btu_th / imperial.lb / imperial.degR
    assert (1 * specific_heat).value_in(si.J / si.kg / si.K) == 4184


def test_us_circular_mil_symbolic_area() -> None:
    assert us.circular_mil.dimension == si.m.dimension**2
    assert us.circular_mil.factor == pi * Fraction(16129, 10**14)
    assert (us.circular_mil / imperial.th**2).factor == pi / 4


@pytest.mark.parametrize(
    ('unit', 'density'),
    [(us.inHg, Fraction('13595.1')), (us.inH2O, Fraction(1000))],
)
def test_us_column_pressure_convention(unit: Unit, density: Fraction) -> None:
    assert (1 * unit).value_in(si.Pa) == density * Fraction('9.80665') * Fraction('0.0254')


@pytest.mark.parametrize('precision', [2, 8, 28])
def test_imperial_decimal_context(precision: int) -> None:
    # Load just this module after SI initialization: SI constant approximations
    # have their own precision policy, outside the Imperial definition contract.
    spec = importlib.util.spec_from_file_location('unitx.imperial._precision_probe', units.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with localcontext() as context:
        context.prec = precision
        context.traps[Inexact] = context.traps[Rounded] = True
        spec.loader.exec_module(module)
        assert (1 * module.us.us_survey_foot).value_in(si.m) == Fraction(1200, 3937)
        assert (1 * module.imperial.btu_th).value_in(si.J) == Fraction(23722880951, 22500000)
        assert module.us.circular_mil.factor == us.circular_mil.factor


@pytest.mark.parametrize(
    ('text', 'unit'),
    [
        ('cmil', us.circular_mil),
        ('[circular mil]', us.circular_mil),
        ('inH₂O', us.inH2O),
        ('[inch of water]', us.inH2O),
        ('[conventional inch of water]', us.inH2O),
        ('[conventional inch of mercury]', us.inHg),
    ],
)
def test_us_spelling_scope(text: str, unit: Unit) -> None:
    assert us.unit(text) == unit
    assert us.unit(text + ' / s') == unit / si.s
    with pytest.raises(UnitSymbolError):
        Unit(text)


@pytest.mark.parametrize(
    ('catalogue', 'text'),
    [(us, 'kcmil'), (us, 'kinH₂O'), (us, 'kinHg'), (imperial, 'kBtu_th')],
)
def test_imperial_invalid_prefix(catalogue: type[UnitCatalog], text: str) -> None:
    with pytest.raises(UnitSymbolError):
        catalogue.unit(text)


@pytest.mark.parametrize('unit', [us.circular_mil, us.inH2O, us.inHg, us.us_survey_foot])
def test_us_reconstruction(unit: Unit) -> None:
    assert copy.copy(unit) is copy.deepcopy(unit) is unit
    assert pickle.loads(pickle.dumps(unit)) == unit
    assert eval(repr(unit)) == unit
    restored = from_dict(to_dict(unit), registry=us._registry)
    assert restored == unit and hash(restored) == hash(unit)
