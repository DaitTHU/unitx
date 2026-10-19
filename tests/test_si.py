"""Public SI namespaces, conversions, and constant values."""

import os
import subprocess
import sys
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import pytest

from unitx.exceptions import UnitSymbolError
from unitx.factor import PI
from unitx.quantity import Quantity, quantity
from unitx.si import SI, si
from unitx.si.constants import Particle
from unitx.singleunit import SingleUnit
from unitx.unit import Unit


@pytest.mark.parametrize(
    ('attribute', 'expression'),
    [
        ('rad', ''),
        ('sr', ''),
        ('Hz', '1/s'),
        ('N', 'kg*m/s2'),
        ('Pa', 'kg/(m*s2)'),
        ('J', 'kg*m2/s2'),
        ('W', 'kg*m2/s3'),
        ('C', 'A*s'),
        ('V', 'kg*m2/(s3*A)'),
        ('F', 's4*A2/(kg*m2)'),
        ('ohm', 'kg*m2/(s3*A2)'),
        ('S', 's3*A2/(kg*m2)'),
        ('Wb', 'kg*m2/(s2*A)'),
        ('T', 'kg/(s2*A)'),
        ('H', 'kg*m2/(s2*A2)'),
        ('lm', 'cd*sr'),
        ('lx', 'cd*sr/m2'),
        ('Bq', '1/s'),
        ('Gy', 'm2/s2'),
        ('Sv', 'm2/s2'),
        ('kat', 'mol/s'),
    ],
)
def test_si_coherent_units(attribute: str, expression: str) -> None:
    unit = getattr(si, attribute)
    assert isinstance(unit, Unit)
    assert unit.dimension == Unit(expression).dimension
    assert Quantity(1, unit).to(expression).value == 1


@pytest.mark.parametrize(
    ('attribute', 'target', 'value'),
    [
        ('min', 's', 60),
        ('h', 's', 3600),
        ('d', 's', 86400),
        ('yr', 's', 31536000),
        ('a_j', 's', 31557600),
        ('mo', 's', 2629800),
        ('L', 'm3', Fraction(1, 1000)),
        ('t', 'kg', 1000),
        ('angstrom', 'm', Fraction(1, 10**10)),
        ('bar', 'Pa', 100000),
        ('au', 'm', 149597870700),
        ('nmile', 'm', 1852),
        ('knot', 'm/s', Fraction(463, 900)),
        ('atm', 'Pa', 101325),
        ('mHg', 'Pa', Fraction(2533125, 19)),
        ('mmHg', 'Pa', Fraction(20265, 152)),
        ('Torr', 'Pa', Fraction(20265, 152)),
        ('ly', 'm', 9460730472580800),
        ('a', 'm2', 100),
        ('ha', 'm2', 10000),
        ('u', 'kg', Decimal('1.66053906892e-27')),
        ('var', 'W', 1),
        ('Wh', 'J', 3600),
        ('eV', 'J', Decimal('1.602176634e-19')),
        ('cal', 'J', Decimal('4.184')),
    ],
)
def test_si_unit_conversions(attribute: str, target: str, value: int | Fraction | Decimal) -> None:
    assert Quantity(1, getattr(si, attribute)).to(target).value == value


def test_si_electronvolt() -> None:
    assert 1 * si.eV == SI.e * si.V
    assert (1 * si.MeV).to(si.eV).value == 10**6
    rest_energy = (SI.me * SI.c**2).to(si.MeV)
    assert float(rest_energy.value) == pytest.approx(0.51099895069, rel=1e-9)


def test_si_angles() -> None:
    assert si.deg.factor == PI / 180
    assert si.arcmin.factor == PI / 10800
    assert si.arcsec.factor == PI / 648000
    assert (1 * si.deg).to(si.arcsec).value == 3600
    assert si.pc.factor == si.au.factor / si.arcsec.factor


@pytest.mark.parametrize(
    ('attribute', 'text'),
    [
        ('us', 'µs'),
        ('um', 'micrometre'),
        ('ohm', 'Ω'),
        ('L', 'litre'),
        ('au', 'astronomical unit'),
        ('nmile', 'nautical mile'),
        ('mHg', 'metre of mercury'),
        ('mmHg', 'millimetre of mercury'),
        ('a_j', 'Julian year'),
    ],
)
def test_si_unit_names(attribute: str, text: str) -> None:
    assert getattr(si, attribute) == Unit(text)


def test_si_formal_identity() -> None:
    assert si.Hz != si.Bq
    assert si.Gy != si.Sv
    assert si.rad != si.one
    assert si.J != si.N * si.m
    assert 1 * si.Hz == 1 * si.Bq
    assert 1 * si.J == 1 * si.N * si.m


@pytest.mark.parametrize(
    ('attribute', 'target', 'value'),
    [
        ('DeltanuCs', 'Hz', '9192631770'),
        ('c', 'm/s', '299792458'),
        ('h', 'J*s', '6.62607015e-34'),
        ('e', 'C', '1.602176634e-19'),
        ('kB', 'J/K', '1.380649e-23'),
        ('NA', 'mol-1', '6.02214076e23'),
        ('Kcd', 'lm/W', '683'),
    ],
)
def test_si_defining_constants(attribute: str, target: str, value: str) -> None:
    with localcontext() as context:
        context.prec = 2
        result = getattr(SI, attribute).to(target)
    assert result.value == Decimal(value)


def test_si_hbar_value() -> None:
    assert SI.hbar.unit == SI.h.unit
    assert float(SI.hbar.value) == pytest.approx(1.0545718176461565e-34, rel=1e-15, abs=0)


@pytest.mark.parametrize(
    ('attribute', 'target'),
    [('Phi0', si.Wb), ('G0', si.S), ('RK', si.ohm), ('a0', si.m), ('lambdaC', si.m), ('Eh', si.J)],
)
def test_si_constant_units(attribute: str, target: Unit) -> None:
    assert getattr(SI, attribute).unit == target


def test_particle_helion_g_factor() -> None:
    assert float(Particle.gh.to(si.one).value) == pytest.approx(-4.2552506995, rel=1e-9)


@pytest.mark.parametrize(('namespace', 'attribute'), [(si, 'm'), (SI, 'c'), (Particle, 'me')])
def test_si_namespace_readonly(namespace, attribute: str) -> None:
    before = getattr(namespace, attribute)
    with pytest.raises(TypeError):
        namespace()
    with pytest.raises(AttributeError):
        setattr(namespace, attribute, None)
    with pytest.raises(AttributeError):
        delattr(namespace, attribute)
    assert getattr(namespace, attribute) is before


@pytest.mark.parametrize(
    'entry', ['unitx', 'unitx.si.bases', 'unitx.si.units', 'unitx.si.constants']
)
def test_si_import_initialization(entry: str, tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = f"""
import importlib
importlib.import_module({entry!r})
from unitx import Unit
from unitx.si import SI, si
assert Unit('N') == si.N
assert Unit('astronomical unit') == si.au
assert 1 * si.eV == SI.e * si.V
assert (760 * si.mmHg).to(si.atm).value == 1
assert (1 * si.ly).to(si.m).value == 9460730472580800
importlib.import_module('unitx.si')
"""
    result = subprocess.run(
        [sys.executable, '-W', 'error', '-c', script],
        cwd=tmp_path,
        env={**os.environ, 'PYTHONPATH': str(root / 'src')},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_quantity_decimal_representation() -> None:
    with localcontext() as context:
        context.prec = 2
        result = quantity(6.62607015e-34, si.J * si.s)
    assert type(result.value) is Decimal
    assert result.value == Decimal('6.62607015e-34')
    assert result.unit == si.J * si.s


@pytest.mark.parametrize('attribute', ['epsilon0', 'RK', 'R', 'F', 'c2', 'Vm'])
def test_si_decimal_display(attribute: str) -> None:
    constant = getattr(SI, attribute)
    assert type(constant.value) is Decimal
    assert str(constant).startswith(str(constant.value))
    assert len(str(constant.value)) < 40
    assert '/' not in str(constant.value)


def test_si_binding_policies() -> None:
    assert Unit('as').dimension == si.s.dimension
    assert Unit('Pa') == si.Pa
    assert Unit('a') == si.a
    assert Quantity(1, 'ha').to('m2').value == 10000
    assert Unit('ha') != Unit('a')
    assert Unit('var') != si.W
    assert Unit('var').factor == si.W.factor
    assert Unit('volt-ampere reactive') == si.var
    assert Unit('marcsec') == Unit('milliarcsecond')
    assert SingleUnit('marcsec').symbol == 'marcsec'
    assert Quantity(1000, 'marcsec').to(si.arcsec).value == 1
    for text in ('m″', 'm"', 'mas', 'mha', 'ka', 'kare', 'khectare'):
        with pytest.raises(UnitSymbolError):
            Unit(text)


def test_si_mercury_prefixes() -> None:
    assert si.mmHg == Unit('millimetre of mercury')
    assert (760 * si.mmHg).to(si.atm).value == 1
    assert (1000 * si.mmHg).to(si.mHg).value == 1
    assert (1 * si.Torr).to(si.mmHg).value == 1
    assert Unit('µmHg').factor == si.mHg.factor / 10**6
    assert Unit('kt') == si.knot
    for unit in (si.mHg, si.mmHg, si.Torr):
        assert Unit(unit.expression()) == unit
    for text in ('mmmHg', 'kmmHg', 'millimillimetre of mercury', 'kTorr'):
        with pytest.raises(UnitSymbolError):
            Unit(text)


def test_si_light_year() -> None:
    assert 1 * si.ly == SI.c * si.a_j
    assert 1 * si.eV == SI.e * si.V


@pytest.mark.parametrize('name', ['volt-ampere reactive', 'kilovolt-ampere reactive'])
def test_si_reactive_power_name(name: str) -> None:
    assert Unit('[' + name + '] / s') == Unit(name) / si.s
