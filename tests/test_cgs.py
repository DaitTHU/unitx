"""Exact catalogue definitions and the explicit electromagnetic boundary."""

import copy
import math
import pickle
import subprocess
import sys
from decimal import Decimal
from fractions import Fraction

import pytest

from unitx import Quantity, Unit, UnitCatalog
from unitx._numerics.conversion import convert
from unitx.cgs import cgs, gaussian
from unitx.dimension import Dimension
from unitx.exceptions import DimensionError, UnitSymbolError
from unitx.factor import Factor, one
from unitx.serialization import from_dict, to_dict
from unitx.si import SI, si


@pytest.mark.parametrize(
    ('unit', 'target', 'expected'),
    [
        (cgs.dyn, si.N, Fraction(1, 10**5)),
        (cgs.erg, si.J, Fraction(1, 10**7)),
        (cgs.Ba, si.Pa, Fraction(1, 10)),
        (cgs.P, si.Pa * si.s, Fraction(1, 10)),
        (cgs.St, si.m**2 / si.s, Fraction(1, 10**4)),
        (cgs.Gal, si.m / si.s**2, Fraction(1, 100)),
    ],
)
def test_cgs_exact_conversion(unit: Unit, target: Unit, expected: Fraction) -> None:
    result = (1 * unit).value_in(target)
    assert result == expected
    assert hash(1 * unit) == hash(expected * target)
    assert Unit(unit, scale=Factor(3)).factor == unit.factor * 3


def test_cgs_mechanical_algebra() -> None:
    assert cgs.cm is si.cm and cgs.g is si.g and cgs.s is si.s and cgs.Gal is si.Gal
    assert (3 * cgs.dyn * (2 * cgs.cm)).to(cgs.erg) == 6 * cgs.erg
    assert (1 * si.J).value_in(cgs.g * cgs.cm**2 / cgs.s**2) == 10**7
    assert (cgs.dyn * cgs.cm / cgs.erg).factor == one
    assert cgs.dyn != cgs.g * cgs.cm / cgs.s**2


@pytest.mark.parametrize(
    ('unit', 'definition'),
    [
        (gaussian.statC, cgs.g ** Fraction(1, 2) * cgs.cm ** Fraction(3, 2) / cgs.s),
        (gaussian.statA, gaussian.statC / cgs.s),
        (gaussian.statV, cgs.erg / gaussian.statC),
        (gaussian.statF, cgs.cm),
        (gaussian.G, cgs.dyn / gaussian.statC),
        (gaussian.Mx, gaussian.statC),
        (gaussian.Oe, gaussian.G),
    ],
)
def test_cgs_gaussian_definition(unit: Unit, definition: Unit) -> None:
    assert unit.dimension == definition.dimension
    assert unit.factor == definition.factor
    assert unit != definition
    assert 1 * unit == 1 * definition
    assert hash(1 * unit) == hash(1 * definition)


def test_cgs_rational_dimensions() -> None:
    assert gaussian.statC.dimension == Dimension(M=Fraction(1, 2), L=Fraction(3, 2), T=-1)
    assert gaussian.statC.factor**2 == Factor(Fraction(1, 10**9))
    assert ((gaussian.statC**2) ** Fraction(1, 2)) == gaussian.statC


@pytest.mark.parametrize(
    ('symbol', 'name'),
    [
        ('dyn', 'dyne'),
        ('erg', 'erg'),
        ('Ba', 'barye'),
        ('P', 'poise'),
        ('St', 'stokes'),
        ('statC', 'statcoulomb'),
        ('statA', 'statampere'),
        ('statV', 'statvolt'),
        ('statF', 'statfarad'),
        ('G', 'gauss'),
        ('Mx', 'maxwell'),
        ('Oe', 'oersted'),
    ],
)
def test_gaussian_spelling(symbol: str, name: str) -> None:
    unit = getattr(gaussian, symbol)
    assert gaussian.parse(symbol) == gaussian.parse(name) == unit
    assert gaussian.parse('m' + symbol).factor == unit.factor / 1000
    assert gaussian.parse('milli' + name) == gaussian.parse('m' + symbol)
    if symbol in ('dyn', 'erg', 'Ba', 'P', 'St'):
        assert cgs.parse(symbol) == cgs.parse(name) == unit
        assert cgs.parse('m' + symbol) == gaussian.parse('m' + symbol)
    with pytest.raises(UnitSymbolError):
        Unit(symbol)


def test_cgs_parser_scope() -> None:
    assert cgs.parse('erg/cm') == cgs.erg / cgs.cm
    assert cgs.parse('gal') == Unit('gal') == si.Gal
    assert cgs.parse('cP').factor == cgs.P.factor / 100
    assert cgs.parse('cSt').factor == cgs.St.factor / 100
    with pytest.raises(UnitSymbolError):
        cgs.parse('Fr')


def test_gaussian_inherits_mechanical_units() -> None:
    assert issubclass(gaussian, cgs)
    assert gaussian._registry is not cgs._registry
    for symbol in ('cm', 'g', 's', 'Gal', 'dyn', 'erg', 'Ba', 'P', 'St'):
        assert getattr(gaussian, symbol) is getattr(cgs, symbol)
        assert gaussian.parse(symbol) == cgs.parse(symbol)
    assert gaussian.parse('erg/statC') == cgs.erg / gaussian.statC
    assert gaussian.parse('g^(1/2) cm^(3/2)/s').factor == gaussian.statC.factor


@pytest.mark.parametrize('symbol', ['statC', 'statA', 'statV', 'statF', 'G', 'Mx', 'Oe'])
def test_cgs_invalid_electromagnetic_unit(symbol: str) -> None:
    assert not hasattr(cgs, symbol)
    with pytest.raises(UnitSymbolError):
        cgs.parse(symbol)


@pytest.mark.parametrize('catalogue', [cgs, gaussian])
def test_cgs_namespace(catalogue: type[UnitCatalog]) -> None:
    with pytest.raises(TypeError):
        catalogue()
    with pytest.raises(AttributeError):
        catalogue.dyn = si.N
    with pytest.raises(AttributeError):
        del catalogue.cm


@pytest.mark.parametrize('unit', [cgs.dyn, gaussian.statC, gaussian.Mx, gaussian.G, gaussian.Oe])
def test_cgs_reconstruction(unit: Unit) -> None:
    scaled = Unit(unit, scale=Factor(3))
    for value in (unit, scaled, Fraction(2, 3) * scaled):
        assert copy.copy(value) == copy.deepcopy(value) == value
        assert pickle.loads(pickle.dumps(value)) == value
        assert eval(repr(value)) == value
        assert from_dict(to_dict(value), registry=gaussian._registry) == value


def test_cgs_fresh_process_reconstruction() -> None:
    record = to_dict(2 * gaussian.statC)
    script = (
        'from unitx.cgs import gaussian; from unitx.serialization import from_dict; '
        f'assert from_dict({record!r}, registry=gaussian._registry) == 2*gaussian.statC'
    )
    subprocess.run([sys.executable, '-c', script], check=True, timeout=20)


@pytest.mark.parametrize(
    ('source', 'target', 'kind'),
    [
        (si.C, gaussian.statC, 'charge'),
        (si.A, gaussian.statA, 'current'),
        (si.V, gaussian.statV, 'potential'),
        (si.V / si.m, gaussian.statV / cgs.cm, 'electric_field'),
        (si.T, gaussian.G, 'magnetic_flux_density'),
        (si.Wb, gaussian.Mx, 'magnetic_flux'),
        (si.F, gaussian.statF, 'capacitance'),
        (si.A / si.m, gaussian.Oe, 'magnetic_field_strength'),
    ],
)
@pytest.mark.parametrize('value', [0, -2.5, 1e-200, 1e200, Fraction(2, 3), 2 + 3j])
def test_gaussian_round_trip(
    source: Unit, target: Unit, kind: str, value: complex | Fraction
) -> None:
    original = Quantity(value, source)
    result = gaussian.convert(original, target, kind=kind)
    restored = gaussian.convert(result, source, kind=kind)
    assert restored.value == pytest.approx(value, rel=2e-15, abs=0)
    assert original.value == value and original.unit == source
    assert type(result) is Quantity
    if kind != 'electric_field':
        assert gaussian.convert(original, target) == result
        assert gaussian.convert(result, source) == restored


def test_gaussian_named_scale() -> None:
    target = Unit(gaussian.parse('mstatC'), scale=Factor(7))
    source = Unit(si.mC, scale=Factor(3))
    result = gaussian.convert(2 * source, target)
    reference = gaussian.convert(2 * source, gaussian.statC)
    assert result.value == pytest.approx(reference.value * 1000 / 7)
    assert gaussian.convert(result, source).value == pytest.approx(2)
    assert (
        gaussian.convert(1 * gaussian.Mx, si.Wb).value
        != gaussian.convert(1 * gaussian.statC, si.C).value
    )


def test_gaussian_same_dimension() -> None:
    result = gaussian.convert(Decimal('1.25') * cgs.erg, si.J)
    assert result.value == Decimal('0.000000125')
    assert type(result.value) is Decimal


@pytest.mark.parametrize(
    ('source', 'target', 'kind'),
    [
        (si.C, si.C, 'charge'),
        (cgs.cm, si.m, 'capacitance'),
        (gaussian.statC * si.dB, si.C, 'charge'),
        (si.C, gaussian.statC * si.dB, 'charge'),
        (gaussian.G, si.C, 'charge'),
    ],
)
def test_gaussian_invalid_dimensions(source: Unit, target: Unit, kind: str) -> None:
    with pytest.raises(DimensionError, match='dimensions do not match'):
        gaussian.convert(1 * source, target, kind=kind)


@pytest.mark.parametrize(
    ('source', 'target'),
    [
        (cgs.g ** Fraction(1, 2) * cgs.cm ** Fraction(3, 2) / cgs.s, si.C),
        (cgs.cm, si.F),
        (gaussian.statV / cgs.cm, si.V / si.m),
        (gaussian.statC.define('statC'), si.C),
        (gaussian.G, si.A / si.m),
    ],
)
def test_gaussian_invalid_inference(source: Unit, target: Unit) -> None:
    with pytest.raises(DimensionError, match='specify kind'):
        gaussian.convert(1 * source, target)


@pytest.mark.parametrize('kind', ['', 'electric_displacement', 'polarization', 'magnetization'])
def test_gaussian_invalid_kind(kind: str) -> None:
    with pytest.raises(ValueError, match='Unknown Gaussian conversion kind'):
        gaussian.convert(1 * si.C, gaussian.statC, kind=kind)


def test_gaussian_invalid_magnitude() -> None:
    with pytest.raises(TypeError, match='Decimal and other domains are unsupported'):
        gaussian.convert(Decimal(1) * si.C, gaussian.statC)


@pytest.mark.parametrize('quantity', [si.C, 1, 20 @ si.degC, 0 @ si.dBm])
def test_gaussian_invalid_quantity(quantity: object) -> None:
    with pytest.raises(TypeError, match='linear Quantity and a target Unit'):
        gaussian.convert(quantity, gaussian.statC)  # type: ignore[arg-type]


@pytest.mark.parametrize('target', ['statC', si.degC, si.dBm])
def test_gaussian_invalid_target(target: object) -> None:
    with pytest.raises(TypeError, match='linear Quantity and a target Unit'):
        gaussian.convert(1 * si.C, target)  # type: ignore[arg-type]


def test_gaussian_invalid_kind_type() -> None:
    with pytest.raises(TypeError, match='kind must be a string'):
        gaussian.convert(1 * si.C, gaussian.statC, kind=[])  # type: ignore[arg-type]


def test_gaussian_linear_boundary() -> None:
    left, right = 1 * gaussian.statC, 1 * si.C
    assert left != right
    assert len({left, right}) == 2
    with pytest.raises(DimensionError):
        left.to(si.C)
    with pytest.raises(DimensionError):
        convert(1, gaussian.statC, si.C)
    with pytest.raises(DimensionError):
        left + right
    with pytest.raises(DimensionError):
        left < right


@pytest.mark.parametrize('em', [si.em, gaussian.em])
def test_gaussian_coefficients(em: object) -> None:
    assert (em.alpha * em.k2 / em.k1).value_in(si.s**2 / si.m**2) == pytest.approx(
        1 / 299792458**2, rel=2e-15, abs=0
    )
    assert em.c.value_in(si.m / si.s) == 299792458
    with pytest.raises(AttributeError):
        em.k1 = 2
    with pytest.raises(AttributeError):
        del em.alpha


def test_gaussian_em_namespace() -> None:
    assert type(si.em) is type(gaussian.em)
    assert not hasattr(cgs, 'em')
    assert not hasattr(gaussian, 'k1')
    assert isinstance(si.c, Unit)
    assert (1 * si.c) == si.em.c == SI.c
    assert type(si.em.c.value) is float
    assert type(SI.c.value) is int
    assert type(SI.mu0.value) is Decimal
    assert SI.mu0.value == Decimal('0.00000125663706127')
    assert si.parse('Em').factor == si.m.factor * 10**18


def test_si_em_invalid_spelling() -> None:
    # Namespace attributes do not become parser spellings.
    with pytest.raises(UnitSymbolError):
        si.parse('em')


@pytest.mark.parametrize('catalogue', [si, gaussian])
def test_gaussian_em_readonly(catalogue: type[UnitCatalog]) -> None:
    with pytest.raises(AttributeError):
        setattr(catalogue, 'em', si.em)
    with pytest.raises(AttributeError):
        delattr(catalogue, 'em')


def test_gaussian_codata_correspondence() -> None:
    # Independent mechanical derivation: sqrt(10^9 * ke_SI) statC per coulomb.
    ke = float(SI.ke.value)
    assert gaussian.convert(1 * si.C, gaussian.statC).value == pytest.approx(math.sqrt(10**9 * ke))
    # Modern mu0 differs from its traditional exact value; do not freeze 1e-4 T.
    b = gaussian.convert(1 * gaussian.G, si.T).value
    assert math.isclose(b, 1e-4, rel_tol=1e-9)
    assert b != 1e-4


def test_gaussian_coulomb_force() -> None:
    q1, q2, r = 2e-9 * si.C, -3e-9 * si.C, 4 * si.cm
    expected = SI.ke * Quantity(Decimal('2e-9'), si.C) * Quantity(Decimal('-3e-9'), si.C) / r**2
    q1g, q2g = gaussian.convert(q1, gaussian.statC), gaussian.convert(q2, gaussian.statC)
    actual = gaussian.em.k1 * q1g * q2g / r**2
    assert actual.value_in(si.N) == pytest.approx(float(expected.value_in(si.N)), rel=2e-15)


def test_gaussian_lorentz_force() -> None:
    q, e, v, b = 2e-9 * si.C, 3.0 * si.V / si.m, 4e5 * si.m / si.s, 0.2 * si.T
    qg = gaussian.convert(q, gaussian.statC)
    eg = gaussian.convert(e, gaussian.statV / cgs.cm, kind='electric_field')
    bg = gaussian.convert(b, gaussian.G)
    for actual, expected in [(qg * eg, q * e), (qg * gaussian.em.alpha * v * bg, q * v * b)]:
        assert actual.value_in(si.N) == pytest.approx(expected.value_in(si.N), rel=2e-15)


def test_gaussian_vacuum_energy() -> None:
    e, b = 20.0 * si.V / si.m, 0.03 * si.T
    eg = gaussian.convert(e, gaussian.statV / cgs.cm, kind='electric_field')
    bg = gaussian.convert(b, gaussian.G)
    actual = (eg**2 + bg**2) / (8 * math.pi)
    expected = float(SI.eps0.value) * e**2 * SI.eps0.unit / 2 + b**2 / (
        2 * Quantity(float(SI.mu0.value), SI.mu0.unit)
    )
    assert actual.value_in(si.J / si.m**3) == pytest.approx(
        expected.value_in(si.J / si.m**3), rel=2e-15
    )


def test_gaussian_sphere_capacitance() -> None:
    radius = 3.0 * cgs.cm
    capacitance = 4 * math.pi * Quantity(float(SI.eps0.value), SI.eps0.unit) * radius
    actual = gaussian.convert(capacitance, cgs.cm, kind='capacitance')
    assert actual.value == pytest.approx(radius.value, rel=2e-15)
    assert gaussian.convert(actual, si.F, kind='capacitance').value == pytest.approx(
        capacitance.value_in(si.F), rel=2e-15
    )
