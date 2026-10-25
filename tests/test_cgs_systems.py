"""CGS conventions agree on physical forces, fields and circuit quantities."""

import math
import pickle
from decimal import Decimal
from fractions import Fraction

import pytest

from unitx import Quantity, Unit, UnitCatalog
from unitx.cgs import cgs, emu, esu, gaussian, hl
from unitx.exceptions import DimensionError, UnitSymbolError
from unitx.factor import Factor
from unitx.serialization import from_dict, to_dict
from unitx.si import SI, si

_Catalogue = type[gaussian] | type[esu] | type[emu] | type[hl]
_SYSTEMS = (
    (gaussian, gaussian.statC, gaussian.statV, gaussian.G, gaussian.Oe, gaussian.statF),
    (esu, esu.statC, esu.statV, esu.magnetic_flux_density, esu.magnetic_field_strength, esu.statF),
    (emu, emu.abC, emu.abV, emu.G, emu.Oe, emu.abF),
    (
        hl,
        hl.charge,
        hl.potential,
        hl.magnetic_flux_density,
        hl.magnetic_field_strength,
        hl.capacitance,
    ),
)
_ROUTES = (
    (esu, si.C, esu.statC, 'charge'),
    (esu, si.A, esu.statA, 'current'),
    (esu, si.V, esu.statV, 'potential'),
    (esu, si.V / si.m, esu.statV / cgs.cm, 'electric_field'),
    (esu, si.T, esu.magnetic_flux_density, 'magnetic_flux_density'),
    (esu, si.Wb, esu.magnetic_flux, 'magnetic_flux'),
    (esu, si.F, esu.statF, 'capacitance'),
    (esu, si.A / si.m, esu.magnetic_field_strength, 'magnetic_field_strength'),
    (emu, si.C, emu.abC, 'charge'),
    (emu, si.A, emu.abA, 'current'),
    (emu, si.V, emu.abV, 'potential'),
    (emu, si.V / si.m, emu.abV / cgs.cm, 'electric_field'),
    (emu, si.T, emu.G, 'magnetic_flux_density'),
    (emu, si.Wb, emu.Mx, 'magnetic_flux'),
    (emu, si.F, emu.abF, 'capacitance'),
    (emu, si.A / si.m, emu.Oe, 'magnetic_field_strength'),
    (hl, si.C, hl.charge, 'charge'),
    (hl, si.A, hl.current, 'current'),
    (hl, si.V, hl.potential, 'potential'),
    (hl, si.V / si.m, hl.electric_field, 'electric_field'),
    (hl, si.T, hl.magnetic_flux_density, 'magnetic_flux_density'),
    (hl, si.Wb, hl.magnetic_flux, 'magnetic_flux'),
    (hl, si.F, hl.capacitance, 'capacitance'),
    (hl, si.A / si.m, hl.magnetic_field_strength, 'magnetic_field_strength'),
)


@pytest.mark.parametrize(('catalogue', 'source', 'target', 'kind'), _ROUTES)
@pytest.mark.parametrize('value', [0, -2.5, 1e-200, 1e200, Fraction(2, 3), 2 + 3j])
def test_cgs_system_round_trip(
    catalogue: _Catalogue, source: Unit, target: Unit, kind: str, value: complex | Fraction
) -> None:
    result = catalogue.convert(Quantity(value, source), target, kind=kind)
    restored = catalogue.convert(result, source, kind=kind)
    assert restored.value == pytest.approx(value, rel=3e-15, abs=0)
    if kind != 'electric_field' or catalogue is hl:
        assert catalogue.convert(Quantity(value, source), target) == result
        assert catalogue.convert(result, source) == restored
        scaled = Unit(catalogue.parse('m' + target.symbol), scale=Factor(7))
        assert catalogue.convert(Quantity(value, source), scaled).value == pytest.approx(
            result.value * 1000 / 7, rel=3e-15, abs=0
        )


@pytest.mark.parametrize(
    ('unit', 'definition'),
    [
        (emu.abC, (cgs.g * cgs.cm) ** Fraction(1, 2)),
        (emu.abA, emu.abC / cgs.s),
        (emu.abV, cgs.erg / emu.abC),
        (emu.abF, cgs.s**2 / cgs.cm),
        (esu.magnetic_flux_density, cgs.dyn * cgs.s / (esu.statC * cgs.cm)),
        (esu.magnetic_flux, esu.magnetic_flux_density * cgs.cm**2),
        (esu.magnetic_field_strength, esu.statA / cgs.cm),
        (hl.charge, gaussian.statC),
        (hl.current, gaussian.statA),
        (hl.potential, gaussian.statV),
        (hl.electric_field, gaussian.statV / cgs.cm),
        (hl.magnetic_flux_density, gaussian.G),
        (hl.magnetic_flux, gaussian.Mx),
        (hl.capacitance, cgs.cm),
        (hl.magnetic_field_strength, gaussian.Oe),
    ],
)
def test_cgs_system_definition(unit: Unit, definition: Unit) -> None:
    assert unit != definition
    assert unit.factor == definition.factor and unit.dimension == definition.dimension
    assert 1 * unit == 1 * definition
    assert hash(1 * unit) == hash(1 * definition)


@pytest.mark.parametrize('catalogue', [esu, emu, hl])
def test_cgs_system_catalogue(catalogue: type[UnitCatalog]) -> None:
    assert issubclass(catalogue, cgs)
    assert catalogue._registry is not cgs._registry
    for name in ('cm', 'g', 's', 'Gal', 'dyn', 'erg', 'Ba', 'P', 'St'):
        unit = getattr(catalogue, name)
        assert unit is getattr(cgs, name)
        assert catalogue.parse(unit.symbol) == unit
    with pytest.raises(AttributeError):
        catalogue.em = si.em
    assert not hasattr(cgs, 'em')


def test_cgs_system_shared_units() -> None:
    for name in ('statC', 'statA', 'statV', 'statF'):
        assert getattr(esu, name) is getattr(gaussian, name)
    for name in ('G', 'Mx', 'Oe'):
        assert getattr(emu, name) is getattr(gaussian, name)
    assert not hasattr(esu, 'G')
    assert not hasattr(emu, 'statC')


@pytest.mark.parametrize(
    ('catalogue', 'unit'), [(catalogue, target) for catalogue, _, target, _ in _ROUTES]
)
def test_cgs_system_reconstruction(catalogue: _Catalogue, unit: Unit) -> None:
    assert catalogue.parse(unit.expression()) == unit
    with pytest.raises(UnitSymbolError):
        Unit(unit.symbol)
    for value in (unit, Unit(unit, scale=Factor(3)), Fraction(2, 3) * unit):
        assert pickle.loads(pickle.dumps(value)) == value
        assert eval(repr(value)) == value
        assert from_dict(to_dict(value), registry=catalogue._registry) == value


@pytest.mark.parametrize(('catalogue', 'q_unit', 'v_unit', 'b_unit', 'h_unit', 'f_unit'), _SYSTEMS)
def test_cgs_system_forces_and_energy(
    catalogue: _Catalogue, q_unit: Unit, v_unit: Unit, b_unit: Unit, h_unit: Unit, f_unit: Unit
) -> None:
    em = catalogue.em
    q, e, b = 2e-9 * si.C, 20.0 * si.V / si.m, 0.03 * si.T
    velocity, radius = 4e5 * si.m / si.s, 4.0 * cgs.cm
    qc = catalogue.convert(q, q_unit)
    ec = catalogue.convert(e, v_unit / cgs.cm, kind='electric_field')
    bc = catalogue.convert(b, b_unit)
    pairs = [
        (em.k1 * qc**2 / radius**2, si.em.k1 * q**2 / radius**2),
        (qc * ec, q * e),
        (qc * em.alpha * velocity * bc, q * velocity * b),
    ]
    for actual, expected in pairs:
        assert actual.value_in(si.N) == pytest.approx(expected.value_in(si.N), rel=3e-15)
    actual_energy = ec**2 / (8 * math.pi * em.k1) + em.alpha * bc**2 / (8 * math.pi * em.k2)
    expected_energy = Quantity(float(SI.eps0.value), SI.eps0.unit) * e**2 / 2 + b**2 / (
        2 * Quantity(float(SI.mu0.value), SI.mu0.unit)
    )
    assert actual_energy.value_in(si.J / si.m**3) == pytest.approx(
        expected_energy.value_in(si.J / si.m**3), rel=3e-15
    )
    hc = catalogue.convert(b / Quantity(float(SI.mu0.value), SI.mu0.unit), h_unit)
    vacuum_h = bc * em.c**2 if catalogue is esu else bc
    assert hc.value_in(h_unit) == pytest.approx(vacuum_h.value_in(h_unit), rel=3e-15)
    sphere_si = 4 * math.pi * Quantity(float(SI.eps0.value), SI.eps0.unit) * radius
    sphere_cgs = catalogue.convert(sphere_si, f_unit)
    sphere_expected = radius / em.k1
    assert sphere_cgs.value == pytest.approx(sphere_expected.value_in(f_unit), rel=3e-15)
    assert (em.alpha * em.k2 / em.k1).value_in(si.s**2 / si.m**2) == pytest.approx(
        1 / 299792458**2, rel=2e-15
    )


def test_cgs_system_codata_correspondence() -> None:
    # Independently derive charge scales from Coulomb's law in CGS mechanics.
    ke = float(SI.ke.value)
    esu_per_coulomb = math.sqrt(10**9 * ke)
    emu_per_coulomb = math.sqrt(10**5 * ke) / 299792458
    hl_per_coulomb = math.sqrt(4 * math.pi * 10**9 * ke)
    assert esu.convert(1 * si.C, esu.statC).value == pytest.approx(esu_per_coulomb)
    assert emu.convert(1 * si.C, emu.abC).value == pytest.approx(emu_per_coulomb)
    assert hl.convert(1 * si.C, hl.charge).value == pytest.approx(hl_per_coulomb)
    # NIST's traditional central scales remain close after the 2019 SI change.
    assert emu.convert(1 * emu.abA, si.A).value == pytest.approx(10, rel=1e-9)
    assert emu.convert(1 * emu.abV, si.V).value == pytest.approx(1e-8, rel=1e-9)
    assert emu.convert(1 * emu.abF, si.F).value == pytest.approx(1e9, rel=1e-9)
    assert esu.convert(1 * esu.magnetic_flux_density, si.T).value == pytest.approx(
        2997924.58, rel=1e-9
    )


@pytest.mark.parametrize('catalogue', [esu, emu, hl])
def test_cgs_system_invalid(catalogue: _Catalogue) -> None:
    with pytest.raises(ValueError, match='Unknown .* conversion kind'):
        catalogue.convert(1 * si.C, cgs.cm, kind='invalid')
    with pytest.raises(DimensionError, match='dimensions do not match'):
        catalogue.convert(1 * si.C, cgs.cm, kind='charge')
    with pytest.raises(DimensionError, match='dimensions do not match'):
        catalogue.convert(1 * si.C, cgs.cm * si.dB, kind='charge')
    assert catalogue.convert(Decimal('1.25') * cgs.erg, si.J).value == Decimal('1.25e-7')


def test_cgs_system_correspondence_boundary() -> None:
    source = 1.0 * gaussian.statC
    assert source.to(hl.charge).value == 1.0
    assert hl.convert(source, hl.charge).value == 1.0
    with pytest.raises(DimensionError):
        hl.convert(source, hl.charge, kind='charge')
    result = hl.convert(gaussian.convert(source, si.C), hl.charge)
    assert result.value == pytest.approx(math.sqrt(4 * math.pi), rel=3e-15)
    with pytest.raises(DimensionError, match='specify kind'):
        hl.convert(1 * gaussian.statC, si.C)
    with pytest.raises(DimensionError, match='specify kind'):
        esu.convert(1 * gaussian.G, si.T)
    with pytest.raises(TypeError, match='Decimal and other domains are unsupported'):
        hl.convert(Decimal(1) * si.C, hl.charge)
