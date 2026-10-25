"""Explicit environments bind text; constructed units keep definition identity."""

from __future__ import annotations

import copy
import json
import os
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from threading import Barrier

import pytest

from unitx import AffineQuantity, AffineUnit, Quantity, Registry, Unit, UnitCatalog
from unitx.exceptions import SerializationError, UnitRegistryError, UnitSymbolError
from unitx.factor import Factor, pi
from unitx.imperial import imperial
from unitx.serialization import from_dict, to_dict
from unitx.si import si
from unitx.singleunit import SingleUnit
from unitx.unitparser import UnitParser


def test_registry_shared_identity() -> None:
    left, right = Registry([si.m, si.s]), Registry([si.m])
    assert left.parse('m') == right.parse('metre') == si.m
    assert hash(left.parse('m')) == hash(si.m)
    assert left.parse('km/s') == si.km / si.s
    right.include(si.s)
    right.include(si.s)
    assert right.parse('m/s') == left.parse('m/s')


def test_registry_concurrent_reads() -> None:
    registry = Registry([si.m, si.s])
    span = registry.define(2 * si.m, 'span', 'project span', symbol_aliases='sp')
    cases = (
        ('sp/s', span / si.s),
        ('[project span] / s', span / si.s),
        ('km^(1/2)/s', si.km ** Fraction(1, 2) / si.s),
        ('project span', span),
    )
    ready = Barrier(len(cases))

    def read(case: tuple[str, Unit]) -> None:
        expression, expected = case
        for _ in range(20):
            ready.wait(timeout=10)
            assert registry.parse(expression) == expected
            with pytest.raises(UnitSymbolError):
                registry.parse('missing')

    with ThreadPoolExecutor(max_workers=len(cases)) as workers:
        list(workers.map(read, cases))


def test_registry_independent_identity() -> None:
    left, right = Registry(), Registry()
    first = left.define(si.m, 'span', 'private span')
    second = right.define(si.m, 'span', 'private span')
    assert first != second
    assert len({first, second, si.m}) == 3
    assert first / second != Unit()
    assert left.parse('span') == first
    assert right.parse('span') == second
    assert Quantity(1, first) == Quantity(1, second) == Quantity(1, si.m)
    assert {Quantity(1, first): 'length'}[Quantity(1, second)] == 'length'
    assert Quantity(3, first).to(second).value == 3
    assert (Quantity(1, first) + Quantity(2, second)).value == 3


def test_registry_conflicting_symbol() -> None:
    alternate = Registry()
    byte = alternate.define(Unit(), 'B', 'byte')
    assert alternate.parse('B') == byte
    assert alternate.parse('kB').factor == 1000
    assert alternate.parse('B').dimension == si.one.dimension
    assert Unit('B') == si.B and Unit('B').dimension != si.one.dimension
    with pytest.raises(UnitSymbolError):
        alternate.parse('bel')


def test_registry_no_fallback() -> None:
    private = Registry()
    assert private.parse('') == Unit()
    with pytest.raises(UnitSymbolError):
        private.parse('m')
    private.define(si.m, 'rvisolated', 'isolated span')
    with pytest.raises(UnitSymbolError):
        Unit('rvisolated')


@pytest.mark.parametrize('units', [None, False, 0])
def test_registry_invalid_units(units: None | bool | int) -> None:
    with pytest.raises(TypeError):
        Registry(units)


def test_registry_parser_retry() -> None:
    private = Registry([si.s])
    parser = UnitParser('[private distance] / s', registry=private)
    with pytest.raises(UnitSymbolError):
        parser.parse()
    distance = private.define(si.m, 'span', 'private distance', prefixable='private distance')
    result = parser.parse()
    assert result is parser.parse()
    assert Unit._from_monomial(result) == distance / si.s
    assert private.parse('kiloprivate distance').factor == 1000
    with pytest.raises(UnitSymbolError):
        private.parse('kspan')


@pytest.mark.parametrize(
    'symbol, name', [('m', 'other'), ('cm', 'other'), ('span', 'metre second')]
)
def test_registry_invalid_conflict(symbol: str, name: str) -> None:
    private = Registry([si.m, si.s])
    original = private.parse('m/s')
    with pytest.raises(UnitRegistryError):
        private.define(si.s, symbol, name)
    assert private.parse('m/s') == original
    with pytest.raises(UnitSymbolError):
        private.parse('other')


def test_registry_include_invalid_conflict() -> None:
    private = Registry([si.s])
    private.define(si.s, 'm', 'private metre', prefixable=False)
    with pytest.raises(UnitRegistryError):
        private.include(si.kg * si.m)
    with pytest.raises(UnitSymbolError):
        private.parse('kg')
    assert private.parse('m').dimension == si.s.dimension


def test_registry_include_shares_definition() -> None:
    first, second = Registry(), Registry()
    unit = first.define(si.m, 'span')
    second.include(unit)
    shared = second.parse('span')
    assert unit == shared and hash(unit) == hash(shared)
    assert next(iter(unit._monomial.bases()))._unit is next(iter(shared._monomial.bases()))._unit


def test_registry_include_batch() -> None:
    private = Registry()
    private.include(si.m, si.s, si.m)
    assert private.parse('metre/second') == si.m / si.s
    assert {private.parse('m'): 'length'}[si.m] == 'length'


def test_registry_include_invalid_batch_conflict() -> None:
    private = Registry([si.m])
    conflicting = si.s.define('m', prefixable=False)
    with pytest.raises(UnitRegistryError):
        private.include(si.s, conflicting)
    assert private.parse('m') == si.m
    with pytest.raises(UnitSymbolError):
        private.parse('second')


def test_registry_include_invalid_batch_object() -> None:
    private = Registry([si.m])
    with pytest.raises(TypeError, match='Registry.include requires Unit or named AffineUnit'):
        private.include(si.s, 3)
    with pytest.raises(UnitSymbolError):
        private.parse('s')


@pytest.mark.parametrize('protocol', range(pickle.HIGHEST_PROTOCOL + 1))
def test_registry_pickle_identity(protocol: int) -> None:
    private = Registry()
    unit = private.define(si.m, 'span', 'private span')
    for original in (unit, Unit._from_monomial(UnitParser('kspan', registry=private).parse())):
        restored = pickle.loads(pickle.dumps(original, protocol=protocol))
        assert restored == original
        assert hash(restored) == hash(original)
        assert {original: 'found'}[restored] == 'found'
    # Two independent meanings of one symbol survive in a single product.
    other = Registry().define(si.s, 'span', 'private span')
    combined = unit / other
    assert pickle.loads(pickle.dumps(combined, protocol=protocol)) == combined


def test_registry_reconstruction() -> None:
    private = Registry([si.s])
    unit = private.define(si.m, 'span', 'private span')
    single = next(iter(UnitParser('kspan', registry=private).parse().bases()))
    for original in (single, unit, Factor(7) * unit / si.s, Quantity(Decimal('2.50'), unit)):
        assert copy.copy(original) == original
        assert copy.deepcopy(original) == original
        assert (
            eval(repr(original), globals() | {'SingleUnit': SingleUnit, 'Unit': Unit}) == original
        )
        assert pickle.loads(pickle.dumps(original)) == original
    definition = single._unit
    assert copy.copy(definition) is copy.deepcopy(definition) is definition
    assert pickle.loads(pickle.dumps(definition)) is definition


def test_registry_json_receiver() -> None:
    private = Registry()
    unit = private.define(si.m, 'span', 'private span')
    record = json.loads(json.dumps(to_dict(Quantity(Fraction(2, 7), unit))))
    assert from_dict(record, registry=private) == Quantity(Fraction(2, 7), unit)
    with pytest.raises(SerializationError):
        from_dict(record)
    other = Registry()
    other.define(si.m, 'span', 'private span')
    with pytest.raises(SerializationError, match='conflicts'):
        from_dict(record, registry=other)


def test_registry_affine_bound_unit() -> None:
    private = Registry()
    kelvin = private.define(si.K, 'degC', 'private temperature')
    # A bound Unit named degC must not acquire the default Celsius offset.
    assert AffineQuantity(0, kelvin).to_quantity(si.K).value == 0
    before = Unit('degC')
    coordinate = AffineUnit(Quantity(2, kelvin))
    named = coordinate.define('rvisolatedepoch', 'private epoch')
    assert (0 @ named).to_quantity(si.K).value == 2
    assert Unit('degC') == before
    with pytest.raises(UnitSymbolError):
        Unit('rvisolatedepoch')
    private.include(named)
    assert AffineQuantity(0, private.parse('rvisolatedepoch')).unit == named
    assert from_dict(to_dict(3 @ named), registry=private) == 3 @ named


@pytest.mark.parametrize('protocol', range(pickle.HIGHEST_PROTOCOL + 1))
def test_registry_scaled_coordinate_reconstruction(protocol: int) -> None:
    reference = Unit(si.K, scale=pi)
    delta = Unit(si.K, scale=pi / 2)
    named = AffineUnit(10 * reference, delta_unit=delta).define('scaledReading')
    private = Registry([named])
    point = 4 @ named
    assert point.value_in(reference) == 12
    assert point - (2 @ named) == 2 * delta
    assert AffineQuantity(4, private.parse('scaledReading')) == point
    restored = pickle.loads(pickle.dumps(named, protocol=protocol))
    assert restored == named and 4 @ restored == point
    assert pickle.loads(pickle.dumps(point)) == point
    assert from_dict(to_dict(point), registry=private) == point


def test_registry_fresh_process_identity(tmp_path: Path) -> None:
    module = tmp_path / 'survey_catalogue.py'
    module.write_text("""
from unitx import AffineUnit, LogUnit, UnitCatalog
from unitx.si import si
class survey(UnitCatalog):
    span = (2 * si.m).define('span')
    temperature = AffineUnit(10 * si.K).define('surveyT')
    level = LogUnit(2 * si.W, at=1, delta_unit=si.dB, kind='power').define('surveyLevel')
""")
    code = """
import json, pickle, sys
from unitx import AffineQuantity, LogQuantity, Quantity
from unitx.serialization import to_dict, from_dict
from survey_catalogue import survey
q = (
    Quantity(3, survey.span),
    AffineQuantity(4, survey.temperature),
    LogQuantity(5, survey.level),
)
if sys.argv[1] == 'write':
    open(sys.argv[2], 'wb').write(pickle.dumps(q))
    open(sys.argv[3], 'w').write(json.dumps([to_dict(value) for value in q]))
else:
    assert pickle.loads(open(sys.argv[2], 'rb').read()) == q
    saved = json.load(open(sys.argv[3]))
    assert tuple(from_dict(value, registry=survey._registry) for value in saved) == q
"""
    arguments = [str(tmp_path / 'unit.pickle'), str(tmp_path / 'unit.json')]
    for action in ('write', 'read'):
        subprocess.run(
            [sys.executable, '-c', code, action, *arguments],
            cwd=tmp_path,
            check=True,
            env=os.environ.copy(),
            timeout=20,
        )


@pytest.mark.parametrize(
    'value, target',
    [
        (imperial.inch, si.m),
        (imperial.ft, si.m),
        (imperial.lb, si.kg),
        (imperial.gal, si.L),
        (imperial.lbf, si.N),
    ],
)
def test_imperial_exact_conversion(value: Unit, target: Unit) -> None:
    expected = {
        imperial.inch: Fraction('0.0254'),
        imperial.ft: Fraction('0.3048'),
        imperial.lb: Fraction('0.45359237'),
        imperial.gal: Fraction('4.54609'),
        imperial.lbf: Fraction('4.4482216152605'),
    }
    assert Quantity(1, value).value_in(target) == expected[value]


def test_imperial_catalogue() -> None:
    from unitx.imperial import imperial as namespace

    assert namespace is imperial
    assert imperial.unit('ft/s') == imperial.ft / si.s
    assert imperial.unit('[fluid ounce]') == imperial.fl_oz
    assert imperial.unit('feet') == imperial.ft
    assert imperial.unit('cm') == si.cm
    assert imperial.unit('gal') == imperial.gal
    assert Unit('gal') == si.Gal
    assert imperial.unit('ton').factor == 2240 * imperial.lb.factor
    assert (20 * imperial.fl_oz).to(imperial.pt).value == 1
    assert (3 * imperial.ft + 1 * si.m).value_in(si.m) == Fraction('1.9144')
    with localcontext() as context:
        context.prec = 2
        assert Quantity(Decimal('1'), imperial.ft).value_in(si.m) == Decimal('0.30')
    with pytest.raises(UnitSymbolError):
        Unit('ft')
    with pytest.raises(UnitSymbolError):
        imperial.unit('kft')
    with pytest.raises(AttributeError):
        namespace.ft = si.m


def test_imperial_module_import() -> None:
    import unitx.imperial.units as definitions

    assert imperial.ft is definitions.imperial.ft


def test_unit_catalog_shared_entry() -> None:
    from unitx.imperial import imperial as namespace

    assert issubclass(si, UnitCatalog)
    assert issubclass(namespace, UnitCatalog)
    assert not hasattr(UnitCatalog, 'parse')
    assert si.unit('kg m/s2') == si.N.factor * si.kg * si.m / si.s**2
    assert namespace.unit('ft/s') == imperial.ft / si.s
    with pytest.raises(TypeError):
        namespace()
    with pytest.raises(AttributeError):
        namespace._registry = Registry()


def test_unit_catalog_custom_entry() -> None:
    class laboratory(UnitCatalog, units=(si.m,)):
        span = (2 * si.m).define('span', 'laboratory span')

    assert laboratory.unit('span/m').factor == 2
    assert laboratory.unit('span') == laboratory.span
    with pytest.raises(UnitSymbolError):
        laboratory.unit('s')


def test_registry_prefix_default() -> None:
    registry = Registry(prefixable=False)
    span = registry.define(si.m, 'span')
    assert registry.parse('[span]') == span
    with registry.defining():
        duration = si.s.define('duration')
        enabled = si.s.define('enabled', prefixable='enabled')
    assert registry.parse('kenabled').factor == 1000
    assert registry.parse('duration') == duration
    for text in ('kspan', 'kduration'):
        with pytest.raises(UnitSymbolError):
            registry.parse(text)
    assert enabled.dimension == enabled.dimension.physical


@pytest.mark.parametrize('policy', [None, 1, 'span', ()])
def test_registry_invalid_prefix_default(policy: None | int | str | tuple[object, ...]) -> None:
    with pytest.raises(TypeError, match='default must be bool'):
        Registry(prefixable=policy)


def test_imperial_prefix_permissions() -> None:
    from unitx.imperial import imperial_historical, us

    for catalogue in (imperial, imperial_historical, us):
        for definition in catalogue._registry._bindings.definitions.values():
            # Included SI definitions retain their policy.
            from unitx._registry.reconstruction import _definition_location

            if _definition_location(definition) == (
                'unitx.imperial.units',
                catalogue.__name__,
                definition.symbol,
            ):
                assert definition.prefixable is False
                with pytest.raises(UnitSymbolError):
                    catalogue.unit('k' + definition.symbol)
    assert imperial.unit('point') == imperial.point
    assert imperial_historical.unit('[barleycorn]') == imperial_historical.barleycorn
    assert us.unit('[cord]') == us.cord
    assert (1 * us.us_gal).value_in(si.L) == Fraction('3.785411784')


@pytest.mark.parametrize('symbol', ['m', 'dB', 'Np', 'dBm'])
def test_registry_coordinate_repr_conflicting_text(symbol: str) -> None:
    registry = Registry([si.K])
    with registry.defining():
        coordinate = AffineUnit(10 * si.K).define(symbol)
    point = 3 @ coordinate
    assert eval(repr(coordinate)) == coordinate
    assert eval(repr(point)) == point
    assert point.value_in(si.K) == 13
    assert Unit('m') == si.m
