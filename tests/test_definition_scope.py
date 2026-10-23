"""Definition scopes choose registration targets without changing string parsing."""

import copy
import pickle

import pytest

from unitx import AffineQuantity, AffineUnit, Quantity, Registry, Unit, UnitCatalog
from unitx.exceptions import UnitRegistryError, UnitSymbolError
from unitx.serialization import from_dict, to_dict
from unitx.si import si
from unitx.unitregistry import default_registry


def test_unit_define_without_registration():
    first = (2 * si.m).define('scspan', 'scope span')
    second = (2 * si.m).define('scspan', 'scope span')
    assert first != second and first.factor == second.factor
    assert 1 * first == 1 * second == 2 * si.m
    assert copy.deepcopy(first) is first
    assert pickle.loads(pickle.dumps(first)) == first
    assert eval(repr(first)) == first
    with pytest.raises(UnitSymbolError):
        Unit('scspan')
    receiver = Registry([first])
    assert receiver.parse('scspan') == first
    assert from_dict(to_dict(first), registry=receiver) == first


def test_unit_define_existing_text():
    private = si.s.define('m', 'metre', prefixable=False)
    assert private != si.m
    assert private.dimension == si.s.dimension
    assert Unit('m') == si.m
    assert Registry([private]).parse('metre') == private
    with pytest.raises(UnitRegistryError):
        default_registry.include(private)
    assert Unit('m') == si.m


def test_unit_catalog_definition_identity():
    class example(UnitCatalog, units=(si.s,)):
        span = (2 * si.m).define('scspan', 'scope span')
        speed = (span / si.s).define('scspeed', 'scope speed')
        before = hash(span)
        lookup = {span: 'distance'}

    assert example.lookup[example.parse('scspan')] == 'distance'
    assert hash(example.span) == example.before
    assert example.parse('scspan/s') == example.span / si.s
    assert (1 * example.speed).value_in(si.m / si.s) == 2
    unbound = si.m.define('scafter', 'after scope')
    with pytest.raises(UnitSymbolError):
        example.parse('scafter')
    with pytest.raises(UnitSymbolError):
        Unit('scspan')
    assert 1 * unbound == 1 * si.m
    with pytest.raises(AttributeError):
        example.span = si.m


def test_unit_catalog_nested_failure():
    class outer(UnitCatalog):
        first = si.m.define('scfirst', 'scope first')
        with pytest.raises(RuntimeError, match='construction failed'):

            class invalid(UnitCatalog):
                other = si.s.define('scinner', 'scope inner')
                error_message = 'construction failed'
                raise RuntimeError(error_message)

        last = si.kg.define('sclast', 'scope last')

    assert outer.parse('scfirst') == outer.first
    assert outer.parse('sclast') == outer.last
    with pytest.raises(UnitSymbolError):
        outer.parse('scinner')
    si.m.define('scoutside', 'scope outside')
    for environment in (outer._registry, default_registry):
        with pytest.raises(UnitSymbolError):
            environment.parse('scoutside')
    with pytest.raises(UnitSymbolError):
        Unit('scinner')


def test_unit_catalog_nested_scope():
    class outer(UnitCatalog):
        first = si.m.define('scsame', 'scope same')

        class inner(UnitCatalog):
            duration = si.s.define('scsame', 'scope same')

        last = si.kg.define('sclast', 'scope last')

    assert outer.parse('scsame') == outer.first
    assert outer.inner.parse('scsame') == outer.inner.duration
    assert outer.first != outer.inner.duration
    with pytest.raises(UnitSymbolError):
        outer.inner.parse('sclast')


def test_unit_catalog_explicit_registry_target():
    explicit = Registry()

    class ambient(UnitCatalog):
        one = explicit.define(si.m, 'scone', 'scope one')
        two = si.s.define('sctwo', 'scope two')

    assert explicit.parse('scone') == ambient.one
    assert ambient.parse('sctwo') == ambient.two
    assert ambient.parse('scone') == ambient.one
    with pytest.raises(UnitSymbolError):
        explicit.parse('sctwo')


def test_unit_catalog_parsing_stays_default():
    class binary(UnitCatalog):
        byte = Unit().define('B', 'byte')
        default_bel = Unit('B').dimension
        default_length = Quantity(3, 'm').value_in('cm')

    assert binary.default_bel == si.B.dimension
    assert binary.default_length == 300
    assert binary.parse('B') == binary.byte
    assert binary.byte.dimension == si.one.dimension and si.B.dimension != si.one.dimension


def test_unit_catalog_helper_scope():
    def make(source):
        return source.define('schelper', 'scope helper')

    class example(UnitCatalog):
        length = make(2 * si.m)

    assert example.parse('schelper') == example.length
    outside = make(2 * si.m)
    assert outside != example.length
    assert 1 * outside == 1 * example.length


def test_unit_catalog_shared_registry():
    registry = Registry([si.s])

    class example(UnitCatalog, registry=registry):
        length = si.m.define('scshared', 'scope shared')

    assert registry.parse('scshared/s') == example.length / si.s
    assert example.parse('s') == si.s


def test_unit_catalog_subclass_environment():
    class base(UnitCatalog, units=(si.m,)):
        length = (2 * si.m).define('scspan', 'scope span')

    class child(base):
        duration = si.s.define('sctime', 'scope time')

    # Attribute inheritance does not silently inherit parser bindings.
    assert child.length == base.length
    assert child.parse('sctime') == child.duration
    with pytest.raises(UnitSymbolError):
        child.parse('scspan')
    with pytest.raises(UnitSymbolError):
        child.parse('m')


def test_unit_catalog_assignment_collects_definition():
    class example(UnitCatalog):
        metre = si.m

    assert example.metre == si.m
    assert example.parse('m') == si.m
    with pytest.raises(UnitSymbolError):
        example.parse('metreAttribute')


def test_affine_unit_define_bound_reference():
    kelvin = si.K.define('scK', 'scope kelvin')
    coordinate = AffineUnit(Quantity(10, kelvin), delta_unit=kelvin).define(
        'scT', 'scope temperature'
    )
    assert (3 @ coordinate).value_in(si.K) == 13
    assert (3 * coordinate).value_in(si.K) == 3
    assert AffineQuantity(2, Unit._from_definition(coordinate._definition)).unit == coordinate
    assert eval(repr(coordinate)) == coordinate
    assert pickle.loads(pickle.dumps(coordinate)) == coordinate
    with pytest.raises(UnitSymbolError):
        Unit('scT')
    receiver = Registry([coordinate])
    assert receiver.parse('scK') == kelvin
    assert AffineQuantity(0, receiver.parse('scT')).unit == coordinate
    assert from_dict(to_dict(3 @ coordinate), registry=receiver) == 3 @ coordinate


def test_unit_catalog_affine_coordinate():
    class example(UnitCatalog):
        step = si.K.define('scK', 'scope kelvin')
        temperature = AffineUnit(Quantity(10, step)).define('scT', 'scope temperature')

    assert AffineQuantity(0, example.parse('scT')).value_in(si.K) == 10
    assert from_dict(to_dict(0 @ example.temperature), registry=example._registry) == (
        0 @ example.temperature
    )


def test_registry_include_coordinate_invalid_conflict():
    receiver = Registry([si.s])
    conflicting = si.K.define('s', 'private kelvin', prefixable=False)
    coordinate = AffineUnit(Quantity(2, conflicting)).define('scT', 'scope temperature')
    with pytest.raises(UnitRegistryError):
        receiver.include(coordinate)
    with pytest.raises(UnitSymbolError):
        receiver.parse('scT')
    assert receiver.parse('s') == si.s


def test_unit_catalog_local_recreation_identity():
    def build(source):
        class example(UnitCatalog):
            length = source.define('screpeat')

        return example

    first, second = build(si.m), build(si.m)
    assert first.length != second.length
    assert 1 * first.length == 1 * second.length
    assert first._registry is not second._registry
    third = build(si.s)
    assert third.length.dimension == si.s.dimension
    assert first.parse('screpeat') == first.length


def test_unit_catalog_method_does_not_reactivate():
    class example(UnitCatalog):
        @classmethod
        def make(cls):
            return si.m.define('scmethod', 'scope method')

    length = example.make()
    assert 1 * length == 1 * si.m
    with pytest.raises(UnitSymbolError):
        example.parse('scmethod')


def test_unit_catalog_collects_thread_result():
    from concurrent.futures import ThreadPoolExecutor

    def make():
        return si.s.define('scthread', 'scope thread')

    class example(UnitCatalog):
        with ThreadPoolExecutor(max_workers=1) as _executor:
            duration = _executor.submit(make).result()
        length = si.m.define('sclength', 'scope length')

    assert example.parse('sclength') == example.length
    assert 1 * example.duration == 1 * si.s
    assert example.parse('scthread') == example.duration
    with pytest.raises(UnitSymbolError):
        Unit('scthread')


def test_unit_catalog_shared_registry_failure():
    registry = Registry()
    with pytest.raises(RuntimeError, match='construction failed'):

        class invalid(UnitCatalog, registry=registry):
            length = si.m.define('scbeforefail', 'scope before failure')
            error_message = 'construction failed'
            raise RuntimeError(error_message)

    with pytest.raises(UnitSymbolError):
        registry.parse('scbeforefail')
    si.s.define('scafterfail', 'scope after failure')
    with pytest.raises(UnitSymbolError):
        registry.parse('scafterfail')
    with pytest.raises(UnitSymbolError):
        Unit('scbeforefail')


def test_unit_catalog_preserves_prefix_permissions():
    def make():
        return (2 * si.m).define('helper', prefixable=False)

    class catalogue(UnitCatalog, units=(si.m,)):
        span = si.m.define('span', prefixable=False)
        helper = make()
        enabled = si.m.define('enabled', prefixable=True)
        coordinate = AffineUnit(2 * si.K).define('reading')

    assert catalogue.parse('span') == catalogue.span
    assert catalogue.parse('[span]') == catalogue.span
    assert catalogue.parse('helper') == catalogue.helper
    assert catalogue.parse('kenabled').factor == 1000
    assert catalogue.parse('cm') == si.cm
    assert catalogue.coordinate.name == 'reading'
    for text in ('kspan', 'khelper', 'kreading'):
        with pytest.raises(UnitSymbolError):
            catalogue.parse(text)
    with pytest.raises(UnitSymbolError):
        Registry([make()]).parse('khelper')


def test_unit_catalog_explicit_block_prefix_default():
    target = Registry(prefixable=True)

    class catalogue(UnitCatalog):
        before = si.m.define('before', prefixable=False)
        with target.defining():
            enabled = si.m.define('enabled')
        after = si.m.define('after', prefixable=False)

    assert target.parse('kenabled').factor == 1000
    for text in ('kbefore', 'kafter'):
        with pytest.raises(UnitSymbolError):
            catalogue.parse(text)


@pytest.mark.parametrize('source', [si.dB, si.dB**2, si.dB / si.Np, 2 * si.dB])
def test_unit_define_logarithmic_conversion(source):
    registry = Registry()
    with registry.defining():
        unit = source.define('logStep')
    linear = source.unit if isinstance(source, Quantity) else source
    assert unit.dimension == linear.dimension
    assert unit.factor == linear.factor * (source.value if isinstance(source, Quantity) else 1)
    assert registry.parse('logStep') == unit
    assert (1 * unit).value_in(linear) == (source.value if isinstance(source, Quantity) else 1)
