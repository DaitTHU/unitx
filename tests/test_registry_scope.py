"""Explicit definition blocks coexist with catalogue class scopes."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from unitx import AffineQuantity, AffineUnit, Quantity, Registry, Unit, UnitCatalog
from unitx.exceptions import UnitRegistryError, UnitSymbolError
from unitx.serialization import from_dict, to_dict
from unitx.si import si


def test_registry_defining_units() -> None:
    registry = Registry([si.m, si.s])
    with registry.defining() as selected:
        assert selected is registry
        span = (Decimal('0.5') * si.m).define('wspan', 'with span', 'with distance')
        speed = (span / si.s).define('wspeed', 'with speed')
        before = hash(span)
        lookup = {span: 'distance'}
    assert registry.parse('[with distance]/s') == span / si.s
    assert registry.parse('kwspan') == registry.parse('kilowith span')
    assert (3 * speed).value_in(si.m / si.s) == Decimal('1.5')
    assert hash(span) == before and lookup[registry.parse('wspan')] == 'distance'
    assert from_dict(to_dict(3 * span), registry=registry) == 3 * span
    with pytest.raises(UnitSymbolError):
        Unit('wspan')
    outside = si.m.define('woutside', 'with outside')
    assert 1 * outside == 1 * si.m
    with pytest.raises(UnitSymbolError):
        registry.parse('woutside')


def test_registry_defining_nested_failure() -> None:
    outer, inner = Registry(), Registry()
    with outer.defining():
        first = si.m.define('wfirst', 'with first')
        with pytest.raises(RuntimeError, match='definition failed'):
            with inner.defining():
                other = si.s.define('winner', 'with inner')
                error_message = 'definition failed'
                raise RuntimeError(error_message)
        last = si.kg.define('wlast', 'with last')
    assert outer.parse('wfirst') == first
    assert outer.parse('wlast') == last
    # A scope chooses a target, rather than rolling back completed bindings.
    assert inner.parse('winner') == other
    with pytest.raises(UnitSymbolError):
        outer.parse('winner')
    with pytest.raises(UnitSymbolError):
        inner.parse('wlast')
    si.m.define('wafter', 'with after')
    for registry in (outer, inner):
        with pytest.raises(UnitSymbolError):
            registry.parse('wafter')


def test_registry_defining_helper() -> None:
    def make() -> Unit:
        return (2 * si.s).define('wtime', 'with time')

    registry = Registry()
    with registry.defining():
        duration = make()
    assert registry.parse('wtime') == duration
    assert make() != duration


def test_registry_defining_affine_unit() -> None:
    registry = Registry()
    with registry.defining():
        step = si.K.define('wK', 'with kelvin')
        temperature = AffineUnit(10 * step).define('wT', 'with temperature')
    assert (3 @ temperature).value_in(si.K) == 13
    assert AffineQuantity(3, registry.parse('wT')).unit == temperature
    assert from_dict(to_dict(3 @ temperature), registry=registry) == 3 @ temperature


def test_registry_defining_parsing_stays_default() -> None:
    registry = Registry()
    with registry.defining():
        byte = Unit().define('B', 'byte')
        assert registry.parse('B') == byte
        assert Unit('B') == si.B
        assert Quantity(3, 'm').value_in('cm') == 300
        # A target string never acquires the selected definition scope.
        with pytest.raises(UnitSymbolError):
            (1 * byte).to('byte')
    assert byte.dimension == si.one.dimension and si.B.dimension != si.one.dimension


def test_registry_defining_invalid_definition() -> None:
    registry = Registry([si.m])
    with registry.defining():
        with pytest.raises(UnitRegistryError):
            si.s.define('m', 'with conflicting metre')
        length = si.m.define('wvalid', 'with valid')
    assert registry.parse('m') == si.m
    assert registry.parse('wvalid') == length
    with pytest.raises(UnitSymbolError):
        registry.parse('[with conflicting metre]')


def test_registry_defining_explicit_target() -> None:
    selected = Registry()
    explicit = Registry()
    with selected.defining():
        one = explicit.define(si.m, 'wone', 'with one')
        two = si.s.define('wtwo', 'with two')
    assert explicit.parse('wone') == one
    assert selected.parse('wtwo') == two
    with pytest.raises(UnitSymbolError):
        selected.parse('wone')
    with pytest.raises(UnitSymbolError):
        explicit.parse('wtwo')


def test_registry_defining_catalogue_nesting() -> None:
    outer = Registry()
    inner = Registry()
    with outer.defining():
        before = si.m.define('wbefore', 'with before')

        class catalogue(UnitCatalog):
            first = si.m.define('wfirst', 'with first')
            with inner.defining():
                duration = si.s.define('wtime', 'with time')

                class nested(UnitCatalog):
                    mass = si.kg.define('wmass', 'with mass')

                after_nested = si.A.define('wcurrent', 'with current')
            last = si.K.define('wlast', 'with last')

        after = si.m.define('wafter', 'with after')
    assert outer.parse('wbefore') == before
    assert outer.parse('wafter') == after
    assert catalogue.parse('wfirst') == catalogue.first
    assert catalogue.parse('wlast') == catalogue.last
    assert inner.parse('wtime') == catalogue.duration
    assert inner.parse('wcurrent') == catalogue.after_nested
    assert catalogue.nested.parse('wmass') == catalogue.nested.mass
    assert outer.parse('wfirst') == catalogue.first
    assert inner.parse('wmass') == catalogue.nested.mass
    assert catalogue.parse('wtime') == catalogue.duration
    with pytest.raises(UnitSymbolError):
        catalogue.parse('wmass')


def test_registry_defining_catalogue_failure() -> None:
    registry = Registry()
    with registry.defining():
        with pytest.raises(RuntimeError, match='catalogue failed'):

            class invalid(UnitCatalog):
                length = si.m.define('wfailed', 'with failed')
                error_message = 'catalogue failed'
                raise RuntimeError(error_message)

        duration = si.s.define('wafter', 'with after')
    assert registry.parse('wafter') == duration
    assert (1 * registry.parse('wfailed')).value_in(si.m) == 1


def test_registry_defining_async_isolation() -> None:
    async def exercise() -> list[tuple[Registry, Unit]]:
        ready = [asyncio.Event(), asyncio.Event()]

        async def worker(index: int, source: Unit) -> tuple[Registry, Unit]:
            registry = Registry()
            with registry.defining():
                ready[index].set()
                await ready[1 - index].wait()
                result = source.define('wshared', 'with shared')
                await asyncio.sleep(0)
                assert registry.parse('wshared') == result
            return registry, result

        return await asyncio.gather(worker(0, si.m), worker(1, si.s))

    (left, length), (right, duration) = asyncio.run(exercise())
    assert left.parse('wshared') == length
    assert right.parse('wshared') == duration
    assert length != duration and length.dimension != duration.dimension
    with pytest.raises(UnitSymbolError):
        Unit('wshared')


def test_registry_defining_async_inheritance() -> None:
    async def make() -> Unit:
        await asyncio.sleep(0)
        return si.m.define('wtask', 'with task')

    registry = Registry()

    async def exercise() -> Unit:
        with registry.defining():
            task = asyncio.create_task(make())
        # Task context is copied at creation, even when it outlives the block.
        return await task

    result = asyncio.run(exercise())
    assert registry.parse('wtask') == result
    assert asyncio.run(make()) != result


def test_registry_defining_failure_restores_catalogue() -> None:
    registry = Registry()

    class catalogue(UnitCatalog):
        with pytest.raises(RuntimeError, match='block failed'):
            with registry.defining():
                duration = si.s.define('wtime', 'with time')
                error_message = 'block failed'
                raise RuntimeError(error_message)
        length = si.m.define('wlength', 'with length')

    assert registry.parse('wtime') == catalogue.duration
    assert catalogue.parse('wlength') == catalogue.length
    assert catalogue.parse('wtime') == catalogue.duration
    with pytest.raises(UnitSymbolError):
        registry.parse('wlength')
