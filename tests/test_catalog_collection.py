"""Collection preserves immutable definitions and keeps persistence separate."""

from __future__ import annotations

import copy
import os
import pickle
import subprocess
import sys
from pathlib import Path
from uuid import UUID

import pytest

from unitx import AffineUnit, Registry, Unit, UnitCatalog
from unitx.exceptions import UnitRegistryError, UnitSymbolError
from unitx.serialization import from_dict, to_dict
from unitx.si import si
from unitx.unitdefinition import UnitDefinition


@pytest.mark.parametrize('protocol', range(pickle.HIGHEST_PROTOCOL + 1))
def test_unit_definition_pickle(protocol: int) -> None:
    definition = UnitDefinition('span')
    assert isinstance(definition._key, UUID)
    assert pickle.loads(pickle.dumps(definition, protocol=protocol)) is definition
    assert copy.copy(definition) is copy.deepcopy(definition) is definition
    with pytest.raises(AttributeError, match='immutable'):
        definition.__init__('changed')
    assert definition.symbol == 'span'


def test_unit_catalog_collects_final_definitions() -> None:
    def make() -> Unit:
        si.s.define('unused')
        return si.m.define('span', prefixable=False)

    class survey(UnitCatalog):
        span = make()
        _hidden = si.s.define('hidden')
        saved = pickle.dumps(span)
        definition = next(iter(span._monomial.components()))[0]._unit
        initial_key = definition._key
        initial_hash = hash(span)

    assert survey.definition._key is survey.initial_key
    assert hash(survey.span) == survey.initial_hash
    assert pickle.loads(survey.saved) == survey.span
    assert survey.parse('span') == survey.span
    for text in ('unused', 'hidden', 'kspan'):
        with pytest.raises(UnitSymbolError):
            survey.parse(text)
    receiver = Registry([survey.span])
    with pytest.raises(UnitSymbolError):
        receiver.parse('kspan')


def test_unit_catalog_no_immediate_parsing() -> None:
    class survey(UnitCatalog):
        span = si.m.define('collectionSpan')
        doubled = (2 * span).define('collectionDouble')
        with pytest.raises(UnitSymbolError):
            Unit('collectionSpan')

    assert survey.parse('collectionDouble') == survey.doubled
    assert 1 * survey.doubled == 2 * si.m


def test_unit_catalog_shared_registry_updates() -> None:
    registry = Registry([si.m])

    class survey(UnitCatalog, registry=registry):
        span = (2 * si.m).define('span')

    with pytest.raises(AttributeError):
        survey.span = si.m
    with pytest.raises(UnitSymbolError):
        survey.parse('s')
    registry.include(si.s)
    assert survey.parse('span/s') == survey.span / si.s


def test_unit_catalog_json_before_collection() -> None:
    class thermal(UnitCatalog):
        step = si.K.define('collectionStep')
        reading = AffineUnit(10 * step).define('collectionReading')
        length_record = to_dict(step)
        point_record = to_dict(3 @ reading)

    assert from_dict(thermal.length_record, registry=thermal._registry) == thermal.step
    assert from_dict(thermal.point_record, registry=thermal._registry) == 3 @ thermal.reading
    other = Registry()
    with other.defining():
        step = si.K.define('collectionStep')
        AffineUnit(10 * step).define('collectionReading')
    from unitx.exceptions import SerializationError

    with pytest.raises(SerializationError):
        from_dict(thermal.point_record, registry=other)


def test_unit_catalog_invalid_collection_atomic() -> None:
    registry = Registry([si.m])
    with pytest.raises(UnitRegistryError):

        class invalid(UnitCatalog, registry=registry):
            first = si.s.define('collectionFirst')
            conflicting = si.s.define('m')

    assert registry.parse('m') == si.m
    with pytest.raises(UnitSymbolError):
        registry.parse('collectionFirst')


def test_unit_catalog_publication_explicit_units() -> None:
    external = si.s.define('unpublishedStep')
    shared = Registry([external])
    saved = to_dict(external)
    namespace = {
        '__name__': 'explicit_publication_catalogue',
        'UnitCatalog': UnitCatalog,
        'shared': shared,
        'metre': si.m,
    }
    exec('class survey(UnitCatalog, registry=shared):\n    length = metre', namespace)
    survey = namespace['survey']
    assert survey.parse('unpublishedStep') == external
    assert survey.parse('m') == si.m
    assert to_dict(external) == saved


def test_unit_catalog_module_recreation_independent() -> None:
    namespace = {'__name__': 'collection_catalogue', 'UnitCatalog': UnitCatalog, 'source': si.m}
    source = "class survey(UnitCatalog):\n    span = source.define('span')"
    exec(source, namespace)
    first = namespace['survey'].span
    saved = pickle.dumps(first)
    namespace['source'] = si.s
    exec(source, namespace)
    second = namespace['survey'].span
    assert first != second and first.dimension != second.dimension
    assert pickle.loads(saved) == first
    assert namespace['survey'].parse('span') == second


@pytest.mark.parametrize('dependency', ['reference', 'delta', 'nested_reference'])
def test_unit_catalog_pickle_dependency_metadata(tmp_path: Path, dependency: str) -> None:
    module = tmp_path / 'thermal_catalogue.py'
    module.write_text("""
import os
from unitx import AffineUnit, UnitCatalog
from unitx.si import si
scale = int(os.environ['CATALOGUE_SCALE'])
class thermal(UnitCatalog):
    step = (scale * si.K).define('step')
    inner = AffineUnit(10 * step, delta_unit=si.K).define('inner')
    reference = AffineUnit(10 * step, delta_unit=si.K).define('reference')
    delta = AffineUnit(10 * si.K, delta_unit=step).define('delta')
    nested_reference = AffineUnit(0 * inner, delta_unit=si.K).define('nestedReference')
""")
    code = """
import json, os, pickle, sys
from unitx.exceptions import SerializationError, UnitRegistryError
from unitx.serialization import from_dict, to_dict
from thermal_catalogue import thermal
value = 3 @ getattr(thermal, sys.argv[2])
if sys.argv[1] == 'write':
    open('value.pickle', 'wb').write(pickle.dumps(value))
    open('value.json', 'w').write(json.dumps(to_dict(value)))
elif os.environ['CATALOGUE_SCALE'] == '2':
    assert pickle.loads(open('value.pickle', 'rb').read()) == value
    assert from_dict(json.load(open('value.json')), registry=thermal._registry) == value
else:
    readers = [(lambda: pickle.loads(open('value.pickle', 'rb').read()), UnitRegistryError)]
    if sys.argv[2] != 'nested_reference':
        readers.append((
            lambda: from_dict(json.load(open('value.json')), registry=thermal._registry),
            SerializationError,
        ))
    else:
        # Here the nested coordinate is used as a linear difference, so its
        # changed origin does not change the JSON record's physical meaning.
        assert from_dict(json.load(open('value.json')), registry=thermal._registry) == value
    for read, error in readers:
        try:
            read()
        except error:
            pass
        else:
            raise AssertionError('changed dependency accepted')
"""
    for action, scale in [('write', '2'), ('read', '2'), ('read', '3')]:
        subprocess.run(
            [sys.executable, '-c', code, action, dependency],
            cwd=tmp_path,
            env={**os.environ, 'CATALOGUE_SCALE': scale},
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
