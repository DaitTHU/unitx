"""Shared attribute protection must preserve the compact instance layout."""

import copy
import pickle

import pytest

from unitx.dimension import Dimension
from unitx.factor import Factor, pi
from unitx.monomial import Monomial
from unitx.si import si
from unitx.singleunit import SingleUnit
from unitx.unit import ScaledUnit, Unit
from unitx.unitregistry import NO_PREFIX, PREFIX_DEFINITIONS, UNIT_DEFINITIONS


@pytest.mark.parametrize(
    'value',
    [
        Unit('m'),
        ScaledUnit('m', Factor(2)),
        si.degC,
        Monomial({'x': 1}),
        Dimension(L=1),
        pi,
        SingleUnit('km'),
        NO_PREFIX,
        PREFIX_DEFINITIONS['k'],
        UNIT_DEFINITIONS['m'],
    ],
)
def test_immutable_slots(value: object) -> None:
    assert not hasattr(value, '__dict__')


@pytest.mark.parametrize('unit', [Unit('m/s'), ScaledUnit('m/s', Factor(2)), si.degC])
def test_unit_immutable_state(unit):
    mapping = {unit: 'value'}
    for name in (
        '_factor',
        '_dimension',
        '_symbol',
        '_monomial',
        '_log_degree',
        '_scale',
        '_origin',
        '_kind',
        '_delta_unit',
        '_definition',
    ):
        with pytest.raises(AttributeError, match='immutable'):
            setattr(unit, name, None)
        with pytest.raises(AttributeError, match='immutable'):
            delattr(unit, name)
    assert copy.copy(unit) is unit
    assert copy.deepcopy(unit) is unit
    for protocol in range(2, pickle.HIGHEST_PROTOCOL + 1):
        rebuilt = pickle.loads(pickle.dumps(unit, protocol=protocol))
        assert mapping[rebuilt] == 'value'
        assert hash(rebuilt) == hash(unit)
        assert rebuilt.dimension == unit.dimension
    if not isinstance(unit, type(si.degC)):
        assert (unit * Unit('s')) / Unit('s') == unit
