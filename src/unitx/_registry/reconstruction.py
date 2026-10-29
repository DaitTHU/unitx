"""Loaded-definition lookup and metadata checks for repr, pickle and JSON.

A runtime handle never merges definitions. A catalogue location is only a
cross-process lookup address, published after the whole class has been collected.
Neither lookup imports modules nor creates or registers definitions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, Literal, NamedTuple
from weakref import WeakKeyDictionary, WeakValueDictionary

from .._numerics import UnfactoredInteger
from ..dimension import Dimension
from ..exceptions import UnitRegistryError
from ..factor import Factor, SymbolicConstant

if TYPE_CHECKING:
    from ..unitdefinition import PrefixFamily, UnitDefinition, UnitTerms

type DefinitionLocation = tuple[str, str, str]  # module, catalogue, canonical symbol
type DefinitionId = int | DefinitionLocation


class _TermState(NamedTuple):
    prefix: str
    key: int
    location: DefinitionLocation | None
    definition: _DefinitionState
    exponent: Fraction


class _DefinitionState(NamedTuple):
    symbols: tuple[str, ...]
    names: tuple[str, ...]
    factors: tuple[tuple[tuple[str, ...], int, int], ...]
    dimension: Dimension
    prefixable: (
        bool | PrefixFamily | tuple[str, ...] | tuple[tuple[str, PrefixFamily | Literal[True]], ...]
    )
    reference: (
        tuple[
            int | float | Fraction | Decimal,
            tuple[tuple[tuple[str, ...], int, int], ...],
            tuple[_TermState, ...],
        ]
        | None
    )
    delta: tuple[tuple[tuple[tuple[str, ...], int, int], ...], tuple[_TermState, ...]] | None
    kind: str | None
    at: Fraction


_LIVE: WeakValueDictionary[int, UnitDefinition] = WeakValueDictionary()
_PUBLISHED: WeakValueDictionary[DefinitionLocation, UnitDefinition] = WeakValueDictionary()
_LOCATIONS: WeakKeyDictionary[UnitDefinition, DefinitionLocation] = WeakKeyDictionary()


def _remember_definition(definition: UnitDefinition) -> None:
    _LIVE[definition._key.int] = definition


def _publish_definitions(
    module: str, catalogue: str, definitions: Iterable[UnitDefinition]
) -> None:
    if '<locals>' in catalogue:
        return
    for definition in definitions:
        if definition not in _LOCATIONS:
            location = (module, catalogue, definition.symbol)
            _LOCATIONS[definition] = location
            _PUBLISHED[location] = definition


def _definition_location(definition: UnitDefinition) -> DefinitionLocation | None:
    return _LOCATIONS.get(definition)


def _definition_id(definition: UnitDefinition) -> DefinitionId:
    return _definition_location(definition) or definition._key.int


def _matches_id(definition: UnitDefinition, identity: DefinitionId) -> bool:
    if isinstance(identity, int):
        return identity == definition._key.int
    return identity == _definition_location(definition)


def _definition_by_id(identity: DefinitionId) -> UnitDefinition:
    definition = _LIVE.get(identity) if isinstance(identity, int) else _PUBLISHED.get(identity)
    if definition is None:
        error_message = f'Definition {identity!r} is unavailable.'
        raise UnitRegistryError(error_message)
    return definition


def _terms_state(terms: UnitTerms) -> tuple[_TermState, ...]:
    return tuple(
        _TermState(
            prefix.symbol,
            definition._key.int,
            _definition_location(definition),
            _definition_state(definition),
            exponent,
        )
        for prefix, definition, exponent in terms
    )


def _definition_state(definition: UnitDefinition) -> _DefinitionState:
    def factor_state(factor: Factor) -> tuple[tuple[tuple[str, ...], int, int], ...]:
        factors: list[tuple[tuple[str, ...], int, int]] = []
        for base, exponent in factor._monomial.components():
            if isinstance(base, SymbolicConstant):
                identity = ('symbolic', base.symbol, base.value.hex())
            elif isinstance(base, UnfactoredInteger):
                identity = ('unfactored', str(base), str(base.limit))
            else:
                identity = ('prime', str(base))
            factors.append((identity, exponent.numerator, exponent.denominator))
        return tuple(sorted(factors))

    from .._coordinates.definition import _CoordinateDefinition
    from .._coordinates.logarithmic import _LogMapping

    coordinate = definition._mapping if isinstance(definition, _CoordinateDefinition) else None
    reference = (
        coordinate.reference
        if isinstance(coordinate, _LogMapping)
        else coordinate.origin
        if coordinate is not None
        else None
    )
    factors = factor_state(definition.factor)
    return _DefinitionState(
        definition.symbols,
        definition.names,
        factors,
        definition.dimension,
        tuple(sorted(definition.prefixable.items()))
        if isinstance(definition.prefixable, Mapping)
        else definition.prefixable,
        (
            reference.value,
            factor_state(reference.unit.scale),
            _terms_state(reference.unit._as_product().terms),
        )
        if reference is not None
        else None,
        (
            factor_state(coordinate.delta_unit.scale),
            _terms_state(coordinate.delta_unit._as_product().terms),
        )
        if coordinate is not None
        else None,
        coordinate.kind if isinstance(coordinate, _LogMapping) else None,
        coordinate.at if isinstance(coordinate, _LogMapping) else Fraction(0),
    )


def _matches_terms(current: tuple[_TermState, ...], saved: tuple[_TermState, ...]) -> bool:
    if len(current) != len(saved):
        return False
    remaining = list(current)
    for term in saved:
        for index, actual in enumerate(remaining):
            same_identity = term.key == actual.key or (
                term.location is not None and term.location == actual.location
            )
            if term.prefix == actual.prefix and term.exponent == actual.exponent and same_identity:
                if not _matches_state(actual.definition, term.definition):
                    return False
                remaining.pop(index)
                break
        else:
            return False
    return True


def _matches_state(current: _DefinitionState, saved: _DefinitionState) -> bool:
    if (
        current.symbols,
        current.names,
        current.factors,
        current.dimension,
        current.prefixable,
        current.kind,
        current.at,
    ) != (
        saved.symbols,
        saved.names,
        saved.factors,
        saved.dimension,
        saved.prefixable,
        saved.kind,
        saved.at,
    ):
        return False
    reference, delta = current.reference, current.delta
    if reference is None:
        return saved.reference is None and saved.delta is None
    if saved.reference is None or saved.delta is None or delta is None:
        return False
    return (
        reference[0] == saved.reference[0]
        and reference[1] == saved.reference[1]
        and _matches_terms(reference[2], saved.reference[2])
        and delta[0] == saved.delta[0]
        and _matches_terms(delta[1], saved.delta[1])
    )


def _restore_definition(
    key: int, location: DefinitionLocation | None, state: _DefinitionState
) -> UnitDefinition:
    definition = _LIVE.get(key)
    if definition is None and location is not None:
        definition = _PUBLISHED.get(location)
    if definition is None or not _matches_state(_definition_state(definition), state):
        error_message = f'Definition {location or key!r} is unavailable or incompatible.'
        raise UnitRegistryError(error_message)
    return definition
