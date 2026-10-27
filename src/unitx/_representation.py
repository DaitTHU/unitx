"""Select explicit unit representations without changing unit identities."""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from typing import cast

from ._numerics import Magnitude, is_numpy_object
from ._numerics.scaling import scale_value
from ._registry.builtin_units import _UNIT_ENTRIES
from ._registry.prefixes import NO_PREFIX, PREFIX_DEFINITIONS
from .dimension import DIMENSIONLESS, Dimension
from .exceptions import DimensionError
from .monomial import Monomial
from .singleunit import SingleUnit
from .unit import Unit


def base_unit(unit: Unit) -> Unit:
    """Select the scale-one SI base product for a complete physical dimension."""
    dimension = unit.dimension
    if dimension != dimension.physical:
        error_message = (
            'SI base units cannot express a nonzero logarithmic degree; '
            'use an explicit ratio or physical-quantity conversion first.'
        )
        raise DimensionError(error_message)
    # Root definitions follow the same T, L, M, I, Theta, N, J order as Dimension.
    # The mass root is gram, so its SI base spelling needs the kilo prefix.
    terms = {
        SingleUnit._from_definitions(
            PREFIX_DEFINITIONS['k'] if index == 2 else NO_PREFIX, definition
        ): exponent
        for index, (definition, exponent) in enumerate(
            zip(_UNIT_ENTRIES[1:8], dimension.as_tuple(), strict=True)
        )
        if exponent
    }
    return Unit._from_monomial(Monomial(terms))


def simplified_unit(unit: Unit) -> Unit:
    """Merge like-dimensional factors, retaining the explicit extra scale."""
    groups: dict[Dimension, list[tuple[SingleUnit, Fraction]]] = {}
    for base, exponent in unit._monomial.components():
        if base.dimension != DIMENSIONLESS:
            groups.setdefault(base.dimension, []).append((base, exponent))
    terms: dict[SingleUnit, Fraction] = {}
    for group in groups.values():
        representative = min(
            (base for base, _ in group),
            key=lambda base: (base.has_prefix(), base.symbol, base._unit._key.int),
        )
        exponent = sum((power for _, power in group), Fraction())
        if exponent:
            terms[representative] = exponent
    return Unit._from_monomial(Monomial(terms), scale=unit.scale)


def _peak_magnitude(value: Magnitude) -> Fraction | None:
    """Find a finite nonzero real scale without float coercion or array writes."""
    if is_numpy_object(value):
        import numpy as np

        array = np.asarray(value)
        if array.dtype.kind == 'c':
            error_message = 'compact requires real magnitudes.'
            raise TypeError(error_message)
        finite = array[np.isfinite(array)]
        if not finite.size:
            return None
        if array.dtype.kind in 'iu':
            peak = max(abs(int(finite.min())), abs(int(finite.max())))
            return Fraction(peak) if peak else None
        peak_float = max(abs(finite.min()), abs(finite.max()))
        if not peak_float:
            return None
        numerator, denominator = peak_float.as_integer_ratio()
        return Fraction(int(numerator), int(denominator))
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction, Decimal)):
        error_message = 'compact requires a real Python scalar or numeric NumPy magnitude.'
        raise TypeError(error_message)
    if isinstance(value, Decimal) and not value.is_finite():
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    peak = abs(Fraction(value))
    return peak if peak else None


def compact_unit(unit: Unit, value: Magnitude) -> Unit:
    """Choose a single-factor prefix using the finite peak in stored magnitudes."""
    peak = _peak_magnitude(value)
    components = tuple(unit._monomial.components())
    if peak is None or len(components) != 1:
        return unit
    base, exponent = components[0]
    if not any(base._unit.allows_prefix(text) for text in (*base._unit.symbols, *base._unit.names)):
        return unit
    best_unit = unit
    best_score: tuple[Fraction | float, bool, Fraction | float, str] | None = None
    for prefix in (NO_PREFIX, *PREFIX_DEFINITIONS.values()):
        candidate = Unit._from_monomial(
            Monomial({SingleUnit._from_definitions(prefix, base._unit): exponent})
        )
        try:
            scaled = cast(Fraction | float, scale_value(peak, unit.factor / candidate.factor))
        except OverflowError:
            # The display choice need not force an overflowing approximate conversion.
            continue
        if scaled <= 0 or (isinstance(scaled, float) and not math.isfinite(scaled)):
            continue
        distance = 0 if 1 <= scaled < 1000 else 1 / scaled if scaled < 1 else scaled / 1000
        score = (distance, prefix.symbol in ('h', 'da', 'd', 'c'), scaled, prefix.symbol)
        if best_score is None or score < best_score:
            best_unit, best_score = candidate, score
    return best_unit
