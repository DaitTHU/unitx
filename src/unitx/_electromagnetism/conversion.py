"""CGS units and explicit SI/Gaussian electromagnetic correspondence.

Gaussian units have mechanical dimensions. Use `gaussian.convert()` to cross
representations; ordinary Quantity conversion remains dimension-strict.
See `docs/catalogues/cgs.md` for kinds, numerical domains and precision.
"""

from __future__ import annotations

# Bound definition identities are used only to recognize documented named routes.
# pyright: reportPrivateUsage=false
import math
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from functools import cache
from typing import cast, overload

from .._numerics import Array, Magnitude, Real, Scalar, is_numpy_object
from .._numerics.conversion import convert
from ..exceptions import DimensionError
from ..quantity import Quantity
from ..si import si
from ..unit import Unit
from ..unitdefinition import UnitDefinition

_lambda = Quantity(math.sqrt(si.em.k1.value), si.em.k1.unit ** Fraction(1, 2))


class _GaussianConversion:
    """Expose the fixed Gaussian boundary without owning catalogue definitions."""

    __slots__ = ()

    @staticmethod
    @overload
    def convert(
        quantity: Quantity[Real], target: Unit, *, kind: str | None = None
    ) -> Quantity[Real]: ...

    @staticmethod
    @overload
    def convert(
        quantity: Quantity[complex], target: Unit, *, kind: str | None = None
    ) -> Quantity[complex]: ...

    @staticmethod
    @overload
    def convert(
        quantity: Quantity[Decimal], target: Unit, *, kind: str | None = None
    ) -> Quantity[Decimal]: ...

    @staticmethod
    @overload
    def convert(
        quantity: Quantity[Array], target: Unit, *, kind: str | None = None
    ) -> Quantity[Array]: ...

    @staticmethod
    def convert(
        quantity: Quantity[Magnitude], target: Unit, *, kind: str | None = None
    ) -> Quantity[Magnitude]:
        """Convert a linear quantity between SI and Gaussian representations.

        Infer a rule only from a documented Gaussian named definition (including
        prefixes and extra scale). Other expressions require an explicit kind.
        With no kind, equal complete dimensions use ordinary linear conversion.
        Cross-representation scaling accepts Python int/float/Fraction/complex
        and supported NumPy magnitudes, producing their native floating results.
        Decimal is supported only by the ordinary same-dimension path.
        """
        if not isinstance(quantity, Quantity) or not isinstance(target, Unit):
            error_message = 'Gaussian conversion requires a linear Quantity and a target Unit.'
            raise TypeError(error_message)
        if kind is not None and not isinstance(kind, str):
            error_message = 'Gaussian conversion kind must be a string or None.'
            raise TypeError(error_message)
        source = quantity.unit
        if kind is None and source.dimension == target.dimension:
            return Quantity(convert(quantity.value, source, target), target)

        rules, named_rules = _gaussian_rules()
        if kind is not None and kind not in rules:
            error_message = (
                f'Unknown Gaussian conversion kind {kind!r}; choose from {tuple(rules)}.'
            )
            raise ValueError(error_message)
        rule = rules[kind] if kind is not None else _named_rule(source, target, named_rules)
        if source.dimension == rule.si.dimension and target.dimension == rule.gaussian.dimension:
            transform = rule.transform
        elif source.dimension == rule.gaussian.dimension and target.dimension == rule.si.dimension:
            transform = Quantity(1.0 / rule.transform.value, rule.transform.unit**-1)
        else:
            error_message = 'Source and target dimensions do not match the selected Gaussian kind.'
            raise DimensionError(error_message)

        value = quantity.value
        if is_numpy_object(value):
            from .._numerics.numpy import validate_magnitude

            validate_magnitude(value)
        elif type(value) not in (int, float, Fraction, complex):
            error_message = (
                'Cross-representation conversion supports Python int, float, Fraction, complex '
                'and plain numeric NumPy magnitudes; Decimal and other domains are unsupported.'
            )
            raise TypeError(error_message)

        # Validate the dimensions first, then combine all unit scales and the
        # approximate physical coefficient before applying one multiplier.
        ratio = source.factor * transform.unit.factor / target.factor
        multiplier = transform.value * float(ratio.value)
        result = cast(Magnitude, cast(Scalar, value) * cast(Scalar, multiplier))
        return Quantity(result, target)


@dataclass(frozen=True, slots=True)
class _Rule:
    si: Unit
    gaussian: Unit
    transform: Quantity[Real]


@cache
def _gaussian_rules() -> tuple[dict[str, _Rule], dict[UnitDefinition, _Rule]]:
    """Derive the fixed table after the public catalogue has been constructed."""
    from ..cgs.units import gaussian

    # One authoritative SI -> Gaussian table; reverse transformations and the
    # named-definition index are derived from it. No graph or symbol matching.
    rules = {
        'charge': _Rule(si.C, gaussian.statC, _lambda),
        'current': _Rule(si.A, gaussian.statA, _lambda),
        'potential': _Rule(si.V, gaussian.statV, 1.0 / _lambda),
        'electric_field': _Rule(si.V / si.m, gaussian.statV / gaussian.cm, 1.0 / _lambda),
        'magnetic_flux_density': _Rule(si.T, gaussian.G, si.em.c / _lambda),
        'magnetic_flux': _Rule(si.Wb, gaussian.Mx, si.em.c / _lambda),
        'capacitance': _Rule(si.F, gaussian.statF, _lambda * _lambda),
        'magnetic_field_strength': _Rule(si.A / si.m, gaussian.Oe, 4 * math.pi * _lambda / si.em.c),
    }
    named_rules: dict[UnitDefinition, _Rule] = {
        terms[0][1]: rule
        for rule in rules.values()
        if len(terms := rule.gaussian._as_product().terms) == 1 and terms[0][2] == 1
    }
    return rules, named_rules


def _named_rule(source: Unit, target: Unit, named_rules: dict[UnitDefinition, _Rule]) -> _Rule:
    for gaussian_unit, si_unit in ((source, target), (target, source)):
        terms = gaussian_unit._as_product().terms
        if len(terms) == 1 and terms[0][2] == 1:
            rule = named_rules.get(terms[0][1])
            if rule is not None and si_unit.dimension == rule.si.dimension:
                return rule
    error_message = (
        'No named Gaussian route matches; specify kind for a supported physical quantity.'
    )
    raise DimensionError(error_message)
