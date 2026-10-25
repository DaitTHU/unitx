"""Explicit SI correspondence for the four CGS electromagnetic conventions.

CGS units have mechanical dimensions. Use the catalogue's `convert()` to cross
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


class _EMConversion:
    """Share the fixed conversion boundary without owning catalogue definitions."""

    __slots__ = ()

    @classmethod
    @overload
    def convert(
        cls, quantity: Quantity[Real], target: Unit, *, kind: str | None = None
    ) -> Quantity[Real]: ...

    @classmethod
    @overload
    def convert(
        cls, quantity: Quantity[complex], target: Unit, *, kind: str | None = None
    ) -> Quantity[complex]: ...

    @classmethod
    @overload
    def convert(
        cls, quantity: Quantity[Decimal], target: Unit, *, kind: str | None = None
    ) -> Quantity[Decimal]: ...

    @classmethod
    @overload
    def convert(
        cls, quantity: Quantity[Array], target: Unit, *, kind: str | None = None
    ) -> Quantity[Array]: ...

    @classmethod
    def convert(
        cls, quantity: Quantity[Magnitude], target: Unit, *, kind: str | None = None
    ) -> Quantity[Magnitude]:
        """Convert a linear quantity between SI and this CGS representation.

        Infer a rule only from a documented catalogue definition (including
        prefixes and extra scale). Other expressions require an explicit kind.
        With no kind, equal complete dimensions use ordinary linear conversion.
        Cross-representation scaling accepts Python int/float/Fraction/complex
        and supported NumPy magnitudes, producing their native floating results.
        Decimal is supported only by the ordinary same-dimension path.
        """
        label = {'gaussian': 'Gaussian', 'esu': 'ESU', 'emu': 'EMU', 'hl': 'Heaviside–Lorentz'}.get(
            cls.__name__, 'Electromagnetic'
        )
        if not isinstance(quantity, Quantity) or not isinstance(target, Unit):
            error_message = f'{label} conversion requires a linear Quantity and a target Unit.'
            raise TypeError(error_message)
        if kind is not None and not isinstance(kind, str):
            error_message = f'{label} conversion kind must be a string or None.'
            raise TypeError(error_message)
        source = quantity.unit
        if kind is None and source.dimension == target.dimension:
            return Quantity(convert(quantity.value, source, target), target)

        rules, named_rules = _em_rules(cls)
        if kind is not None and kind not in rules:
            error_message = f'Unknown {label} conversion kind {kind!r}; choose from {tuple(rules)}.'
            raise ValueError(error_message)
        rule = rules[kind] if kind is not None else _named_rule(source, target, named_rules)
        if source.dimension == rule.si.dimension and target.dimension == rule.cgs.dimension:
            transform = rule.transform
        elif source.dimension == rule.cgs.dimension and target.dimension == rule.si.dimension:
            transform = Quantity(1.0 / rule.transform.value, rule.transform.unit**-1)
        else:
            error_message = f'Source and target dimensions do not match the selected {label} kind.'
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
    cgs: Unit
    transform: Quantity[Real]


@cache
def _em_rules(
    catalogue: type[_EMConversion],
) -> tuple[dict[str, _Rule], dict[UnitDefinition, _Rule]]:
    """Derive each fixed table after the public catalogues have been constructed."""
    from ..cgs.units import emu, esu, gaussian, hl

    representatives = {
        gaussian: (
            gaussian.statC,
            gaussian.statA,
            gaussian.statV,
            gaussian.statV / gaussian.cm,
            gaussian.G,
            gaussian.Mx,
            gaussian.statF,
            gaussian.Oe,
        ),
        esu: (
            esu.statC,
            esu.statA,
            esu.statV,
            esu.statV / esu.cm,
            esu.magnetic_flux_density,
            esu.magnetic_flux,
            esu.statF,
            esu.magnetic_field_strength,
        ),
        emu: (
            emu.abC,
            emu.abA,
            emu.abV,
            emu.abV / emu.cm,
            emu.G,
            emu.Mx,
            emu.abF,
            emu.Oe,
        ),
        hl: (
            hl.charge,
            hl.current,
            hl.potential,
            hl.electric_field,
            hl.magnetic_flux_density,
            hl.magnetic_flux,
            hl.capacitance,
            hl.magnetic_field_strength,
        ),
    }
    catalogue = next(base for base in catalogue.__mro__ if base in representatives)
    coefficients = {gaussian: gaussian.em, esu: esu.em, emu: emu.em, hl: hl.em}[catalogue]
    # Coulomb and Lorentz forces determine charge, potential and B mappings.
    # H uses the unrationalized 4*pi current convention, except in HL.
    ratio = si.em.k1 / coefficients.k1
    charge = Quantity(math.sqrt(ratio.value), ratio.unit ** Fraction(1, 2))
    potential = 1.0 / charge
    magnetic = 1.0 / (charge * coefficients.alpha)
    strength = charge * coefficients.alpha * (1.0 if catalogue is hl else 4 * math.pi)
    q, i, v, e, b, flux, capacitance, h = representatives[catalogue]
    rules = {
        'charge': _Rule(si.C, q, charge),
        'current': _Rule(si.A, i, charge),
        'potential': _Rule(si.V, v, potential),
        'electric_field': _Rule(si.V / si.m, e, potential),
        'magnetic_flux_density': _Rule(si.T, b, magnetic),
        'magnetic_flux': _Rule(si.Wb, flux, magnetic),
        'capacitance': _Rule(si.F, capacitance, charge * charge),
        'magnetic_field_strength': _Rule(si.A / si.m, h, strength),
    }
    named_rules: dict[UnitDefinition, _Rule] = {
        terms[0][1]: rule
        for rule in rules.values()
        if len(terms := rule.cgs._as_product().terms) == 1 and terms[0][2] == 1
    }
    return rules, named_rules


def _named_rule(source: Unit, target: Unit, named_rules: dict[UnitDefinition, _Rule]) -> _Rule:
    for cgs_unit, si_unit in ((source, target), (target, source)):
        terms = cgs_unit._as_product().terms
        if len(terms) == 1 and terms[0][2] == 1:
            rule = named_rules.get(terms[0][1])
            if rule is not None and si_unit.dimension == rule.si.dimension:
                return rule
    error_message = (
        'No named catalogue route matches; specify kind for a supported physical quantity.'
    )
    raise DimensionError(error_message)
