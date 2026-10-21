"""Finite coordinate conversions; ordinary unit algebra stays multiplicative."""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from typing import Any, cast

from ._numerics import Magnitude, Scalar, is_numpy_object
from .affineunit import AffineUnit, LogKind
from .exceptions import DimensionError
from .factor import Factor, ln10, one
from .unit import Unit


def linear_unit(unit: object) -> Unit:
    """Project a unit descriptor to its linear scale, without applying an origin."""
    if isinstance(unit, Unit):
        return unit
    if isinstance(unit, str):
        return Unit(unit)
    if isinstance(unit, AffineUnit):
        # Retain the registered linear spelling; anonymous coordinates only
        # have their explicit difference unit.
        return Unit(unit.expression()) if unit._definition is not None else unit.delta_unit
    raise TypeError('Expected a Unit, AffineUnit descriptor, or unit spelling.')


def compatible(source: Unit | AffineUnit, target: Unit | AffineUnit) -> bool:
    return (
        not isinstance(source, AffineUnit)
        and not isinstance(target, AffineUnit)
        and source.dimension == target.dimension
        and source.log_degree == target.log_degree
    )


def scale_value(value: Magnitude, factor: Factor) -> Magnitude:
    """Apply a unit ratio in the magnitude's numerical domain."""
    if is_numpy_object(value):
        from ._numpy import scale_magnitude

        return cast(Magnitude, scale_magnitude(value, factor))
    value = cast(Scalar, value)
    if factor == one:
        return value
    if isinstance(value, Decimal):
        if not value.is_finite() or value.is_zero():
            return value * 1
        rational, irrational = factor.decompose()
        scaled = Fraction(value) * rational
        if irrational != one:
            scaled *= Fraction(Decimal(str(irrational.value)))
        # Round the final value once, avoiding intermediate Decimal range loss.
        return Decimal(scaled.numerator) / Decimal(scaled.denominator)
    return cast(Scalar, value * cast(Scalar, factor.value))


def _affine(value: Any, a: Any, b: Any) -> Any:
    if a == 1 and b == 0:
        return value
    if is_numpy_object(value):
        from ._numpy import affine_magnitude

        return affine_magnitude(value, a, b)
    if isinstance(value, Decimal):
        if not value.is_finite():
            return value * 1
        result = Fraction(value) * Fraction(a) + Fraction(b)
        return Decimal(result.numerator) / Decimal(result.denominator)
    return value * a + b


def _origin_in(unit: AffineUnit, target: Unit) -> Any:
    # Cancel common exact factors before numerical evaluation.
    return scale_value(unit.origin.value, unit.origin.unit.factor / target.factor)


def _ten_exponent(ratio: Fraction) -> int | None:
    numerator, denominator = ratio.numerator, ratio.denominator
    exponent = 0
    while numerator > 1 and numerator % 10 == 0:
        numerator //= 10
        exponent += 1
    while denominator > 1 and denominator % 10 == 0:
        denominator //= 10
        exponent -= 1
    return exponent if numerator == denominator == 1 else None


def _log10(value: Any) -> Any:
    if is_numpy_object(value):
        from ._numpy import logarithm_magnitude

        return logarithm_magnitude(value)
    if value < 0:
        raise ValueError('Logarithmic conversion requires nonnegative physical values.')
    if isinstance(value, Decimal):
        return Decimal('-Infinity') if value.is_zero() else value.log10()
    return -math.inf if value == 0 else math.log10(value)


def _exp10(value: Any) -> Any:
    if isinstance(value, complex):
        raise TypeError('Logarithmic conversion requires real magnitudes.')
    if is_numpy_object(value):
        from ._numpy import exponential_magnitude

        return exponential_magnitude(value)
    if isinstance(value, Decimal):
        return Decimal(10) ** value
    if isinstance(value, (int, Fraction)) and Fraction(value).denominator == 1:
        n = int(value)
        return 10**n if n >= 0 else Fraction(1, 10**-n)
    return 10.0**value


def to_ratio(value: Any, unit: Unit, kind: LogKind) -> Any:
    if kind not in ('power', 'amplitude'):
        raise ValueError("kind must be 'power' or 'amplitude'.")
    if not unit.is_dimensionless() or unit.log_degree != 1:
        raise DimensionError('Ratio conversion requires a pure logarithmic difference.')
    return _exp10(scale_value(value, unit.factor / ln10 * (2 if kind == 'power' else 1)))


def to_log(value: Any, source: Unit, target: Unit, kind: LogKind) -> Any:
    if kind not in ('power', 'amplitude'):
        raise ValueError("kind must be 'power' or 'amplitude'.")
    if not source.is_dimensionless() or source.log_degree:
        raise DimensionError('Logarithmic conversion requires an ordinary dimensionless ratio.')
    if not target.is_dimensionless() or target.log_degree != 1:
        raise DimensionError('Target must be a pure logarithmic difference.')
    coefficient = Factor(ln10) / (2 if kind == 'power' else 1) / target.factor
    return scale_value(_log10(scale_value(value, source.factor)), coefficient)


def _log_shift(source: AffineUnit, target: AffineUnit, *, exact: bool = False) -> Any:
    ratio = cast(
        Fraction | float,
        scale_value(
            Fraction(cast(int | float | Fraction | Decimal, source.origin.value))
            / Fraction(cast(int | float | Fraction | Decimal, target.origin.value)),
            source.origin.unit.factor / target.origin.unit.factor,
        ),
    )
    exponent = _ten_exponent(ratio) if isinstance(ratio, Fraction) else None
    coefficient = Factor(ln10) / (2 if target.kind == 'power' else 1) / target.delta_unit.factor
    if exponent is not None:
        if exact and exponent and coefficient.decompose()[1] != one:
            raise TypeError(
                'These references require approximate comparison; convert explicitly first.'
            )
        return exponent * coefficient.value
    if ratio == 1:
        return 0
    if exact:
        raise TypeError(
            'These references require approximate comparison; convert explicitly first.'
        )
    return math.log10(ratio) * coefficient.value


def convert(value: Any, source: Unit | AffineUnit, target: Unit | AffineUnit) -> Any:
    if isinstance(source, AffineUnit) or isinstance(target, AffineUnit):
        if isinstance(value, complex) or (is_numpy_object(value) and value.dtype.kind == 'c'):
            raise TypeError('Affine conversions require real magnitudes.')
    if source == target:
        return value
    if source.dimension != target.dimension:
        raise DimensionError(
            f'Cannot convert quantity with dimension {source.dimension} '
            f'to unit with dimension {target.dimension}.'
        )
    if not isinstance(source, AffineUnit) and not isinstance(target, AffineUnit):
        if not compatible(source, target):
            raise DimensionError('Logarithmic differences require explicit ratio conversion.')
        return scale_value(value, source.factor / target.factor)
    if isinstance(source, AffineUnit) and isinstance(target, AffineUnit):
        if (source.kind is None) != (target.kind is None):
            physical_unit = source.origin.unit
            physical = convert(value, source, physical_unit)
            return convert(physical, physical_unit, target)
        a = source.delta_unit.factor / target.delta_unit.factor
        if source.kind is None:
            b = _origin_in(source, target.delta_unit) - _origin_in(target, target.delta_unit)
        else:
            k_source = 2 if source.kind == 'power' else 1
            k_target = 2 if target.kind == 'power' else 1
            a = a * Fraction(k_source, k_target)
            b = _log_shift(source, target)
        return _affine(value, a.value, b)
    if isinstance(source, AffineUnit):
        assert isinstance(target, Unit)
        if target.log_degree:
            raise DimensionError('Convert a point to a physical quantity, not a difference.')
        if source.kind is None:
            return _affine(
                value,
                (source.delta_unit.factor / target.factor).value,
                _origin_in(source, target),
            )
        ref = source.origin
        assert isinstance(ref.unit, Unit)
        factor = (
            Factor(Fraction(cast(int | float | Fraction | Decimal, ref.value)))
            * ref.unit.factor
            / target.factor
        )
        return scale_value(to_ratio(value, source.delta_unit, source.kind), factor)
    assert isinstance(source, Unit) and isinstance(target, AffineUnit)
    if source.log_degree:
        raise DimensionError('Cannot convert a logarithmic difference into an absolute point.')
    if target.kind is None:
        return _affine(
            value,
            (source.factor / target.delta_unit.factor).value,
            -_origin_in(target, target.delta_unit),
        )
    ref = target.origin
    assert isinstance(ref.unit, Unit)
    ratio = scale_value(
        value,
        source.factor
        / ref.unit.factor
        / Factor(Fraction(cast(int | float | Fraction | Decimal, ref.value))),
    )
    return to_log(ratio, Unit(), target.delta_unit, target.kind)
