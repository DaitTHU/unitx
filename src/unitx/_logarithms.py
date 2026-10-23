"""The two fixed logarithmic conventions; no unit or registry dependencies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal

from ._numerics import is_numpy_object
from ._scaling import scale_value
from .dimension import _LOG_DIMENSION, DIMENSIONLESS
from .exceptions import DimensionError
from .factor import Factor, ln10, one

if TYPE_CHECKING:
    from .unit import Unit

type LogKind = Literal['power', 'amplitude']

_POWER_COEFFICIENT = ln10 / 2
_AMPLITUDE_COEFFICIENT = Factor(ln10)


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
        error_message = 'Logarithmic conversion requires nonnegative physical values.'
        raise ValueError(error_message)
    if isinstance(value, Decimal):
        return Decimal('-Infinity') if value.is_zero() else value.log10()
    return -math.inf if value == 0 else math.log10(value)


def _exp10(value: Any) -> Any:
    if isinstance(value, complex):
        error_message = 'Logarithmic conversion requires real magnitudes.'
        raise TypeError(error_message)
    if is_numpy_object(value):
        from ._numpy import exponential_magnitude

        return exponential_magnitude(value)
    if isinstance(value, Decimal):
        return Decimal(10) ** value
    if isinstance(value, (int, Fraction)) and Fraction(value).denominator == 1:
        n = int(value)
        return 10**n if n >= 0 else Fraction(1, 10**-n)
    return 10.0**value


@dataclass(frozen=True, slots=True)
class _LogConvention:
    """Own one convention's coefficient, numerical functions and reference shifts."""

    kind: LogKind

    @property
    def coefficient(self) -> Factor:
        return _POWER_COEFFICIENT if self.kind == 'power' else _AMPLITUDE_COEFFICIENT

    def to_ratio(self, value: Any, delta: Factor) -> Any:
        return _exp10(scale_value(value, delta / self.coefficient))

    def to_log(self, value: Any, source: Factor, target: Factor) -> Any:
        return scale_value(_log10(scale_value(value, source)), self.coefficient / target)

    def shift(self, ratio: Fraction | float, delta: Factor, *, exact: bool = False) -> Any:
        coefficient = self.coefficient / delta
        exponent = _ten_exponent(ratio) if isinstance(ratio, Fraction) else None
        if exponent is not None:
            if exact and exponent and coefficient.decompose()[1] != one:
                error_message = (
                    'These references require approximate comparison; convert explicitly first.'
                )
                raise TypeError(error_message)
            return exponent * coefficient.value
        if ratio == 1:
            return 0
        if exact:
            error_message = (
                'These references require approximate comparison; convert explicitly first.'
            )
            raise TypeError(error_message)
        return math.log10(ratio) * coefficient.value


_POWER_CONVENTION = _LogConvention('power')
_AMPLITUDE_CONVENTION = _LogConvention('amplitude')


def _log_convention(kind: LogKind) -> _LogConvention:
    if kind == 'power':
        return _POWER_CONVENTION
    if kind == 'amplitude':
        return _AMPLITUDE_CONVENTION
    error_message = "kind must be 'power' or 'amplitude'."
    raise ValueError(error_message)


def to_ratio(value: Any, unit: Unit, kind: LogKind) -> Any:
    if unit.dimension != _LOG_DIMENSION:
        error_message = 'Ratio conversion requires a pure logarithmic difference.'
        raise DimensionError(error_message)
    return _log_convention(kind).to_ratio(value, unit.factor)


def to_log(value: Any, source: Unit, target: Unit, kind: LogKind) -> Any:
    if source.dimension != DIMENSIONLESS:
        error_message = 'Logarithmic conversion requires an ordinary dimensionless ratio.'
        raise DimensionError(error_message)
    if target.dimension != _LOG_DIMENSION:
        error_message = 'Target must be a pure logarithmic difference.'
        raise DimensionError(error_message)
    return _log_convention(kind).to_log(value, source.factor, target.factor)
