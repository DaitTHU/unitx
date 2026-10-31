"""Half-turn trigonometry with reduction before numerical pi multiplication."""

import math
from decimal import Decimal
from fractions import Fraction
from typing import Any


def scalar_trigpi(value: Any, *, cosine: bool) -> float:
    """Reduce exact scalar periods before rounding the small residual to float."""
    if (isinstance(value, float) and not math.isfinite(value)) or (
        isinstance(value, Decimal) and not value.is_finite()
    ):
        # Match math's NaN and infinity behavior, including signalling Decimal NaN.
        return (math.cos if cosine else math.sin)(value)
    if value == 0:
        return 1.0 if cosine else math.copysign(0.0, value)
    rational = Fraction(value)
    reduced = rational % 2
    quadrant = round(2 * reduced)
    residual = reduced - Fraction(quadrant, 2)
    angle = math.pi * float(residual)
    quadrant %= 4
    if not cosine and rational.denominator == 1:
        return -0.0 if value < 0 else 0.0
    sine, cosine_value = math.sin(angle), math.cos(angle)
    if cosine:
        return (cosine_value, -sine, -cosine_value, sine)[quadrant]
    return (sine, cosine_value, -sine, -cosine_value)[quadrant]


def numpy_trigpi(value: Any, *, cosine: bool) -> Any:
    """Retain floating precision and NumPy error handling after period reduction."""
    import numpy as np

    scalar = isinstance(value, np.generic)
    finite = np.isfinite(value)
    special = None
    if not np.all(finite):
        # Some NumPy remainder kernels do not signal invalid for infinity.
        # The trigonometric kernel owns the special-value/error convention.
        special = (np.cos if cosine else np.sin)(value)
        value = np.where(finite, value, 0)
    # Integer reduction precedes floating conversion, including uint64 > 2**53.
    reduced = np.fmod(value, 2)
    if reduced.dtype.kind in 'iu':
        reduced = reduced.astype(np.float64)
    quadrant = np.rint(2 * reduced)
    residual = reduced - quadrant / 2
    # Compute pi in the working dtype: np.pi carries only float64 precision.
    pi = np.arccos(np.full((), -1, dtype=residual.dtype))
    angle = pi * residual
    sine, cosine_value = np.sin(angle), np.cos(angle)
    phase = np.remainder(quadrant, 4)
    if cosine:
        result = np.where(
            phase == 0,
            cosine_value,
            np.where(phase == 1, -sine, np.where(phase == 2, -cosine_value, sine)),
        )
    else:
        result = np.where(
            phase == 0,
            sine,
            np.where(phase == 1, cosine_value, np.where(phase == 2, -sine, -cosine_value)),
        )
        result = np.where(reduced == np.trunc(reduced), np.copysign(0, reduced), result)
    if special is not None:
        result = np.where(finite, result, special)
    return result[()] if scalar else result
