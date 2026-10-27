"""Magnitude-specific result types for explicit representation APIs."""

# pyright: reportUnnecessaryTypeIgnoreComment=true

import math
from decimal import Decimal
from fractions import Fraction
from typing import assert_type

from unitx import AffineQuantity, AffineUnit, LogQuantity, LogUnit, Quantity, Unit
from unitx._numerics import Number, Real
from unitx.si import SI, si

assert_type(Quantity(Decimal('25'), si.percent).n, Decimal)
assert_type(Quantity(Fraction(1), si.percent).n, Fraction | float)
assert_type(Quantity(1.0, si.percent).n, float)
assert_type(Quantity(1, si.percent).n, Real)
assert_type(Quantity(1j, si.percent).n, Number)
assert_type(Quantity[float](1, si.percent).n, float)
assert_type(Quantity[complex](1, si.percent).n, Number)
assert_type(Quantity.n, property)


def check_n_real(value: Real) -> None:
    assert_type(Quantity(value, si.percent).n, Real)


def electron_beta(energy: Quantity[float]) -> float:
    ratio = energy / (SI.me * SI.c**2)
    assert_type(ratio, Quantity[float])
    gamma = ratio.n
    assert_type(gamma, float)
    beta = math.sqrt(1 - 1 / gamma**2)
    assert_type(beta, float)
    return beta


def check_n_read_only(quantity: Quantity[float]) -> None:
    # An unnecessary-ignore error would mean writes are no longer rejected.
    quantity.n = 1.0  # pyright: ignore[reportAttributeAccessIssue]


assert_type(Quantity(Decimal('25'), si.cm).to_base(), Quantity[Decimal])
assert_type(Quantity(Fraction(1), si.cm).to_base(), Quantity[Fraction | float])
assert_type(Quantity(1.0, si.cm).to_base(), Quantity[Real])
assert_type(Quantity(1j, si.cm).to_base(), Quantity[complex])
assert_type(Quantity(Decimal('25'), si.cm / si.m).simplify(), Quantity[Decimal])
assert_type(Quantity(Fraction(1), si.cm).simplify(), Quantity[Fraction | float])
assert_type(Quantity(1.0, si.cm).simplify(), Quantity[Real])
assert_type(Quantity(1j, si.cm).simplify(), Quantity[complex])
assert_type(Quantity(Decimal('25'), si.cm).compact(), Quantity[Decimal])
assert_type(Quantity(Fraction(1), si.cm).compact(), Quantity[Fraction | float])
assert_type(Quantity(1.0, si.cm).compact(), Quantity[Real])
assert_type(Quantity(Decimal('25'), si.cm).as_pair(), tuple[Decimal, Unit])
assert_type(AffineQuantity(20, si.degC).as_pair(), tuple[int, AffineUnit])
assert_type(LogQuantity(30, si.dBm).as_pair(), tuple[int, LogUnit])
