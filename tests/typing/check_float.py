"""Float scientific formulas with practical float arithmetic inference."""

from decimal import Decimal
from fractions import Fraction
from typing import assert_type

from unitx import Quantity
from unitx._numerics import Real
from unitx.math import sqrt
from unitx.si import SI, si


def speed(distance: Quantity[float], duration: Quantity[float]) -> Quantity[float]:
    return distance / duration


def distance(duration: Quantity[float]) -> Quantity[float]:
    return SI.c * duration


def scaled_length(length: Quantity[float], count: int) -> Quantity[float]:
    return length * count


def duration(length: Quantity[float]) -> Quantity[float]:
    return (length / SI.c).to(si.s).astype(float)


def float_contracts(q: Quantity[float], integer: Quantity[int]) -> None:
    assert_type(q * 2, Quantity[float])
    assert_type(2 * q, Quantity[float])
    assert_type(q * 2.0, Quantity[float])
    assert_type(integer * q, Quantity[float])
    assert_type(q / 2, Quantity[float])
    assert_type(2 / q, Quantity[float])
    assert_type(integer / q, Quantity[float])
    assert_type(q / integer, Quantity[float])
    assert_type(q**2, Quantity[float])
    assert_type(q**-1, Quantity[float])
    assert_type(si.m / q, Quantity[float])
    assert_type(sqrt(q), Quantity[float])
    q *= 2
    q /= 2
    q **= 2
    assert_type(q, Quantity[float])
    assert_type(q.to(si.cm), Quantity[Real])
    assert_type(q + integer, Quantity[float])
    assert_type(q - q, Quantity[float])
    assert_type(q.to(si.cm).astype(float), Quantity[float])
    assert_type(q.astype(int), Quantity[int])
    assert_type(q.astype(Decimal), Quantity[Decimal])
    assert_type(q.astype(Fraction), Quantity[Fraction])
    assert_type(q.astype(complex), Quantity[complex])
    assert_type(SI.e.astype(float), Quantity[float])
    assert_type(SI.re.astype(float), Quantity[float])
    assert_type(Quantity[float](1, 'cm').to(si.m), Quantity[Real])


def add_one(x: Quantity[float]) -> Quantity[float]:
    return 1 + x


def float_addition(q: Quantity[float], integer: Quantity[int]) -> None:
    assert_type(q + q, Quantity[float])
    assert_type(q + 1, Quantity[float])
    assert_type(1 + q, Quantity[float])
    assert_type(integer + q, Quantity[float])
    assert_type(q - 1, Quantity[float])
    assert_type(1 - q, Quantity[float])
    assert_type(integer - q, Quantity[float])
    q += 1
    q -= integer
    assert_type(q, Quantity[float])


def float_decimal(q: Quantity[float], d: Quantity[Decimal]) -> None:
    assert_type(q * SI.e, Quantity[float])
    assert_type(SI.e * q, Quantity[float])
    assert_type(q / SI.e, Quantity[float])
    assert_type(SI.e / q, Quantity[float])
    assert_type(q + d, Quantity[float])
    assert_type(d - q, Quantity[float])
    assert_type(0.5 * SI.e, Quantity[float])
