"""Public scalar annotations are usable with optional dependencies absent."""

from typing import assert_type

from unitx import Quantity
from unitx.scipy import jv
from unitx.si import si
from unitx.typing import Array, Magnitude, Scalar


def attach[T: Magnitude](value: T) -> Quantity[T]:
    return Quantity(value, si.m)


def scalar[T: Scalar](q: Quantity[T]) -> T:
    return q.value


def array[T: Array](q: Quantity[T]) -> T:
    return q.value


assert_type(attach(1), Quantity[int])
assert_type(scalar(attach(1.0)), float)


def bessel_without_optional_imports() -> None:
    assert_type(jv(0, 1.0 * si.one), Quantity[Array])
