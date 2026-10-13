import warnings
from decimal import Decimal
from fractions import Fraction

import pytest

import unitx._numerics as numerics_module
from unitx._numerics import as_fraction
from unitx.exceptions import InexactFloatWarning


def test_as_fraction() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error', InexactFloatWarning)
        integer = 42
        assert as_fraction(integer) == Fraction(integer)
        fraction = Fraction(1, 2)
        assert as_fraction(fraction) == fraction
        decimal = Decimal('0.5')
        assert as_fraction(decimal) == fraction
        zero_float = 0.0
        assert as_fraction(zero_float) == 0
        integer_float = 42.0
        assert as_fraction(integer_float) == Fraction(integer_float)
        negative_float = -1.5
        assert as_fraction(negative_float) == Fraction(negative_float)
        large_float = 1e30
        assert as_fraction(large_float) == int(large_float) != 10**30


@pytest.mark.parametrize('value', ['1/2', True, False, None, 1j])
def test_as_fraction_invalid_types(value: object) -> None:
    with pytest.raises(TypeError, match=f'convert {type(value).__name__}'):
        as_fraction(value)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ['value', 'error'],
    [
        (float('nan'), ValueError),
        (float('inf'), OverflowError),
        (float('-inf'), OverflowError),
        (Decimal('NaN'), ValueError),
        (Decimal('Infinity'), OverflowError),
        (Decimal('-Infinity'), OverflowError),
    ],
)
def test_as_fraction_invalid_values(value: float | Decimal, error: type[Exception]) -> None:
    with pytest.raises(error):
        as_fraction(value)


@pytest.mark.parametrize(
    ['value', 'expected'],
    [
        (Decimal('1e-10'), Fraction(1, 10**10)),
        (Decimal('1000000000000000000000000000001'), Fraction(10**30 + 1)),
        (Fraction(1, 1000001), Fraction(1, 1000001)),
    ],
)
def test_as_fraction_exact(value: Decimal | Fraction, expected: Fraction) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error', InexactFloatWarning)
        result = as_fraction(value)
    assert type(result) is Fraction
    assert result == expected


def test_as_fraction_float_literal() -> None:
    float_literal = 1.47
    string_literal = str(float_literal)
    decimal_literal = Decimal(string_literal)  # 1.47
    fraction_literal = Fraction(string_literal)  # 147/100
    with pytest.warns(InexactFloatWarning, match=string_literal):
        result = as_fraction(float_literal)
    assert result == fraction_literal != Fraction(float_literal)
    assert as_fraction(decimal_literal) == fraction_literal


def test_as_fraction_float_max_denominator() -> None:
    with pytest.warns(InexactFloatWarning, match='approximate'):
        too_small = as_fraction(1e-10)
    assert too_small == 0
    with pytest.warns(InexactFloatWarning):
        very_small = as_fraction(1e-10, float_max_denominator=10**12)
    assert very_small == Fraction(1, 10000000000)
    with pytest.raises(ValueError, match='at least 1'):
        as_fraction(1.3, float_max_denominator=-10000)


def test_as_fraction_float_division() -> None:
    with pytest.warns(InexactFloatWarning, match='approximate'):
        result = as_fraction(1 / 3)
    assert result == Fraction(1, 3) != Fraction(1 / 3)


def test_as_fraction_warning_stacklevel() -> None:
    # stacklevel=0 should point inside as_fraction() itself, not the caller.
    with pytest.warns(InexactFloatWarning) as warnings:
        as_fraction(1 / 3, stacklevel=0)
    assert warnings[0].filename == numerics_module.__file__

    def as_fraction_outside_stacklevel(value: float) -> None:
        """stacklevel=2 should point to the caller of this function."""
        as_fraction(value, stacklevel=2)

    with pytest.warns(InexactFloatWarning) as warnings:
        as_fraction_outside_stacklevel(1 / 3)
    assert warnings[0].filename == __file__
