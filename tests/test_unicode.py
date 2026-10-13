from fractions import Fraction

import pytest

from unitx._unicode import superscript


@pytest.mark.parametrize(
    ['value', 'expected'], [(1, ''), (2, '²'), (-1, '⁻¹'), (0, '⁰'), (-102, '⁻¹⁰²')]
)
def test_superscript_integer(value: int, expected: str) -> None:
    assert superscript(value) == expected


@pytest.mark.parametrize(['value', 'expected'], [(1, '¹'), (-3, '⁻³'), (0, '⁰')])
def test_superscript_integer_with_identity(value: int, expected: str) -> None:
    assert superscript(value, omit_identity=False) == expected


@pytest.mark.parametrize(
    ['value', 'expected'],
    [
        (Fraction(1, 2), '¹ᐟ²'),
        (Fraction(-3, 4), '⁻³ᐟ⁴'),
        (Fraction(0, 1), '⁰'),
        (Fraction(3, -40), '⁻³ᐟ⁴⁰'),
        (Fraction(4, 2), '²'),
    ],
)
def test_superscript_fraction(value: Fraction, expected: str) -> None:
    assert superscript(value) == expected


def test_superscript_invalid_types() -> None:
    with pytest.raises(TypeError, match='got bool'):
        superscript(True)
    with pytest.raises(TypeError, match='got str'):
        superscript('1')  # type: ignore[arg-type]
