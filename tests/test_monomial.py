from fractions import Fraction

import pytest

from unitx.exceptions import InexactFloatWarning
from unitx.monomial import Monomial


def test_creation_and_basic_mapping_behaviour() -> None:
    m = Monomial({'x': 2, 'y': 3, 'zero': 0})
    assert m == Monomial({'x': 2, 'y': 3})
    assert len(m) == 2
    assert 'x' in m and 'missing' not in m
    assert m['missing'] == 0
    assert set(m.bases()) == {'x', 'y'}
    assert set(m.exponents()) == {Fraction(2), Fraction(3)}
    assert set(m.components()) == {('x', Fraction(2)), ('y', Fraction(3))}
    assert repr(m) == "Monomial({'x': 2, 'y': 3})"
    assert str(m) == 'x²⋅y³'


def test_empty_monomial_formats_as_one() -> None:
    m = Monomial()
    assert str(m) == '1'
    assert m.format(frac=False) == '1'
    assert m.numerator == Monomial()
    assert m.denominator == Monomial()


def test_format_exponent_styles() -> None:
    m = Monomial({'x': 2, 'y': Fraction(-3, 2)})
    assert m.format(exp='sup') == 'x²/y³ᐟ²'
    assert m.format(exp='^') == 'x^2/y^(3/2)'
    assert m.format(exp='**') == 'x**2/y**(3/2)'
    assert m.format(exp='^', frac=False) == 'x^2⋅y^(-3/2)'


def test_format_multiplication_styles() -> None:
    m = Monomial({'x': 1, 'y': 1, 'z': -1})
    assert m.format(mul='⋅') == 'x⋅y/z'
    assert m.format(mul='*') == 'x*y/z'
    assert m.format(mul=' * ') == 'x * y / z'
    assert m.format(mul=' ') == 'x y/z'


def test_format_fraction_and_negative_exponents() -> None:
    m = Monomial({'x': 2, 'y': -1, 'z': -3})
    assert m.format() == 'x²/(y⋅z³)'
    assert m.format(frac=False) == 'x²⋅y⁻¹⋅z⁻³'
    assert m.numerator == Monomial({'x': 2})
    assert m.denominator == Monomial({'y': 1, 'z': 3})


def test_format_order_uses_comparison_when_keys_are_comparable() -> None:
    m = Monomial({3: 1, 1: 1, 2: 1})
    assert m.format(order='insertion', frac=False) == '3⋅1⋅2'
    assert m.format(order='ascending', frac=False) == '1⋅2⋅3'
    assert m.format(order='descending', frac=False) == '3⋅2⋅1'


def test_format_order_falls_back_for_incomparable_keys() -> None:
    m = Monomial({'b': 1, 2: 1, 'a': 1, 1: 1})
    assert m.format(order='ascending', frac=False) == '1⋅2⋅a⋅b'
    assert m.format(order='descending', frac=False) == 'b⋅a⋅2⋅1'


def test_format_rejects_invalid_options() -> None:
    m = Monomial({'x': 1})
    with pytest.raises(ValueError, match='mul'):
        m.format(mul='.')  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='exp'):
        m.format(exp='plain')  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='order'):
        m.format(order='random')  # type: ignore[arg-type]


def test_zero_assignment_and_copy_are_canonical() -> None:
    m = Monomial({'x': 2})
    m_copy = m.copy()
    m['x'] = 0
    assert m == Monomial()
    assert m_copy == Monomial({'x': 2})


def test_monomial_arithmetic_and_inverse() -> None:
    first = Monomial({'x': 2, 'y': 3})
    second = Monomial({'x': 1, 'z': 4})
    assert first * second == Monomial({'x': 3, 'y': 3, 'z': 4})
    assert first / second == Monomial({'x': 1, 'y': 3, 'z': -4})
    assert first**2 == Monomial({'x': 4, 'y': 6})
    assert 1 / first == Monomial({'x': -2, 'y': -3})
    assert str(1 / first) == '1/(x²⋅y³)'


def test_invalid_creation_and_operations() -> None:
    with pytest.raises(TypeError):
        Monomial({'x': 2, 'y': None})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Monomial({'x', 'y'})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Monomial({'x': 2}) * 2  # type: ignore[operator]
    with pytest.raises(TypeError):
        Monomial({'x': 2}) / 0  # type: ignore[operator]


def test_float_conversion_warning() -> None:
    with pytest.warns(InexactFloatWarning):
        m = Monomial({'x': 1/7})  # type: ignore[arg-type]
    assert repr(m) == "Monomial({'x': 1/7})"
    with pytest.warns(InexactFloatWarning):
        m **= 1 / 3
    with pytest.warns(InexactFloatWarning):
        m['x'] = 0.2
