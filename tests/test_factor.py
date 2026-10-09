import pytest
from fractions import Fraction

from unitx.exceptions import IncompleteFactorWarning, InexactFloatWarning
from unitx.factor import Factor, PI, SymbolicNumber


def test_construct_from_integer():
    factor = Factor(12)
    assert str(factor) == '12'
    assert float(factor) == 12.0


def test_construct_from_fraction():
    factor = Factor(Fraction(1, 12))
    assert str(factor) == '1/12'
    assert float(factor) == pytest.approx(1 / 12)


def test_construct_from_float_uses_bounded_fraction():
    with pytest.warns(InexactFloatWarning):
        factor = Factor(2.54)
    assert str(factor) == '127/50'
    assert float(factor) == pytest.approx(2.54)


@pytest.mark.parametrize('value', [0, -1, Fraction(-1, 2)])
def test_factor_must_be_positive(value):
    with pytest.raises(ValueError, match='Factor must be positive'):
        Factor(value)


def test_rejects_unsupported_types():
    with pytest.raises(TypeError, match='Factor must be a number.'):
        Factor('3.14')  # type: ignore[arg-type]


def test_builtin_pi():
    assert str(PI) == 'π'
    assert float(PI) == pytest.approx(3.141592653589793)


def test_symbolic_number():
    e = SymbolicNumber('e', 2.718281828459045)
    assert e.symbol == 'e'
    assert e.value == pytest.approx(2.718281828459045)
    assert str(e) == 'e'
    assert float(e) == pytest.approx(e.value)


def test_invalid_symbolic_number():
    with pytest.raises(ValueError, match='cannot be empty'):
        SymbolicNumber('', 1.0)
    with pytest.raises(ValueError, match='must be positive'):
        SymbolicNumber('x', 0.0)


def test_multiplication_and_division():
    assert str(Factor(12) * Factor(7)) == '84'
    assert str(Factor(12) / 10) == '6/5'
    assert str(10 / Factor(12)) == '5/6'


def test_multiplication_is_commutative_for_numeric_values():
    assert str(Factor(2) * 3) == str(3 * Factor(2)) == '6'


def test_symbolic_arithmetic():
    assert str(2 * PI) == str(PI * 2) == '2⋅π'
    assert str(2 / PI) == '2/π'
    assert str(PI**2) == str(PI * PI) == 'π²'


def test_integer_powers():
    assert str(Factor(12) ** 0) == '1'
    assert str(Factor(12) ** 1) == '12'
    assert str(Factor(12) ** 2) == '144'


def test_fractional_power_preserves_exact_root():
    assert str(Factor(12) ** Fraction(1, 2)) == '2⋅3¹ᐟ²'


def test_cancelling_factors_produces_one():
    assert str(Factor(12) / Factor(12)) == '1'


def test_compact_format():
    assert Factor(12).format() == '12'
    assert Factor(12).format(compact=False) == '2²⋅3'


def test_format_options():
    assert Factor(12).format(mul='*', exp='^', compact=False) == '2^2*3'
    assert Factor(12).format(mul=' ', exp='**', compact=False) == '2**2 3'
    assert Factor(12).format(order='descending', compact=False) == '3⋅2²'


def test_fractional_format():
    assert Factor(Fraction(3, 10)).format() == '3/10'
    assert Factor(Fraction(3, 10)).format(compact=False) == '3/(2⋅5)'


def test_large_integer_factorization():
    factor = Factor(12345678901234567890)
    assert factor.format(compact=False) == '2⋅3²⋅5⋅101⋅3541⋅3607⋅3803⋅27961'


def test_incomplete_fallback_factorization_without_sympy():
    with pytest.warns(IncompleteFactorWarning):
        factor = Factor(1234567890123456789012345678901234567890)
    assert factor.format(compact=False) == \
        '2⋅3²⋅5⋅73⋅101⋅137⋅3541⋅3607⋅3803⋅27961⋅9999000099990001'
