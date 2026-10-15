"""Tests for the Factor and SymbolicConstant class"""

import math
import operator
import warnings
from collections.abc import Callable
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import NoReturn

import pytest

import unitx._numerics as numerics_module
import unitx.factor as factor_module
from unitx._numerics import Fractionable, UnfactoredInteger
from unitx.exceptions import IncompleteFactorWarning, InexactFloatWarning
from unitx.factor import ONE, PI, TEN, TWO, Factor, SymbolicConstant


@pytest.fixture
def fallback_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the built-in backend even when SymPy is installed."""
    monkeypatch.setattr(factor_module, 'factorint', numerics_module.factorint)
    monkeypatch.setattr(factor_module, 'isprime', numerics_module.isprime)


# SymbolicConstant


def test_symbolic_constant_creation() -> None:
    e = SymbolicConstant('e', math.e)
    assert e.symbol == 'e'
    assert e.value == math.e
    assert str(e) == 'e'
    assert float(e) == pytest.approx(e.value)


@pytest.mark.parametrize('symbol', [None, 12])
def test_symbolic_constant_invalid_symbol_types(symbol: object) -> None:
    with pytest.raises(TypeError, match='Symbol must be str'):
        SymbolicConstant(symbol, math.pi)  # pyright: ignore[reportArgumentType]


def test_symbolic_constant_invalid_empty_symbol() -> None:
    with pytest.raises(ValueError, match='cannot be empty'):
        SymbolicConstant('', math.pi)


@pytest.mark.parametrize(
    'symbol', ['0', '123', '3.14', '.5', '3.', '-2', '+2', '1e3', '1E-3', '²', '½', '１２.５']
)
def test_symbolic_constant_invalid_numeric_symbols(symbol: str) -> None:
    with pytest.raises(ValueError, match='cannot be numeric'):
        SymbolicConstant(symbol, math.pi)


@pytest.mark.parametrize('symbol', ['\x00', '\u200b', '\t', '\n'])
def test_symbolic_constant_invalid_nonprintable_symbols(symbol: str) -> None:
    with pytest.raises(ValueError, match='must contain only printable characters'):
        SymbolicConstant(symbol, math.pi)


@pytest.mark.parametrize(
    'symbol', [' ', '  ', ' x', 'x ', 'x y'] + [f'a{char}b' for char in '+-−*/×÷⋅·∗⁄∕^()[]{}=']
)
def test_symbolic_constant_invalid_characters(symbol: str) -> None:
    with pytest.raises(ValueError, match='whitespace or operators/delimiters'):
        SymbolicConstant(symbol, math.pi)


@pytest.mark.parametrize('value', [1, True, Fraction(1), Decimal('1')])
def test_symbolic_constant_invalid_value_types(value: int | Fraction | Decimal) -> None:
    with pytest.raises(TypeError, match='Value must be float'):
        SymbolicConstant('x', value)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize('value', [0.0, -1.0, float('nan'), float('inf'), float('-inf')])
def test_symbolic_constant_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match='must be finite and positive'):
        SymbolicConstant('x', value)


def test_symbolic_constant_hash() -> None:
    same_pi = SymbolicConstant('π', math.pi)
    other_constant = SymbolicConstant('e', math.e)
    assert PI == same_pi
    assert hash(PI) == hash(same_pi)
    assert {PI: 'pi'}[same_pi] == 'pi'
    assert len({PI, same_pi, other_constant}) == 2
    assert PI != other_constant
    assert PI != math.pi
    assert math.pi != PI
    assert math.pi not in {PI: 'pi'}
    assert other_constant != math.e


@pytest.mark.parametrize('attribute', ['symbol', 'value', '_symbol', '_value'])
def test_symbolic_constant_immutable(attribute: str) -> None:
    constant = SymbolicConstant('e', math.e)
    factor = Factor(constant)
    original_hash = hash(constant)
    lookup = {constant: 'unchanged'}
    with pytest.raises(AttributeError, match='immutable'):
        setattr(constant, attribute, 3.0)
    with pytest.raises(AttributeError, match='immutable'):
        delattr(constant, attribute)
    assert hash(constant) == original_hash
    assert lookup[constant] == 'unchanged'
    assert factor.value == (factor * ONE).value == math.e
    assert factor / constant == ONE


def test_symbolic_constant_copy_and_pickle() -> None:
    import pickle
    from copy import copy, deepcopy

    constant = SymbolicConstant('e', math.e)
    for rebuilt in (copy(constant), deepcopy(constant), pickle.loads(pickle.dumps(constant))):
        assert rebuilt == constant
        assert hash(rebuilt) == hash(constant)
        with pytest.raises(AttributeError, match='immutable'):
            rebuilt._value = 3.0


def test_symbolic_constant_merge() -> None:
    # Reconstruct the same mathematical constant, rather than introducing an alias.
    first = SymbolicConstant('e', math.e)
    second = SymbolicConstant('e', math.e)
    assert first == second
    assert hash(first) == hash(second)
    assert (first * second).format() == 'e²'
    assert (first * second).format(compact=False) == 'e²'
    assert first / second == ONE


def test_symbolic_constant_operations() -> None:
    assert str(2 * PI) == str(PI * 2) == '2⋅π'
    assert str(2 / PI) == '2/π'
    assert str(PI**2) == str(PI * PI) == 'π²'
    assert 3 < PI < 4
    assert math.pi <= PI <= math.pi
    assert PI > SymbolicConstant('e', math.e)
    assert Decimal(3) < PI < Decimal(4)
    assert 3 < PI < Fraction(22, 7)
    assert PI**0 == 1
    assert PI**-1 == 1 / PI
    assert PI == Factor(PI) == PI
    assert (PI ** Fraction(1, 3)) ** 3 == PI
    assert Factor(PI) * PI == PI * Factor(PI) == PI * PI == PI**2


@pytest.mark.parametrize('operation', [operator.mul, operator.truediv])
def test_symbolic_constant_decimal_operations(
    operation: Callable[[object, object], object],
) -> None:
    decimal = Decimal('0.1')
    assert operation(PI, decimal) == operation(Factor(PI), Fraction(1, 10))
    assert operation(decimal, PI) == operation(Fraction(1, 10), Factor(PI))


@pytest.mark.parametrize('exponent', [Decimal('0.5'), Decimal('-2'), Decimal(0)])
def test_symbolic_constant_decimal_powers(exponent: Decimal) -> None:
    assert PI**exponent == Factor(PI) ** Fraction(exponent)


# Factor construction


def test_factor_creation() -> None:
    with pytest.warns(InexactFloatWarning):
        factor = Factor(2.54)
    assert type(factor.value) is Fraction
    assert factor.value == Fraction(127, 50)


@pytest.mark.parametrize('base', [TWO, PI])
def test_factor_inexact_powers(base: Factor | SymbolicConstant) -> None:
    with pytest.warns(InexactFloatWarning):
        root = base ** (1 / 3)
    assert root**3 == base


@pytest.mark.parametrize('value', [True, False, None, 1j, '3.14'])
def test_factor_invalid_types(value: bool | complex | str | None) -> None:
    with pytest.raises(TypeError, match='Factor must be a number.'):
        Factor(value)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    'value', [0, -1, Fraction(-1, 2), 0.0, Decimal(0), Fraction(0), -0.5, Decimal('-0.5')]
)
def test_factor_invalid_nonpositive(value: Fractionable) -> None:
    with pytest.raises(ValueError, match='Factor must be positive'):
        Factor(value)


@pytest.mark.parametrize('value', [float('nan'), Decimal('NaN')])
def test_factor_invalid_nan(value: float | Decimal) -> None:
    with pytest.raises(ValueError, match='cannot convert NaN to integer ratio'):
        Factor(value)


@pytest.mark.parametrize('value', [float('inf'), float('-inf'), Decimal('Infinity')])
def test_factor_invalid_infinity(value: float | Decimal) -> None:
    with pytest.raises(OverflowError, match='cannot convert Infinity to integer ratio'):
        Factor(value)


def test_factor_from_dict() -> None:
    from types import MappingProxyType

    components = MappingProxyType({2: -1, 3: Fraction(1, 2), PI: 0})
    factor = Factor.from_dict(components)
    assert factor == Factor(3) ** Fraction(1, 2) / 2
    assert str(factor) == '3¹ᐟ²/2'
    assert Factor.from_dict({2: 0}) == ONE
    with pytest.warns(InexactFloatWarning):
        converted = Factor.from_dict({2: 2.54})
    assert converted == TWO ** Fraction(127, 50)
    reciprocal = Factor(Fraction(1, 2))
    assert eval(repr(reciprocal)) == reciprocal


@pytest.mark.parametrize('components', [None, [], (), 1, '2'])
def test_factor_from_dict_invalid_components(components: object) -> None:
    with pytest.raises(TypeError, match=f'mapping for components, got {type(components).__name__}'):
        Factor.from_dict(components)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize('base', [0, -2, 1, 4, True, False])
def test_factor_from_dict_invalid_nonprime_bases(base: int) -> None:
    # The two primality backends report different messages for bool inputs.
    with pytest.raises(ValueError):
        Factor.from_dict({base: 1})


@pytest.mark.parametrize('base', ['m', 2.0, None])
def test_factor_from_dict_invalid_base_types(base: object) -> None:
    with pytest.raises(TypeError, match='All bases must be int or SymbolicConstant'):
        Factor.from_dict({base: 1})  # pyright: ignore[reportArgumentType]


def test_factor_symbol_conflict() -> None:
    symbol = 'x'
    first = SymbolicConstant(symbol, math.pi)
    second = SymbolicConstant(symbol, math.e)
    assert first != second
    assert len({first, second}) == 2  # no merge
    assert str(Factor(first)) == str(Factor(second)) == symbol  # no rename
    assert Factor.from_dict({first: 0, second: 1}) == Factor(second)  # no error
    with pytest.raises(ValueError, match=f'Conflicting values for symbol {symbol!r}'):
        Factor.from_dict({first: 1, second: -1})


# Arithmetic and exact equality


def test_factor_operations() -> None:
    assert Factor(12) * Factor(7) == Factor(12 * 7)
    assert str(Factor(12) / 10) == '6/5'
    assert str(10 / Factor(12)) == '5/6'
    assert str(TWO * 3) == str(3 * TWO) == '6'
    assert str(TEN**0) == '1'
    assert str(TEN**1) == '10'
    assert str(TEN**2) == '100'
    assert str(Factor(12) ** Fraction(1, 2)) == '2⋅3¹ᐟ²'
    assert str(Factor(12) / Factor(12)) == '1'
    root = Factor(12) ** Fraction(1, 2)
    assert root * root == Factor(12)
    assert root / root == ONE
    assert root**2 == Factor(12)


@pytest.mark.parametrize('operand', [2, 0.5, Fraction(3, 2), Decimal('0.1')])
def test_factor_numeric_operations(operand: Fractionable) -> None:
    exact = Fraction(operand)
    factor = Factor(8)
    assert factor * operand == Factor(8 * exact)
    assert operand * factor == Factor(8 * exact)
    assert factor / operand == Factor(Fraction(8) / exact)
    assert operand / factor == Factor(exact / 8)
    assert factor.value == 8


@pytest.mark.parametrize('zero', [0, 0.0, Fraction(0), Decimal(0)])
def test_factor_division_by_zero(zero: Fractionable) -> None:
    with pytest.raises(ZeroDivisionError):
        TWO / zero  # pyright: ignore[reportUnusedExpression]
    with pytest.raises(ZeroDivisionError):
        operator.truediv(PI, zero)


@pytest.mark.parametrize('operation', [operator.mul, operator.truediv])
@pytest.mark.parametrize('operand', [-1, Fraction(-1, 2), Decimal('-0.5')])
@pytest.mark.parametrize('base', [TWO, PI])
def test_factor_invalid_negative_factors(
    operation: Callable[[object, object], object],
    operand: int | Fraction | Decimal,
    base: Factor | SymbolicConstant,
) -> None:
    with pytest.raises(ValueError, match='must be positive'):
        operation(base, operand)
    with pytest.raises(ValueError, match='must be positive'):
        operation(operand, base)


@pytest.mark.parametrize(
    ('operation', 'reverse'),
    [(operator.mul, False), (operator.mul, True), (operator.truediv, True)],
)
def test_symbolic_constant_invalid_zero(
    operation: Callable[[object, object], object], reverse: bool
) -> None:
    with pytest.raises(ValueError, match='Factor must be positive'):
        operation(0, PI) if reverse else operation(PI, 0)


@pytest.mark.parametrize('operation', [operator.mul, operator.truediv])
def test_factor_invalid_operands(
    operation: Callable[[object, object], object],
) -> None:
    with pytest.raises(TypeError):
        operation(TWO, 'x')


@pytest.mark.parametrize('operation', [operator.mul, operator.truediv])
@pytest.mark.parametrize('as_factors', [False, True])
def test_factor_invalid_symbol_conflict(
    operation: Callable[[object, object], object], as_factors: bool
) -> None:
    first = SymbolicConstant('x', math.pi)
    second = SymbolicConstant('x', math.e)
    left, right = (Factor(first), Factor(second)) if as_factors else (first, second)
    with pytest.raises(ValueError, match="Conflicting values for symbol 'x'"):
        operation(left, right)
    assert Factor(first).value == math.pi
    assert Factor(second).value == math.e


@pytest.mark.parametrize(
    ('base', 'exponent', 'expected'),
    [
        (2, -3, Fraction(1, 8)),
        (4, 0.5, 2),
        (8, Fraction(1, 3), 2),
        (9, Decimal('0.5'), 3),
        (2, Decimal(0), 1),
    ],
)
def test_factor_numeric_powers(base: int, exponent: Fractionable, expected: int | Fraction) -> None:
    assert Factor(base) ** exponent == Factor(expected)


@pytest.mark.parametrize('exponent', [True, False, 'x'])
def test_factor_invalid_exponents(exponent: bool | str) -> None:
    with pytest.raises(TypeError, match=f'Cannot convert {type(exponent).__name__} to Fraction'):
        TWO**exponent  # pyright: ignore[reportOperatorIssue, reportUnusedExpression]


@pytest.mark.parametrize('factor', [PI**1000, TWO ** Fraction(-2201, 2)])
def test_factor_division_extreme_cancellation(factor: Factor) -> None:
    assert factor / factor == ONE
    assert (1 / factor) * factor == ONE


def test_factor_extreme_rationals() -> None:
    large = 2**53
    assert Factor(large) == large
    assert Factor(large) != large + 1
    assert large + 1 != Factor(large)
    tiny = Fraction(1, 10**400)
    assert Factor(tiny) == tiny
    assert Factor(tiny) != 0
    assert (ONE / Factor(tiny)).value == 10**400
    factor = Factor(10**400)
    assert type(factor.value) is int
    assert factor.value == 10**400
    assert factor.is_integer()


def test_factor_exact_equality() -> None:
    root = TWO ** Fraction(1, 2)
    assert root != math.sqrt(2)
    assert root != Fraction.from_float(math.sqrt(2))
    assert Factor(PI) != math.pi
    assert Factor(Decimal('0.1')) != 0.1
    assert Factor(Decimal('0.1')) == Decimal('0.1')
    # Equality can reject inconsistent definitions without combining them.
    assert Factor(SymbolicConstant('x', math.pi)) != Factor(SymbolicConstant('x', math.e))


@pytest.mark.parametrize('number', [2, 2.0, Fraction(2), Decimal(2)])
def test_factor_numeric_equality(number: Fractionable) -> None:
    assert TWO == number
    assert number == TWO


@pytest.mark.parametrize(
    ('factor', 'number'),
    [
        (PI**1000, 2),
        (PI**-1000, 0),
        (TWO ** Fraction(-2201, 2), 1),
        (Factor(PI), -1),
        (Factor(PI), Decimal(0)),
        (Factor(PI), float('inf')),
        (Factor(PI), float('nan')),
    ],
)
def test_factor_equality_without_evaluation(
    factor: Factor, number: int | Decimal | float, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_evaluation(self: Factor) -> NoReturn:
        pytest.fail('Equality must not evaluate a floating-point approximation.')

    monkeypatch.setattr(Factor, 'value', property(unexpected_evaluation))
    assert factor != number
    assert number != factor


# Classification and decomposition


@pytest.mark.parametrize(
    ('factor', 'expected'),
    [
        (ONE, (True, True)),
        (Factor(12), (True, True)),
        (Factor(Fraction(1, 3)), (False, True)),
        (TWO ** Fraction(1, 2), (False, False)),
        (Factor(PI), (False, False)),
        (PI**1000, (False, False)),
        (PI**-1000, (False, False)),
        (TWO**100000, (True, True)),
        (TWO ** Fraction(-2201, 2), (False, False)),
        ((Factor(12) ** Fraction(1, 2)) ** 2, (True, True)),
        (PI / PI, (True, True)),
    ],
)
def test_factor_numeric_classification(
    factor: Factor, expected: tuple[bool, bool], monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_evaluation(self: Factor) -> NoReturn:
        pytest.fail('Classification must not evaluate the factor.')

    monkeypatch.setattr(Factor, 'value', property(unexpected_evaluation))
    assert (factor.is_integer(), factor.is_rational()) == expected


def test_factor_subclass_classification() -> None:
    class Integer(int):
        pass

    class Rational(Fraction):
        pass

    class Constant(SymbolicConstant):
        pass

    assert Factor(Integer(12)).is_integer()
    assert Factor(Integer(12)).is_rational()
    assert not Factor(Rational(1, 3)).is_integer()
    assert Factor(Rational(1, 3)).is_rational()
    factor = Constant('e', math.e) ** 1000
    assert not factor.is_integer()
    assert not factor.is_rational()


@pytest.mark.parametrize(
    ('factor', 'rationalize', 'expected_rational', 'expected_irrational'),
    [
        (Factor(12), False, 12, ONE),
        (2 * PI, False, 2, Factor(PI)),
        (TWO ** Fraction(-2, 3), True, Fraction(1, 2), TWO ** Fraction(1, 3)),
        (TWO ** Fraction(3, 2), False, 2, TWO ** Fraction(1, 2)),
        (TWO ** Fraction(3, 2), True, 2, TWO ** Fraction(1, 2)),
        (TWO ** Fraction(-3, 2), False, Fraction(1, 2), TWO ** Fraction(-1, 2)),
        (TWO ** Fraction(-3, 2), True, Fraction(1, 4), TWO ** Fraction(1, 2)),
    ],
)
def test_factor_decompose(
    factor: Factor,
    rationalize: bool,
    expected_rational: int | Fraction,
    expected_irrational: Factor,
) -> None:
    rational, irrational = factor.decompose(rationalize_denominator=rationalize)
    assert rational == expected_rational
    assert irrational == expected_irrational
    assert Factor(rational) * irrational == factor


# Numerical evaluation and scaling


@pytest.mark.parametrize(
    ('source', 'expected', 'expected_type'),
    [
        (12, 12, int),
        (12.0, 12, int),
        (Fraction(12), 12, int),
        (Decimal('12.0'), 12, int),
        (1.5, Fraction(3, 2), Fraction),
        (Fraction(1, 3), Fraction(1, 3), Fraction),
        (Decimal('0.0254'), Fraction(127, 5000), Fraction),
        (Decimal('1e-12'), Fraction(1, 10**12), Fraction),
        (PI, math.pi, float),
    ],
)
def test_factor_value_exact_types(
    source: Fractionable | SymbolicConstant,
    expected: int | Fraction | float,
    expected_type: type[int | Fraction | float],
) -> None:
    factor = Factor(source)
    assert type(factor.value) is expected_type
    assert factor.value == expected
    assert type(float(factor)) is float
    assert float(factor) == float(expected)


@pytest.mark.parametrize(
    ('twos', 'pis', 'es'),
    [
        (1000, -400, -400),  # Three factors balance to a finite value.
        (2000, -1000, 0),  # Individual terms exceed the float range.
        (-2000, 1000, 0),
        (-1155, 700, 0),  # Overflowing symbolic term and underflowing coefficient.
        (1000, -645, 0),  # Intermediate subnormal terms lose precision.
        (1000, -650, 0),
    ],
)
def test_factor_value_balanced_product(twos: int, pis: int, es: int) -> None:
    factor = TWO**twos * PI**pis * SymbolicConstant('e', math.e) ** es
    with localcontext() as context:
        context.prec = 60
        expected = float(
            Decimal(2) ** twos
            * Decimal.from_float(math.pi) ** pis
            * Decimal.from_float(math.e) ** es
        )
    assert math.isfinite(factor.value) and factor.value > 0
    assert factor.value == pytest.approx(expected, rel=1e-12, abs=0)


def test_factor_value_float_limits() -> None:
    with pytest.raises(OverflowError):
        (PI**1000).value
    assert (PI**-1000).value == 0.0


@pytest.mark.parametrize('exponent', [Decimal('3.2305925317824'), Decimal('3.23059253178236')])
def test_factor_value_maximum_boundary(exponent: Decimal) -> None:
    factor = PI**622 / TWO**exponent
    with localcontext() as context:
        context.prec = 100
        expected = float(Decimal.from_float(PI.value) ** 622 / Decimal(2) ** exponent)
    assert math.isfinite(expected)
    with localcontext() as context:
        context.prec = 2
        context.Emax = 9
        context.Emin = -9
        result = factor.value
        assert (context.prec, context.Emax, context.Emin) == (2, 9, -9)
    assert type(result) is float
    assert math.isfinite(result)
    assert result == pytest.approx(expected, rel=1e-14, abs=0)


def test_factor_value_overflow_boundary() -> None:
    exponent = Decimal('3.2305925317822')
    with localcontext() as context:
        context.prec = 100
        expected = float(Decimal.from_float(PI.value) ** 622 / Decimal(2) ** exponent)
    assert math.isinf(expected)
    with pytest.raises(OverflowError, match='Factor value is too large for a float'):
        (PI**622 / TWO**exponent).value


@pytest.mark.parametrize(
    ('exponent', 'expected'),
    [(Decimal('2102.2305925317822'), math.ulp(0.0)), (Decimal('2102.2305925317824'), 0.0)],
)
def test_factor_value_underflow_boundary(exponent: Decimal, expected: float) -> None:
    factor = PI**622 / TWO**exponent
    with localcontext() as context:
        context.prec = 100
        reference = float(Decimal.from_float(PI.value) ** 622 / Decimal(2) ** exponent)
    assert reference == expected
    with localcontext() as context:
        context.prec = 2
        context.Emax = 9
        context.Emin = -9
        result = factor.value
    assert type(result) is float
    assert result == expected


def test_factor_value_subnormal_result() -> None:
    factor = PI**1000 * TWO**-2725
    with localcontext() as context:
        context.prec = 60
        expected = float(Decimal.from_float(math.pi) ** 1000 / Decimal(2) ** 2725)
    assert expected == math.ulp(0.0)
    assert factor.value == expected


@pytest.mark.parametrize(
    ('factor', 'expected', 'expected_type'),
    [
        (TWO * Factor(3), 6, int),
        (ONE / Factor(3), Fraction(1, 3), Fraction),
        (TWO ** Fraction(1, 2), math.sqrt(2), float),
        (PI**-1000, 0.0, float),
    ],
)
def test_factor_value_cache(
    factor: Factor,
    expected: int | Fraction | float,
    expected_type: type[int | Fraction | float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = factor.value
    assert type(first) is expected_type
    assert first == (pytest.approx(expected) if expected_type is float else expected)

    def unexpected_decompose(self: Factor, *, rationalize_denominator: bool = False) -> NoReturn:
        pytest.fail('A cached value must not decompose the expression again.')

    monkeypatch.setattr(Factor, 'decompose', unexpected_decompose)
    assert factor.value is first
    assert float(factor) == float(first)


def test_factor_conversion_round_trip() -> None:
    inch = Factor(Decimal('0.0254'))
    centimetre = Factor(Decimal('0.01'))
    conversion = inch / centimetre
    assert conversion.value == Fraction(127, 50)
    for magnitude in [Fraction(3, 2), Decimal('1.5')]:
        converted = conversion(magnitude)
        assert type(converted) is Fraction
        assert converted == Fraction(381, 100)
        assert (centimetre / inch)(converted) == Fraction(3, 2)
    assert Factor(Fraction(1, 3))(3) == 1


@pytest.mark.parametrize(
    ('factor', 'magnitude', 'expected'),
    [
        (TWO, Decimal('1.5'), 3),
        (ONE / 3, Decimal('1.5'), Fraction(1, 2)),
        (ONE / 3, Decimal('-1.5'), Fraction(-1, 2)),
        (ONE / 3, Decimal(0), 0),
        (ONE / 3, Decimal('1e400'), Fraction(10**400, 3)),
        (Factor(PI), Decimal('1.5'), math.pi * 1.5),
        (Factor(PI), Decimal('-1.5'), -math.pi * 1.5),
        (Factor(PI), Decimal(0), 0.0),
    ],
)
def test_factor_decimal_scaling(
    factor: Factor, magnitude: Decimal, expected: int | Fraction | float
) -> None:
    with localcontext() as context:
        context.prec = 2
        result = factor(magnitude)
    assert type(result) is type(expected)
    assert result == (pytest.approx(expected) if isinstance(expected, float) else expected)


@pytest.mark.parametrize(
    ('exponent', 'magnitude', 'expected'),
    [(200, Decimal('1e-400'), math.pi * 1e-200), (-200, Decimal('1e400'), math.pi * 1e200)],
)
def test_factor_decimal_extremes(exponent: int, magnitude: Decimal, expected: float) -> None:
    factor = PI * TEN**exponent
    with localcontext() as context:
        context.prec = 2
        result = factor(magnitude)
    assert type(result) is float
    assert result == pytest.approx(expected, rel=1e-12, abs=0)


def test_factor_scaling_without_factorization(monkeypatch: pytest.MonkeyPatch) -> None:
    factor = Factor(Fraction(1, 3))

    def unexpected_factorization(number: int) -> NoReturn:
        pytest.fail('Applying a factor must not factor the magnitude.')

    monkeypatch.setattr(factor_module, 'factorint', unexpected_factorization)
    magnitude = 1000003 * 1000033
    assert factor(magnitude) == Fraction(magnitude, 3)
    assert factor(-magnitude) == Fraction(-magnitude, 3)
    assert factor(0) == 0


# Formatting and representation


def test_factor_format_forms() -> None:
    assert Factor(12).format() == '12'
    assert Factor(12).format(compact=False) == '2²⋅3'
    assert Factor(Fraction(3, 10)).format() == '3/10'
    assert Factor(Fraction(3, 10)).format(compact=False) == '3/(2⋅5)'
    factor = Factor(12345678901234567890)
    assert factor.format(compact=False) == '2⋅3²⋅5⋅101⋅3541⋅3607⋅3803⋅27961'


@pytest.mark.parametrize(
    ('exponent', 'expected'),
    [
        (Fraction(1, 2), '6¹ᐟ²'),
        (Fraction(1, 3), '6¹ᐟ³'),
        (Fraction(2, 3), '6²ᐟ³'),
        (Fraction(3, 2), '6⋅6¹ᐟ²'),
        (Fraction(-1, 2), '6¹ᐟ²/6'),
    ],
)
def test_factor_format_merged_roots(exponent: Fraction, expected: str) -> None:
    factor = (TWO**exponent) * (Factor(3) ** exponent)
    original = repr(factor)
    assert factor.format(compact=True) == expected
    assert str(factor) == expected
    assert factor == Factor(6) ** exponent
    assert repr(factor) == original


def test_factor_format_mixed_roots() -> None:
    factor = Factor.from_dict(
        {
            2: Fraction(1, 2),
            PI: Fraction(1, 2),
            5: Fraction(1, 3),
            3: Fraction(1, 2),
            7: Fraction(1, 3),
        }
    )
    assert factor.format() == '6¹ᐟ²⋅π¹ᐟ²⋅35¹ᐟ³'
    assert factor.format(compact=False) == '2¹ᐟ²⋅π¹ᐟ²⋅5¹ᐟ³⋅3¹ᐟ²⋅7¹ᐟ³'


def test_factor_format_mixed_factors() -> None:
    constant = SymbolicConstant('e', math.e)
    factor = 6 * Factor(6) ** Fraction(1, 2) * constant
    assert factor.format() == '6⋅6¹ᐟ²⋅e'
    assert factor.value == pytest.approx(6 * math.sqrt(6) * math.e)


@pytest.mark.parametrize(
    ('base', 'exponent', 'expected'),
    [
        (8, Fraction(1, 2), '2⋅2¹ᐟ²'),
        (2, Fraction(-1, 2), '2¹ᐟ²/2'),
        (2, Fraction(-3, 2), '2¹ᐟ²/4'),
    ],
)
def test_factor_format_coefficients(base: int, exponent: Fraction, expected: str) -> None:
    assert str(Factor(base) ** exponent) == expected


@pytest.mark.parametrize(
    'exponent', [Fraction(10**20 + 1, 3), Fraction(-(10**20) - 1, 7), Fraction(1, 1000001)]
)
def test_factor_repr_round_trip(exponent: Fraction) -> None:
    factor = TWO**exponent
    with warnings.catch_warnings():
        warnings.simplefilter('error', InexactFloatWarning)
        rebuilt = eval(repr(factor))
    assert rebuilt == factor


# Factorization backends


def test_factor_sympy_factorization() -> None:
    pytest.importorskip('sympy')
    p, q = 1000003, 1000033
    with warnings.catch_warnings():
        warnings.simplefilter('error', IncompleteFactorWarning)
        factor = Factor(p * q)
        reciprocal = Factor(Fraction(1, p * q))
    assert factor.format(compact=False) == f'{p}⋅{q}'
    assert factor / Factor(p) / Factor(q) == ONE
    assert reciprocal == ONE / Factor(p) / Factor(q)


def test_factor_sympy_primality() -> None:
    pytest.importorskip('sympy')
    prime = 2**89 - 1
    with warnings.catch_warnings():
        warnings.simplefilter('error', IncompleteFactorWarning)
        factor = Factor.from_dict({prime: 1})
    assert factor.value == prime


@pytest.mark.usefixtures('fallback_backend')
def test_factor_fallback_incomplete() -> None:
    value = 1234567890123456789012345678901234567890
    with pytest.warns(IncompleteFactorWarning, match='exceeded fallback factoring limit'):
        factor = Factor(value)
    assert factor.value == value
    assert factor.format(compact=False) == '2⋅3²⋅5⋅73⋅101⋅137⋅3541⋅3607⋅3803⋅27961⋅9999000099990001'


@pytest.mark.usefixtures('fallback_backend')
@pytest.mark.parametrize(
    'value',
    [
        1000003 * 1000033,
        Fraction(1, 1000003 * 1000033),
        (1000003 * 1000033) ** 2,
        Fraction(1, (1000003 * 1000033) ** 2),
    ],
)
def test_factor_fallback_unresolved_warning(value: int | Fraction) -> None:
    with pytest.warns(IncompleteFactorWarning):
        factor = Factor(value)
    assert factor.value == value
    assert any(
        isinstance(base, numerics_module.UnfactoredInteger) for base in factor._monomial.bases()
    )


@pytest.mark.usefixtures('fallback_backend')
@pytest.mark.parametrize('base', [TWO, PI])
@pytest.mark.parametrize('operation', [operator.mul, operator.truediv])
@pytest.mark.parametrize('reverse', [False, True])
def test_factor_fallback_arithmetic(
    base: Factor | SymbolicConstant,
    operation: Callable[[object, object], Factor],
    reverse: bool,
) -> None:
    value = 1000003 * 1000033
    with pytest.warns(IncompleteFactorWarning):
        result = operation(value, base) if reverse else operation(base, value)
    expected = operation(value, float(base)) if reverse else operation(float(base), value)
    assert float(result) == pytest.approx(expected, rel=1e-12, abs=0)


def test_factor_constructor_unfactored_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    def factor_with_unresolved_base(number: int) -> dict[int, int]:
        return {numerics_module.UnfactoredInteger(number, 10): 1}

    monkeypatch.setattr(factor_module, 'factorint', factor_with_unresolved_base)
    factor = Factor(15)
    assert factor.value == 15
    (base,) = factor._monomial.bases()
    assert isinstance(base, UnfactoredInteger)
    assert int(base) == 15
    assert base.limit == 10


@pytest.mark.usefixtures('fallback_backend')
def test_factor_from_dict_unproven_warning() -> None:
    value = 2**89 - 1
    with pytest.warns(IncompleteFactorWarning, match='not guaranteed to be definitive'):
        factor = Factor.from_dict({value: 1})
    assert factor.value == value


def test_factor_from_dict_unfactored_warning() -> None:
    with pytest.warns(IncompleteFactorWarning, match='has not been fully factored'):
        factor = Factor.from_dict({numerics_module.UnfactoredInteger(15, 10): 1})
    assert factor.value == 15


@pytest.mark.usefixtures('fallback_backend')
@pytest.mark.parametrize(
    ('value', 'exponent', 'expected'),
    [
        (1000003**2, Fraction(1, 2), 1000003),
        (1000003**2, Fraction(-1, 2), Fraction(1, 1000003)),
        (Fraction(1, 1000003**2), Fraction(1, 2), Fraction(1, 1000003)),
        (1000003**3, Fraction(1, 3), 1000003),
        (1000003**3, Fraction(-2, 3), Fraction(1, 1000003**2)),
        (1000003**3, Fraction(2, 3), 1000003**2),
    ],
)
def test_factor_fallback_roots(
    value: int | Fraction, exponent: Fraction, expected: int | Fraction
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error', IncompleteFactorWarning)
        factor = Factor(value) ** exponent
    assert factor == expected
    assert type(factor.value) is type(expected)
    assert factor.value == expected
    assert factor.is_integer() == (type(expected) is int)
    assert factor.is_rational()


@pytest.mark.usefixtures('fallback_backend')
@pytest.mark.parametrize('exponent', [1, Fraction(1, 2), Fraction(-2, 3)])
def test_factor_fallback_structural_comparison(
    exponent: int | Fraction,
) -> None:
    p, q = 1000003, 1000033
    with pytest.warns(IncompleteFactorWarning):
        combined = Factor(p * q) ** exponent
    split = Factor(p) ** exponent * Factor(q) ** exponent
    with warnings.catch_warnings():
        warnings.simplefilter('error', IncompleteFactorWarning)
        assert combined != split
        assert combined / split != ONE
        assert split / combined != ONE
        assert combined / combined == ONE
    assert float(combined / split) == pytest.approx(1)
    assert combined.is_rational() == (exponent == 1)


@pytest.mark.usefixtures('fallback_backend')
def test_factor_fallback_partial_cancellation() -> None:
    p, q = 1000003, 1000033
    with pytest.warns(IncompleteFactorWarning):
        combined = Factor(p**3 * q) ** Fraction(1, 3)
    result = combined / (Factor(q) ** Fraction(1, 3))
    assert result != Factor(p)
    assert float(result) == pytest.approx(p)
    # The result is rational, but the unresolved bases prevent recognizing it.
    assert not result.is_rational()


@pytest.mark.usefixtures('fallback_backend')
def test_factor_equality_preserves_state() -> None:
    p, q = 1000003, 1000033
    with pytest.warns(IncompleteFactorWarning):
        combined = Factor(p * q)
    split = Factor(p) * Factor(q)
    before = repr(combined), repr(split)
    combined_value, split_value = combined.value, split.value
    assert combined != split
    assert split != combined
    assert (repr(combined), repr(split)) == before
    assert combined.value is combined_value
    assert split.value is split_value


@pytest.mark.usefixtures('fallback_backend')
def test_factor_repr_unresolved_bases() -> None:
    with pytest.warns(IncompleteFactorWarning):
        factor = Factor(1000003 * 1000033) ** Fraction(1, 3)
    assert any(isinstance(base, UnfactoredInteger) for base in factor._monomial.bases())
    with pytest.warns(IncompleteFactorWarning):
        rebuilt = eval(repr(factor))
    assert rebuilt == factor
    assert rebuilt.format() == factor.format()


@pytest.mark.usefixtures('fallback_backend')
def test_factor_unresolved_copy() -> None:
    import pickle
    from copy import copy, deepcopy

    with pytest.warns(IncompleteFactorWarning):
        factor = Factor(1000003 * 1000033) ** Fraction(1, 3)
    (original_base,) = factor._monomial.bases()
    assert isinstance(original_base, UnfactoredInteger)
    for rebuilt in (copy(factor), deepcopy(factor), pickle.loads(pickle.dumps(factor))):
        assert rebuilt == factor
        assert repr(rebuilt) == repr(factor)
        (base,) = rebuilt._monomial.bases()
        assert isinstance(base, UnfactoredInteger)
        assert base.limit == original_base.limit
        assert float(rebuilt) == pytest.approx(float(factor))
