import warnings
from decimal import Decimal
from fractions import Fraction

import pytest

import unitx._numerics as numerics_module
from unitx._numerics import as_exponent, as_fraction, factorint, isprime
from unitx.exceptions import IncompleteFactorWarning, InexactFloatWarning

_LARGE_PRIME = 1000003  # a prime number larger than the default limit of 1,000,000
_MERSENNE_PRIME = 2**61 - 1  # a known large prime number


def test_as_exponent_fraction_subclass() -> None:
    class Rational(Fraction):
        pass

    value = Rational(1, 10**30)
    exponent = as_exponent(value)
    assert type(exponent) is Fraction
    assert exponent == Fraction(1, 10**30)


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


@pytest.mark.parametrize('value', [float('nan'), Decimal('NaN')])
def test_as_fraction_invalid_nan(value: float | Decimal) -> None:
    with pytest.raises(ValueError, match='cannot convert NaN to integer ratio'):
        as_fraction(value)


@pytest.mark.parametrize(
    'value', [float('inf'), float('-inf'), Decimal('Infinity'), Decimal('-Infinity')]
)
def test_as_fraction_invalid_infinity(value: float | Decimal) -> None:
    with pytest.raises(OverflowError, match='cannot convert Infinity to integer ratio'):
        as_fraction(value)


def test_as_fraction_invalid_denominator() -> None:
    with pytest.raises(ValueError, match='at least 1'):
        as_fraction(1.3, float_max_denominator=-10000)


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


def test_as_fraction_float_division() -> None:
    with pytest.warns(InexactFloatWarning, match='approximate'):
        result = as_fraction(1 / 3)
    assert result == Fraction(1, 3) != Fraction(1 / 3)


@pytest.mark.parametrize(
    ('number', 'expected'),
    [
        (1, {}),
        (12, {2: 2, 3: 1}),
        (35, {5: 1, 7: 1}),
        (121, {11: 2}),
        (2**64, {2: 64}),
        (5 * 7 * 11, {5: 1, 7: 1, 11: 1}),
        (5**2 * 7**3 * 11**2, {5: 2, 7: 3, 11: 2}),
        (2**9 * 3**7 * 5**4 * 7**3, {2: 9, 3: 7, 5: 4, 7: 3}),
    ],
)
def test_factorint_prime_factors(number: int, expected: dict[int, int]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error', IncompleteFactorWarning)
        assert factorint(number) == expected


@pytest.mark.parametrize('number', [0, -1])
def test_factorint_invalid_nonpositive(number: int) -> None:
    with pytest.raises(ValueError, match='positive integer'):
        factorint(number)


def test_factorint_trial_limit() -> None:
    p1, p2 = 11, 13
    composite = p1 * p2
    with pytest.warns(IncompleteFactorWarning, match=f'limit {p1 - 1}.'):
        assert factorint(composite, limit=p1 - 1) == {composite: 1}
    assert factorint(composite, limit=p1) == {p1: 1, p2: 1}


@pytest.mark.parametrize('degree', [1, 2, 3, 9])
def test_factorint_unresolved_powers(degree: int) -> None:
    composite = 11 * 13
    with pytest.warns(IncompleteFactorWarning):
        factors = factorint(composite**degree, limit=5)
    assert factors == {composite: degree}


@pytest.mark.parametrize('prime', [_LARGE_PRIME, _MERSENNE_PRIME])
def test_factorint_large_primes(prime: int) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error', IncompleteFactorWarning)
        assert factorint(prime, limit=1) == {prime: 1}


def test_factorint_recognizes_prime_remainder() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error', IncompleteFactorWarning)
        assert factorint(5 * _MERSENNE_PRIME, limit=5) == {5: 1, _MERSENNE_PRIME: 1}


@pytest.mark.parametrize(
    ('number', 'limit', 'expected'),
    [
        (_LARGE_PRIME**2, 1, {_LARGE_PRIME: 2}),
        (_LARGE_PRIME**4, 1, {_LARGE_PRIME: 4}),
        ((6 * _LARGE_PRIME) ** 2, 1, {2: 2, 3: 2, _LARGE_PRIME: 2}),
        (5 * _LARGE_PRIME**2, 5, {5: 1, _LARGE_PRIME: 2}),
    ],
)
def test_factorint_square_factors(number: int, limit: int, expected: dict[int, int]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error', IncompleteFactorWarning)
        assert factorint(number, limit=limit) == expected


def test_factorint_unresolved_square() -> None:
    composite = _LARGE_PRIME * _MERSENNE_PRIME
    limit = 5
    with pytest.warns(IncompleteFactorWarning, match=f'limit {limit}.'):
        assert factorint(composite**2, limit=limit) == {composite: 2}


@pytest.mark.parametrize('degree', [3, 5, 7, 9, 15, 31])
def test_factorint_perfect_powers(degree: int) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error', IncompleteFactorWarning)
        assert factorint(_LARGE_PRIME**degree, limit=1) == {_LARGE_PRIME: degree}


def test_factorint_power_remainder() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error', IncompleteFactorWarning)
        assert factorint(5 * _LARGE_PRIME**3, limit=5) == {5: 1, _LARGE_PRIME: 3}


def test_factorint_unresolved_power() -> None:
    composite = _LARGE_PRIME * _MERSENNE_PRIME
    with pytest.warns(IncompleteFactorWarning, match='limit 1.'):
        factors = factorint(composite**3, limit=1)
    assert factors == {composite: 3}
    base = next(iter(factors))
    assert isinstance(base, numerics_module.UnfactoredInteger)
    assert base.limit == 1


@pytest.mark.parametrize(
    ('value', 'degree', 'root'),
    [(1, 7, 1), (7, 1, 7), (16, 2, 4), (17, 2, 4), (27, 3, 3), (28, 3, 3), (2, 99, 1)],
)
def test_integer_root_returns_floor_root(value: int, degree: int, root: int) -> None:
    assert numerics_module._integer_root(value, degree) == root


def test_factorint_with_bounded_isprime(monkeypatch: pytest.MonkeyPatch) -> None:
    number = 11 * _MERSENNE_PRIME

    def bounded_isprime(number: int) -> bool:
        assert number < 2**64, 'Primality shortcut exceeded its deterministic range.'
        return True

    monkeypatch.setattr(numerics_module, 'isprime', bounded_isprime)
    with pytest.warns(IncompleteFactorWarning, match='limit 1.'):
        assert factorint(number, limit=1) == {number: 1}


@pytest.mark.parametrize('number', [2, 3, 5, 37, 41, _LARGE_PRIME, _MERSENNE_PRIME])
def test_isprime_accepts_primes(number: int) -> None:
    assert isprime(number)


@pytest.mark.parametrize('number', [-1, 0, 1, 4, 91, 561, 1105, 1729, 3215031751])
def test_isprime_nonprimes(number: int) -> None:
    assert not isprime(number)


def test_isprime_unproven_warning() -> None:
    mersenne_prime = 2**89 - 1
    with pytest.warns(IncompleteFactorWarning, match='not guaranteed to be definitive'):
        assert isprime(mersenne_prime)


def test_unfactored_integer_copy() -> None:
    import pickle
    from copy import copy, deepcopy

    original = numerics_module.UnfactoredInteger(15, 10)
    for rebuilt in (copy(original), deepcopy(original), pickle.loads(pickle.dumps(original))):
        assert isinstance(rebuilt, numerics_module.UnfactoredInteger)
        assert int(rebuilt) == 15
        assert rebuilt.limit == 10
