"""Tests for the Dimension class."""

import pickle
import warnings
from copy import copy, deepcopy
from decimal import Decimal
from fractions import Fraction

import pytest

from unitx.dimension import DIMENSIONLESS, Dimension
from unitx.exceptions import InexactFloatWarning


def test_dimensionless() -> None:
    dimensionless = Dimension()
    assert dimensionless is DIMENSIONLESS
    assert repr(dimensionless) == 'Dimension()'
    assert str(dimensionless) == '1'


def test_dimension_creation() -> None:
    dimension = Dimension(T=1, L=2)
    assert dimension.as_tuple() == (1, 2, 0, 0, 0, 0, 0)
    assert dimension.as_dict() == {'T': 1, 'L': 2}
    assert repr(dimension) == 'Dimension(T=1, L=2)'
    assert str(dimension) == 'TL²'
    fraction_dimension = Dimension(T=Fraction(1, 2), L=Fraction(2, 3))
    assert dimension != fraction_dimension
    assert repr(fraction_dimension) == 'Dimension(T=Fraction(1, 2), L=Fraction(2, 3))'
    assert str(fraction_dimension) == 'T¹ᐟ²L²ᐟ³'


@pytest.mark.parametrize(
    'dimension',
    [
        DIMENSIONLESS,
        Dimension(T=1, L=2),
        Dimension(T=Fraction(1, 2), L=Fraction(2, 3)),
        Dimension(Theta=Fraction(-1, 10**10)),
    ],
)
def test_dimension_roundtrip_conversion(dimension: Dimension) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error', InexactFloatWarning)
        assert Dimension(*dimension.as_tuple()) == dimension
        assert Dimension(**dimension.as_dict()) == dimension
        assert eval(repr(dimension)) == dimension


def test_dimension_float_exponent() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error', InexactFloatWarning)
        assert Dimension(L=0.1) == Dimension(L=Fraction(1, 10))
        assert Dimension(L=1e-10).L == Fraction(1, 10**10)


def test_dimension_properties() -> None:
    dimension = Dimension(T=1, L=2, M=3, I=4, Theta=5, N=6, J=7)
    assert dimension.T == dimension.time == 1
    assert dimension.L == dimension.length == 2
    assert dimension.M == dimension.mass == 3
    assert dimension.I == dimension.current == 4
    assert dimension.Θ == dimension.Theta == dimension.temperature == 5
    assert dimension.N == dimension.amount == 6
    assert dimension.J == dimension.luminous == 7


def test_dimension_temperature_alias() -> None:
    dimension = Dimension(Theta=1)
    assert dimension.Theta == 1
    assert dimension.Θ == 1
    assert dimension.temperature == 1
    assert str(dimension) == 'Θ'


def test_dimension_operations() -> None:
    dimension = Dimension(T=1, L=2)
    mass_dimension = Dimension(M=1)
    assert dimension * mass_dimension == Dimension(T=1, L=2, M=1)
    assert dimension / mass_dimension == Dimension(T=1, L=2, M=-1)
    assert dimension**2 == Dimension(T=2, L=4)
    assert 1 / dimension == Dimension(T=-1, L=-2)


def test_dimension_inplace_operations() -> None:
    dimension = Dimension(T=1, L=2)
    mass_dimension = Dimension(M=1)
    area_dimension = Dimension(L=2)
    original_dimension = dimension
    original_mass = mass_dimension
    original_area = area_dimension
    dimension *= mass_dimension
    assert dimension == Dimension(T=1, L=2, M=1)
    assert original_dimension == Dimension(T=1, L=2)
    mass_dimension /= mass_dimension
    assert mass_dimension is DIMENSIONLESS
    assert original_mass == Dimension(M=1)
    area_dimension **= Fraction(1, 3)
    assert area_dimension == Dimension(L=Fraction(2, 3))
    assert original_area == Dimension(L=2)


def test_dimension_fractional_power() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error', InexactFloatWarning)
        dimension = Dimension(L=1) ** Fraction(1, 2)
    assert Fraction(1, 2) == dimension.L
    assert str(dimension) == 'L¹ᐟ²'


def test_dimension_invalid_operations() -> None:
    with pytest.raises(TypeError, match='unsupported operand'):
        Dimension(L=1) * 2  # pyright: ignore[reportUnusedExpression, reportOperatorIssue]
    with pytest.raises(TypeError, match='unsupported operand'):
        2 / Dimension(T=1)  # pyright: ignore[reportUnusedExpression, reportOperatorIssue]
    with pytest.raises(TypeError, match='unsupported operand'):
        Dimension(L=1) + Dimension(L=1)  # pyright: ignore[reportUnusedExpression, reportOperatorIssue]
    with pytest.raises(TypeError, match='convert bool'):
        Dimension(L=1) ** True  # pyright: ignore[reportUnusedExpression]


def test_dimension_hash() -> None:
    assert hash(Dimension(L=1)) == hash(Dimension(L=1))
    assert {Dimension(L=1), Dimension(L=1)} == {Dimension(L=1)}


def test_dimension_float_power() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error', InexactFloatWarning)
        assert (Dimension(L=1) ** 0.1) ** 10 == Dimension(L=1)
        assert (Dimension(L=1) ** 1e-10).L == Fraction(1, 10**10)


@pytest.mark.parametrize('dimension', [Dimension(L=1), DIMENSIONLESS])
@pytest.mark.parametrize('attribute', ['_monomial', 'L', 'length'])
def test_dimension_immutable(dimension: Dimension, attribute: str) -> None:
    exponents = dimension.as_tuple()
    monomial = dimension._monomial
    original_hash = hash(dimension)
    mapping = {dimension: 'unchanged'}
    try:
        with pytest.raises(AttributeError, match='immutable'):
            setattr(dimension, attribute, Dimension(T=1).as_tuple())
        with pytest.raises(AttributeError, match='immutable'):
            delattr(dimension, attribute)
        assert dimension.as_tuple() == exponents
        assert hash(dimension) == original_hash
        assert mapping[dimension] == 'unchanged'
    finally:
        # A regression must not leave the shared singleton corrupted for other tests.
        object.__setattr__(dimension, '_monomial', monomial)


def test_dimension_from_iter() -> None:
    """Test the internal `Dimension._from_iter`"""
    original_dimension = Dimension(L=1, T=2)
    restored_dimension = Dimension._from_iter(original_dimension.as_tuple())
    assert restored_dimension == original_dimension
    assert restored_dimension is not original_dimension
    with pytest.raises(AssertionError, match='Expected 7 exponents'):
        Dimension._from_iter(map(Fraction, [1, 2]))
    with pytest.raises(AssertionError, match='All exponents must be Fraction'):
        Dimension._from_iter((1, 2, 0, 0, 0, 0, 0))  # pyright: ignore[reportArgumentType]


def test_dimension_copy_and_pickle() -> None:
    dimension = Dimension(L=Fraction(1, 3))
    assert copy(dimension) is dimension
    assert deepcopy(dimension) is dimension
    restored = pickle.loads(pickle.dumps(dimension))
    assert restored == dimension
    assert restored is not DIMENSIONLESS
    assert restored.L == Fraction(1, 3)
    restored_zero = pickle.loads(pickle.dumps(DIMENSIONLESS))
    assert restored_zero is DIMENSIONLESS
    assert DIMENSIONLESS.as_tuple() == (Fraction(0),) * 7


@pytest.mark.parametrize(
    'exponent',
    [Fraction(1, 1000001), Fraction(-1, 10**10), Fraction(10**30 + 1, 3)],
)
def test_dimension_repr_preserves_exact_exponents(exponent: Fraction) -> None:
    dimension = Dimension(Theta=exponent)
    with warnings.catch_warnings():
        warnings.simplefilter('error', InexactFloatWarning)
        restored = eval(repr(dimension), {'Dimension': Dimension, 'Fraction': Fraction})
    assert restored == dimension
    assert restored.Theta == exponent
    assert restored is not DIMENSIONLESS


@pytest.mark.parametrize('protocol', range(pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize('dimension', [DIMENSIONLESS, Dimension(L=Fraction(-1, 10**10))])
def test_dimension_pickle_protocols(dimension: Dimension, protocol: int) -> None:
    restored = pickle.loads(pickle.dumps(dimension, protocol=protocol))
    assert restored == dimension
    assert (restored is DIMENSIONLESS) == (dimension is DIMENSIONLESS)
    assert DIMENSIONLESS.as_tuple() == (Fraction(0),) * 7
    assert Dimension() is DIMENSIONLESS


@pytest.mark.parametrize('dimension', [DIMENSIONLESS, Dimension(L=Fraction(-2, 3))])
def test_dimension_copy_preserves_identity(dimension: Dimension) -> None:
    assert copy(dimension) is dimension
    assert deepcopy(dimension) is dimension
    assert DIMENSIONLESS.as_tuple() == (Fraction(0),) * 7


def test_dimension_algebra_and_exact_numeric_inputs() -> None:
    dimension = Dimension(T=Decimal('-0.125'), L=Fraction(1, 1000001))
    other = Dimension(M=Fraction(-2, 3))
    assert all(type(exponent) is Fraction for exponent in dimension.as_tuple())
    assert dimension.T == Fraction(-1, 8)
    assert dimension * DIMENSIONLESS == DIMENSIONLESS * dimension == dimension
    assert dimension / DIMENSIONLESS == dimension
    assert dimension / dimension is DIMENSIONLESS
    assert dimension * (1 / dimension) is DIMENSIONLESS
    assert dimension**0 is DIMENSIONLESS
    assert DIMENSIONLESS ** Fraction(-1, 2) is DIMENSIONLESS
    assert (dimension * other) / other == dimension
    assert (dimension ** Fraction(-2, 3)) ** Fraction(-3, 2) == dimension
    assert dimension ** Decimal('0.5') == dimension ** Fraction(1, 2)


@pytest.mark.parametrize('value', [True, False, '1/2', None, 1j])
@pytest.mark.parametrize('dimension', [DIMENSIONLESS, Dimension(L=1)])
def test_dimension_rejects_invalid_exponents(dimension: Dimension, value: object) -> None:
    with pytest.raises(TypeError, match='Cannot convert'):
        Dimension(L=value)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match='Cannot convert'):
        dimension**value  # pyright: ignore[reportUnusedExpression, reportOperatorIssue]


@pytest.mark.parametrize('value', [2, 1.0, True, None])
def test_dimension_rejects_invalid_operands(value: object) -> None:
    dimension = Dimension(L=1)
    with pytest.raises(TypeError, match='unsupported operand'):
        dimension / value  # pyright: ignore[reportUnusedExpression, reportOperatorIssue]
    with pytest.raises(TypeError, match='unsupported operand'):
        value / dimension  # pyright: ignore[reportUnusedExpression, reportOperatorIssue]
    with pytest.raises(TypeError, match='unsupported operand'):
        value * dimension  # pyright: ignore[reportUnusedExpression, reportOperatorIssue]


def test_dimension_comparison_allows_reflected_equality() -> None:
    class MatchesDimension:
        def __eq__(self, other: object) -> bool:
            return isinstance(other, Dimension)

    dimension = Dimension(L=1)
    assert dimension != 1
    assert dimension == MatchesDimension()


def test_dimension_exponent_aliases_and_order() -> None:
    dimension = Dimension(1, 2, 3, 4, 5, 6, 7)
    aliases = [
        ('T', 'time'),
        ('L', 'length'),
        ('M', 'mass'),
        ('I', 'current'),
        ('Theta', 'Θ', 'temperature'),
        ('N', 'amount'),
        ('J', 'luminous'),
    ]
    assert dimension.as_tuple() == tuple(map(Fraction, range(1, 8)))
    for exponent, names in enumerate(aliases, start=1):
        for name in names:
            assert getattr(dimension, name) == Fraction(exponent)


@pytest.mark.parametrize(
    ['value', 'error'],
    [
        (float('nan'), ValueError),
        (float('inf'), OverflowError),
        (float('-inf'), OverflowError),
        (Decimal('NaN'), ValueError),
        (Decimal('Infinity'), OverflowError),
    ],
)
@pytest.mark.parametrize('dimension', [DIMENSIONLESS, Dimension(L=1)])
def test_dimension_rejects_nonfinite_exponents(
    dimension: Dimension, value: float | Decimal, error: type[Exception]
) -> None:
    with pytest.raises(error):
        Dimension(T=value)
    with pytest.raises(error):
        dimension**value  # pyright: ignore[reportUnusedExpression]


def test_dimension_sparse_physical_projection() -> None:
    physical = Dimension(J=Fraction(2, 3)) * Dimension(T=2, Theta=Fraction(-1, 2))
    complete = Dimension._with_log(physical, Fraction(-2, 3))
    original_hash = hash(complete)
    projected = complete.physical
    assert projected == physical
    assert hash(projected) == hash(physical)
    assert complete != projected and complete.physical.physical == projected
    assert complete.as_tuple() == (2, 0, 0, 0, Fraction(-1, 2), 0, Fraction(2, 3))
    assert list(complete.as_dict()) == ['T', 'Theta', 'J']
    assert complete.as_dict() == {'T': 2, 'Theta': Fraction(-1, 2), 'J': Fraction(2, 3)}
    assert repr(projected) == 'Dimension(T=2, Theta=Fraction(-1, 2), J=Fraction(2, 3))'
    assert str(complete) == str(physical)
    assert eval(repr(complete)) == complete
    assert hash(complete) == original_hash
