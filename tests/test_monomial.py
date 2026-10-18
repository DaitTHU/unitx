import pickle
import warnings
from copy import copy, deepcopy
from decimal import Decimal
from fractions import Fraction

import pytest

from unitx.exceptions import InexactFloatWarning
from unitx.monomial import Monomial


def test_monomial_empty() -> None:
    monomial = Monomial()
    assert repr(monomial) == 'Monomial()'
    assert str(monomial) == '1'
    assert monomial.numerator == Monomial()
    assert monomial.denominator == Monomial()
    assert monomial ** Fraction(-1, 2) == monomial


def test_monomial_creation() -> None:
    monomial = Monomial({'x': 2, 'y': 3, 'deleted': 0})
    assert monomial == Monomial({'x': 2, 'y': 3})
    assert 'x' in monomial
    assert 'deleted' not in monomial
    assert 'missing' not in monomial
    assert monomial.exponent_of('x') == 2
    assert monomial.exponent_of('missing') == 0
    assert list(monomial.bases()) == ['x', 'y']
    assert list(monomial.exponents()) == [2, 3]
    assert list(monomial.components()) == [('x', 2), ('y', 3)]
    assert repr(monomial) == "Monomial({'x': 2, 'y': 3})"
    assert str(monomial) == 'x²⋅y³'


def test_monomial_numerator_and_denominator() -> None:
    monomial = Monomial({'x': -1, 'y': 1, 'z': Fraction(-3, 2)})
    assert str(monomial) == 'y/(x⋅z³ᐟ²)'
    assert monomial.numerator == Monomial({'y': 1})
    assert monomial.denominator == Monomial({'x': 1, 'z': Fraction(3, 2)})
    assert monomial.numerator / monomial.denominator == monomial


@pytest.mark.parametrize(
    ('components', 'expected'),
    [
        ({'x': 0}, '1'),
        ({'x': 1, 'y': 0}, 'x'),
        ({'x': -1}, '1/x'),
        ({'x': -1, 'y': -2}, '1/(x⋅y²)'),
        ({'x': Fraction(-1, 2)}, '1/x¹ᐟ²'),
        (
            {'x': Fraction(1, 2), 'y': 2, 'z': Fraction(-3, 2), 'w': -1},
            'x¹ᐟ²⋅y²/(z³ᐟ²⋅w)',
        ),
        ({Decimal('NaN'): -2}, '1/NaN²'),  # Monomial do not check base values
    ],
)
def test_monomial_str_format(components, expected: str) -> None:
    assert str(Monomial(components)) == expected


def test_monomial_str_view() -> None:
    monomial = Monomial({2: 1, 'b': -1, 'x': 2, 3: -2})
    before = list(monomial.components()), hash(monomial), repr(monomial)
    assert str(monomial) == '2⋅x²/(b⋅3²)'
    assert (list(monomial.components()), hash(monomial), repr(monomial)) == before


def test_monomial_input_independent() -> None:
    components = {'x': 2}
    monomial = Monomial(components)
    components['x'] = 0
    components['y'] = 1
    assert monomial == Monomial({'x': 2})


@pytest.mark.parametrize('monomial', [Monomial(), Monomial({'x': Fraction(1, 3)})])
def test_monomial_immutable(monomial: Monomial[str]) -> None:
    original = dict(monomial.components())
    original_hash = hash(monomial)
    lookup = {monomial: 'unchanged'}
    with pytest.raises(TypeError):
        monomial._components['x'] = Fraction(2)  # pyright: ignore[reportIndexIssue]
    with pytest.raises(AttributeError, match='immutable'):
        monomial._components = {}  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(AttributeError, match='immutable'):
        del monomial._components
    assert dict(monomial.components()) == original
    assert hash(monomial) == original_hash
    assert lookup[monomial] == 'unchanged'


def test_monomial_hash() -> None:
    first = Monomial({'x': Fraction(1, 3), 'y': -2})
    second = Monomial({'zero': 0, 'y': -2, 'x': Fraction(1, 3)})
    assert first == second
    assert hash(first) == hash(second)
    assert {first, second} == {first}
    assert hash(Monomial({'x': 0})) == hash(Monomial())


@pytest.mark.parametrize('components', [{}, {'x': Fraction(1, 1000001), 'y': -2}])
def test_monomial_copy(components: dict[str, int | Fraction]) -> None:
    original = Monomial(components)
    assert copy(original) is original
    cloned = deepcopy(original)
    assert cloned == original
    assert hash(cloned) == hash(original)
    with pytest.raises(AttributeError, match='immutable'):
        cloned._components = {}  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize('monomial', [Monomial(), Monomial({'x': Fraction(-1, 10**10)})])
def test_monomial_pickle(monomial: Monomial[str]) -> None:
    restored = pickle.loads(pickle.dumps(monomial))
    assert restored == monomial
    assert hash(restored) == hash(monomial)
    with pytest.raises(TypeError):
        restored._components['x'] = Fraction(2)
    with pytest.raises(AttributeError, match='immutable'):
        restored._components = {}


@pytest.mark.parametrize(
    'exponent',
    [Fraction(1, 7), Fraction(1, 1000001), Fraction(-1, 10**10), Fraction(10**30 + 1, 3)],
)
def test_monomial_eval_repr(exponent: Fraction) -> None:
    monomial = Monomial({'x': exponent, 'y': -2})
    with warnings.catch_warnings():
        warnings.simplefilter('error', InexactFloatWarning)
        assert eval(repr(monomial)) == monomial


def test_monomial_operations() -> None:
    first = Monomial({'x': Fraction(1, 3), 'y': -2})
    second = Monomial({'x': Fraction(-1, 3), 'z': 2})
    assert first * second == Monomial({'y': -2, 'z': 2})
    assert first / second == Monomial({'x': Fraction(2, 3), 'y': -2, 'z': -2})
    assert first == Monomial({'x': Fraction(1, 3), 'y': -2})
    assert second == Monomial({'x': Fraction(-1, 3), 'z': 2})
    assert first * second == second * first
    assert (first * second) / second == first
    assert first / first == Monomial()
    assert first * (1 / first) == Monomial()
    assert first * Monomial() == first
    assert first**0 == Monomial()
    assert list((first / second).bases()) == ['x', 'y', 'z']


def test_monomial_inplace_operations() -> None:
    monomial = Monomial({'x': Fraction(1, 3), 'y': -2})
    original = monomial
    monomial *= monomial
    assert monomial == Monomial({'x': Fraction(2, 3), 'y': -4})
    assert original == Monomial({'x': Fraction(1, 3), 'y': -2})
    alias = monomial
    monomial /= monomial
    assert monomial == Monomial()
    assert alias == Monomial({'x': Fraction(2, 3), 'y': -4})
    monomial = original
    monomial **= Fraction(-2, 3)
    assert monomial == Monomial({'x': Fraction(-2, 9), 'y': Fraction(4, 3)})
    assert original == Monomial({'x': Fraction(1, 3), 'y': -2})


def test_monomial_exact_exponents() -> None:
    monomial = Monomial({'x': Decimal('0.125'), 'y': Fraction(1, 1000001)})
    assert monomial.exponent_of('x') == Fraction(1, 8)
    assert monomial.exponent_of('y') == Fraction(1, 1000001)
    assert monomial ** Decimal('0.5') == monomial ** Fraction(1, 2)
    assert (monomial ** Fraction(-2, 3)) ** Fraction(-3, 2) == monomial


@pytest.mark.parametrize('value', [True, False, '1/2'])
def test_monomial_invalid_types(value: object) -> None:
    with pytest.raises(TypeError):
        Monomial({'x': value})  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        Monomial({'x': 1}) ** value  # pyright: ignore[reportUnusedExpression, reportOperatorIssue]


@pytest.mark.parametrize(
    ('value', 'error'), [(float('nan'), ValueError), (float('inf'), OverflowError)]
)
def test_monomial_invalid_exponents(value: float, error: type[Exception]) -> None:
    with pytest.raises(error):
        Monomial({'x': value})
    with pytest.raises(error):
        Monomial({'x': 1}) ** value  # pyright: ignore[reportUnusedExpression]


def test_monomial_invalid_components() -> None:
    with pytest.raises(TypeError, match='mapping'):
        Monomial({'x', 'y'})  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize('value', [2, 1.0, True])
def test_monomial_invalid_operands(value: object) -> None:
    monomial = Monomial({'x': 2})
    with pytest.raises(TypeError, match='unsupported operand'):
        monomial * value  # pyright: ignore[reportUnusedExpression, reportOperatorIssue]
    with pytest.raises(TypeError, match='unsupported operand'):
        value * monomial  # pyright: ignore[reportUnusedExpression, reportOperatorIssue]
    with pytest.raises(TypeError, match='unsupported operand'):
        monomial / value  # pyright: ignore[reportUnusedExpression, reportOperatorIssue]
    with pytest.raises(TypeError, match='unsupported operand'):
        value / monomial  # pyright: ignore[reportUnusedExpression, reportOperatorIssue]


def test_monomial_float_exponents() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error', InexactFloatWarning)
        assert Monomial({'x': 1e-10}).exponent_of('x') == Fraction(1, 10**10)
        monomial = Monomial({'x': 0.1})
        assert monomial == Monomial({'x': Fraction(1, 10)})
        assert monomial**10 == Monomial({'x': 1})
        assert (Monomial({'x': 1}) ** 0.1) ** 10 == Monomial({'x': 1})
