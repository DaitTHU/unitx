from __future__ import annotations

import math
import operator
from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from typing import Literal, Self

from .monomial import Monomial
from .utils.number import common_fraction, factorfrac, factorint, isprime


class SymbolicNumber:
    '''
    A symbolic representation of a mathematical constant.

    The constant is identified by a specific symbol and value. Its value is not
    expressible as a product of prime numbers raised to rational exponents.
    '''
    __slots__ = ('_symbol', '_value')

    def __new__(cls, symbol: str, value: float) -> SymbolicNumber:
        if not isinstance(symbol, str):
            raise TypeError(f'Symbol must be str, got {type(symbol)!r}.')
        if not isinstance(value, float):
            raise TypeError(f'Value must be float, got {type(value)!r}.')
        if not math.isfinite(value) or value <= 0:
            raise ValueError('SymbolicNumber value must be positive.')
        if symbol == '':
            raise ValueError('SymbolicNumber symbol cannot be empty.')
        obj = super().__new__(cls)
        obj._symbol, obj._value = symbol, value
        return obj

    @property
    def symbol(self) -> str: return self._symbol
    @property
    def value(self) -> float: return self._value

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self._symbol!r}, {self._value!r})'

    def __str__(self) -> str: return self._symbol

    def __float__(self) -> float: return self._value

    def __hash__(self) -> int: return hash((self._symbol, self._value))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SymbolicNumber):
            return self._symbol == other._symbol and self._value == other._value
        if isinstance(other, (int, float)):
            return self._value == other
        return NotImplemented

    @staticmethod
    def _comparison_op(op) -> Callable[[SymbolicNumber, object], bool]:
        def comparison(self: SymbolicNumber, other: object) -> bool:
            if isinstance(other, SymbolicNumber):
                return op(self._value, other._value)
            if isinstance(other, (int, float)):
                return op(self._value, other)
            return NotImplemented
        return comparison

    __lt__ = _comparison_op(operator.lt)
    __le__ = _comparison_op(operator.le)
    __gt__ = _comparison_op(operator.gt)
    __ge__ = _comparison_op(operator.ge)

    @staticmethod
    def _arithmetic_op(op) -> Callable[[SymbolicNumber, object], Factor]:
        def arithmetic(self: SymbolicNumber, other: object) -> Factor:
            if isinstance(other, (int, float, Fraction, SymbolicNumber)):
                return op(Factor(self), Factor(other))
            return NotImplemented
        return arithmetic

    __mul__ = __rmul__ = _arithmetic_op(operator.mul)
    __truediv__ = _arithmetic_op(operator.truediv)
    __rtruediv__ = _arithmetic_op(lambda self, other: other / self)

    def __pow__(self, other: Fraction | float) -> Factor:
        if not isinstance(other, (int, float, Fraction)):
            return NotImplemented
        return Factor._from_monomial(Monomial({self: common_fraction(other, stacklevel=2)}))


class Factor:
    '''
    An exact representation of scaling factors for unit conversions.

    This class uses `Monomial` to represent factor number absolutely precisely.
    Instead of using floats which suffer from precision drift during repeated 
    multiplications and roots, `Factor` decomposes numbers into a mapping of
    prime bases to their rational exponents.

    Algebraic Mapping:
    - Base Elements: Prime integers (e.g., 2, 3, 5) or `SymbolicNumber`s (e.g., π).
    - Exponents: `Fraction` representing powers.

    Examples
    --------
    >>> Factor(12)
    Factor.from_dict({2: 2, 3: 1})
    >>> Factor(2.54)  # 127 / 50
    Factor.from_dict({127: 1, 2: -1, 5: -2})
    >>> str(Factor(3) / 10)
    '3/10'
    >>> float(Factor(2.54))
    2.54
    '''
    __slots__ = ('_monomial', '_value')
    _monomial: Monomial[int | SymbolicNumber]
    _value: float
    
    def __init__(self, value: float | Fraction | Decimal | SymbolicNumber, /) -> None:
        try:
            if value <= 0:
                raise ValueError('Factor must be positive.')
        except TypeError:
            raise TypeError('Factor must be a number.')
        if isinstance(value, int):
            self._monomial = Monomial(factorint(value))
            self._value = float(value)
            return
        elif isinstance(value, SymbolicNumber):
            self._monomial = Monomial({value: 1})
            self._value = float(value)
            return
        elif isinstance(value, (float, Decimal)):
            value = common_fraction(value, stacklevel=2)
        if isinstance(value, Fraction):
            self._monomial = Monomial(factorfrac(value))  # type: ignore
            self._value = float(value)
            return
        raise TypeError(f'Expected int, float, Fraction, Decimal or SymbolicNumber, got {type(value)!r}.')

    @classmethod
    def _from_monomial(cls, monomial: Monomial[int | SymbolicNumber], /) -> Self:
        '''Direct constructor from Monomial without copy. Internal use only.'''
        obj = object.__new__(cls)
        obj._monomial = monomial
        return obj

    @classmethod
    def from_dict(cls, elements: dict[int | SymbolicNumber, int | Fraction], /) -> Self:
        '''Construct a Factor from a dictionary of base elements to their exponents.'''
        monomial = Monomial(elements)
        if not all(isinstance(base, (int, SymbolicNumber)) for base in monomial.bases()):
            raise TypeError('All bases must be int or SymbolicNumber.')
        if not all(isinstance(exp, Fraction) for exp in monomial.exponents()):
            raise TypeError('All exponents must be Fraction.')
        if not all(exp > 0 for base, exp in monomial.components() if isinstance(base, int)):
            raise ValueError('All integer bases must have positive exponents.')
        if not all(isprime(base) for base in monomial.bases() if isinstance(base, int)):
            raise ValueError('All integer bases must be prime numbers.')
        return cls._from_monomial(monomial)

    def decompose(self, *, extract_integer=True, rationalization_denominator=False) -> tuple[Fraction, Factor]:
        '''
        Decompose the factor into a rational part and an irrational part.
        
        Parameters
        ----------
        extract_integer : bool, optional
            If True, extract integer part from the fraction exponents of integer bases.
            This is the difference between 2^(3/2) and 2 * 2^(1/2).
        rationalization_denominator : bool, optional
            If True, exponents of integer bases in the irrational part are all
            positive to rationalize the denominator. This is the difference
            between 2^(-3/2) and 2^(1/2) / 4.

        Returns
        -------
        tuple[Fraction, Factor]
            A tuple containing the rational part and the irrational part.
        '''
        rational = Fraction(1)
        irrational = {}
        for base, exponent in self._monomial.components():
            if isinstance(base, SymbolicNumber):
                irrational[base] = exponent
                continue
            if exponent.denominator == 1:
                rational *= Fraction(base) ** exponent.numerator
                continue
            if extract_integer or (rationalization_denominator and exponent > 0):
                integer_exponent = math.floor(exponent) \
                    if rationalization_denominator else int(exponent)
                rational *= Fraction(base) ** integer_exponent
                irrational[base] = exponent - integer_exponent
            else:
                irrational[base] = exponent
        return rational, Factor._from_monomial(Monomial._from_dict(irrational))

    @property
    def value(self) -> float:
        if hasattr(self, '_value_cache'):
            return self._value
        rational, irrational = self.decompose()
        if not irrational._monomial:
            return float(rational)
        float_factors = [float(base) ** float(exponent) for base, exponent in irrational._monomial.components()]
        float_factors.append(float(rational))
        float_factors.sort()
        # balance multiplication to minimize floating-point error
        self._value = 1.0
        lo, hi = 0, len(float_factors) - 1
        while lo <= hi:
            if self._value > 1.0:
                self._value *= float_factors[hi]
                hi -= 1
            else:
                self._value *= float_factors[lo]
                lo += 1
        return self._value

    def __float__(self) -> float: return self.value

    def format(self, mul: Literal['⋅', '*', ' * ', ' '] = '⋅',
               exp: Literal['sup', '^', '**'] = 'sup', frac: bool = True, 
               order: Literal['insertion', 'ascending', 'descending'] = 'insertion',
               compact: bool = True) -> str:
        '''
        Format the factor as a string.
        - `mul`, `exp`, `frac`, and `order` are passed to the `Monomial.format()` method.
        - `compact=True` indicates that the factor is formatted in a compact form,
            with the rational part extracted as a single fraction rather than as
            a product of factors with integer exponents.
        >>> Factor(12).format()
        '12'
        >>> Factor(12).format(compact=False)
        '2²⋅3'
        '''
        if not compact:
            return self._monomial.format(mul=mul, exp=exp, frac=frac, order=order)
        rational, irrational = self.decompose(rationalization_denominator=True)
        display_elements = {}
        if rational.numerator > 1:
            display_elements[rational.numerator] = Fraction(1)
        if rational.denominator > 1:
            display_elements[rational.denominator] = Fraction(-1)
        display_elements.update(irrational._monomial.components())
        return Monomial._from_dict(display_elements).format(mul=mul, exp=exp, frac=frac, order=order)

    def __str__(self) -> str: return self.format()

    def __repr__(self) -> str:
        kv = ', '.join(f'{base!r}: {exponent!r}' for base, exponent in self._monomial.components())
        return f'{type(self).__name__}.from_dict({{{kv}}})'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Factor):
            return self._monomial == other._monomial
        if isinstance(other, (int, float, Fraction)):
            return self.value == float(other)
        return NotImplemented

    def __mul__(self, other: Fraction | float | SymbolicNumber | Factor) -> Factor:
        if isinstance(other, (int, float, Fraction, SymbolicNumber)):
            other = Factor(other)
        if not isinstance(other, Factor):
            return NotImplemented
        return Factor._from_monomial(self._monomial * other._monomial)

    __rmul__ = __mul__

    def __truediv__(self, other: Fraction | float | SymbolicNumber | Factor) -> Factor:
        if other == 0:
            raise ZeroDivisionError('Division by zero.')
        if isinstance(other, (int, float, Fraction, SymbolicNumber)):
            other = Factor(other)
        if not isinstance(other, Factor):
            return NotImplemented
        return Factor._from_monomial(self._monomial / other._monomial)

    def __rtruediv__(self, other: Fraction | float | SymbolicNumber) -> Factor:
        if isinstance(other, (int, float, Fraction, SymbolicNumber)):
            return Factor(other) / self
        return NotImplemented

    def __pow__(self, other: Fraction | float) -> Factor:
        if other == 0:
            return Factor(1)
        other = common_fraction(other, stacklevel=2)
        return Factor._from_monomial(self._monomial ** other)

    def __call__(self, magnitude):
        '''Mapping `factor(x) = factor * x`'''
        return self.value * magnitude


ONE = Factor(1)
TWO = Factor(2)
TEN = Factor(10)
PI = SymbolicNumber('π', 3.141592653589793)
'''The ratio of the circumference of a circle to its diameter.'''
