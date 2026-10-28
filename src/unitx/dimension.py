from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from typing import Literal

from ._formatting.unicode import superscript
from ._immutable import _Immutable
from ._numerics import Fractionable, as_exponent, as_int_if_integral
from .monomial import Monomial, _MutableMonomial

__all__ = ['DIMENSIONLESS', 'Dimension']

_SYMBOLS = ('T', 'L', 'M', 'I', 'Θ', 'N', 'J')
_ASCII_SYMBOLS = ('T', 'L', 'M', 'I', 'Theta', 'N', 'J')
_LEN = len(_SYMBOLS)


class Dimension(_Immutable):
    """Immutable algebraic dimensions with a seven-axis physical projection.

    - `T`: time
    - `L`: length
    - `M`: mass
    - `I`: current
    - `Θ`: temperature (ASCII alias `Theta`)
    - `N`: amount
    - `J`: luminous

    Zero exponents represent the `DIMENSIONLESS` singleton. Use Fraction or
    Decimal for exact exponents; floats use their shortest decimal text,
    without a denominator limit, consistently with Monomial, Factor, and Unit.

    >>> dim_velocity = Dimension(L=1, T=-1)
    >>> dim_force = Dimension(M=1, L=1, T=-2)
    >>> dim_force.T
    Fraction(-2, 1)
    >>> dim_force.mass
    Fraction(1, 1)
    >>> dim_force * dim_velocity
    Dimension(T=-3, L=2, M=1)
    >>> dim_velocity / Dimension(T=1)
    Dimension(T=-2, L=1)
    >>> dim_force ** 2
    Dimension(T=-4, L=2, M=2)
    >>> 1 / dim_force
    Dimension(T=2, L=-1, M=-1)

    """

    __slots__ = ('_monomial',)
    _monomial: Monomial[str]

    def __new__(
        cls,
        T: Fractionable = 0,
        L: Fractionable = 0,
        M: Fractionable = 0,
        I: Fractionable = 0,  # noqa: E741
        Theta: Fractionable = 0,
        N: Fractionable = 0,
        J: Fractionable = 0,
    ) -> Dimension:
        return cls._from_monomial(Monomial(dict(zip(_ASCII_SYMBOLS, (T, L, M, I, Theta, N, J)))))

    @classmethod
    def _from_monomial(cls, monomial: Monomial[str], /) -> Dimension:
        if not monomial.components():
            return DIMENSIONLESS
        obj = object.__new__(cls)
        object.__setattr__(obj, '_monomial', monomial)
        return obj

    @classmethod
    def _product(cls, terms: Iterable[tuple[Dimension, Fraction]], /) -> Dimension:
        """Combine powered dimensions without immutable intermediate products."""
        product = _MutableMonomial[str]()
        for dimension, exponent in terms:
            product.multiply(dimension._monomial, exponent)
        return cls._from_monomial(product.freeze())

    @classmethod
    def _from_iter(cls, exponents: Iterable[Fraction], /) -> Dimension:
        """Construct the seven-axis physical projection. Internal use only."""
        exponents = tuple(exponents)
        assert len(exponents) == _LEN, f'Expected {_LEN} exponents, got {len(exponents)}.'
        assert all(type(exp) is Fraction for exp in exponents), 'All exponents must be Fraction.'
        return cls._from_monomial(Monomial(dict(zip(_ASCII_SYMBOLS, exponents))))

    @classmethod
    def _with_log(cls, physical: Dimension, exponent: Fractionable = 1) -> Dimension:
        """Attach the internal logarithmic axis at a model/serialization boundary."""
        return cls._from_monomial(physical._monomial * Monomial({'_log': exponent}))

    @property
    def physical(self) -> Dimension:
        """Project onto the seven physical axes, ignoring internal algebraic axes."""
        if self._monomial.exponent_of('_log') == 0:
            return self
        return self._from_iter(self.as_tuple())

    def __repr__(self) -> str:
        arguments = ', '.join(
            f'{sym}={as_int_if_integral(exp)!r}'
            for sym, exp in zip(_ASCII_SYMBOLS, self.as_tuple())
            if exp
        )
        physical = f'{type(self).__name__}({arguments})'
        exponent = self._monomial.exponent_of('_log')
        return f'Dimension._with_log({physical}, {exponent!r})' if exponent else physical

    def __str__(self) -> str:
        formatted_expression = ''.join(
            sym + superscript(exp) for sym, exp in zip(_SYMBOLS, self.as_tuple()) if exp
        )
        return formatted_expression or '1'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dimension):
            return NotImplemented
        return self._monomial == other._monomial

    def __hash__(self) -> int:
        return hash(self._monomial)

    def __copy__(self) -> Dimension:
        return self  # Dimensions are immutable

    def __deepcopy__(self, memo: dict[int, object]) -> Dimension:
        return self

    def __reduce__(self) -> tuple[object, tuple[Monomial[str]]]:
        return type(self)._from_monomial, (self._monomial,)

    def __mul__(self, other: Dimension) -> Dimension:
        if not isinstance(other, Dimension):
            return NotImplemented
        return self._from_monomial(self._monomial * other._monomial)

    def __truediv__(self, other: Dimension) -> Dimension:
        if not isinstance(other, Dimension):
            return NotImplemented
        return self._from_monomial(self._monomial / other._monomial)

    def __pow__(self, power: Fractionable) -> Dimension:
        return self._from_monomial(self._monomial ** as_exponent(power))

    def __rtruediv__(self, other: Literal[1]) -> Dimension:
        if type(other) is int and other == 1:
            return self._from_monomial(1 / self._monomial)
        return NotImplemented

    @property
    def T(self) -> Fraction:
        return self._monomial.exponent_of('T')

    @property
    def L(self) -> Fraction:
        return self._monomial.exponent_of('L')

    @property
    def M(self) -> Fraction:
        return self._monomial.exponent_of('M')

    @property
    def I(self) -> Fraction:  # noqa: E743
        return self._monomial.exponent_of('I')

    @property
    def Θ(self) -> Fraction:
        return self._monomial.exponent_of('Theta')

    @property
    def N(self) -> Fraction:
        return self._monomial.exponent_of('N')

    @property
    def J(self) -> Fraction:
        return self._monomial.exponent_of('J')

    Theta = Θ  # Alias for Θ
    time, length, mass, current, temperature, amount, luminous = T, L, M, I, Θ, N, J

    def as_tuple(self) -> tuple[Fraction, ...]:
        return tuple(self._monomial.exponent_of(axis) for axis in _ASCII_SYMBOLS)

    def as_dict(self) -> dict[str, Fraction]:
        return {sym: exp for sym, exp in zip(_ASCII_SYMBOLS, self.as_tuple()) if exp}


DIMENSIONLESS = object.__new__(Dimension)
object.__setattr__(DIMENSIONLESS, '_monomial', Monomial())


_LOG_DIMENSION = Dimension._with_log(DIMENSIONLESS)
