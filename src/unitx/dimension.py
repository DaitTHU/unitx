from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from typing import Literal

from ._numerics import Fractionable, as_fraction, as_int_if_integral
from ._unicode import superscript

__all__ = ['DIMENSIONLESS', 'Dimension']

_SYMBOLS = ('T', 'L', 'M', 'I', 'Θ', 'N', 'J')
_ASCII_SYMBOLS = ('T', 'L', 'M', 'I', 'Theta', 'N', 'J')
_LEN = len(_SYMBOLS)


class Dimension:
    """Immutable physical dimensions represented by seven rational exponents.

    - `T`: time
    - `L`: length
    - `M`: mass
    - `I`: current
    - `Θ`: temperature (ASCII alias `Theta`)
    - `N`: amount
    - `J`: luminous

    Zero exponents represent the `DIMENSIONLESS` singleton. Use Fraction or
    Decimal for exact exponents; floats may be approximated with a warning.

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

    __slots__ = ('_exponents',)
    _exponents: tuple[Fraction, ...]  # Exponents for (T, L, M, I, Θ, N, J)

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
        exponents = tuple(as_fraction(exp) for exp in (T, L, M, I, Theta, N, J))
        if not any(exponents):
            return DIMENSIONLESS
        obj = object.__new__(cls)
        object.__setattr__(obj, '_exponents', exponents)
        return obj

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError('Dimension objects are immutable.')

    def __delattr__(self, name: str) -> None:
        raise AttributeError('Dimension objects are immutable.')

    @classmethod
    def _from_iter(cls, exponents: Iterable[Fraction], /) -> Dimension:
        """Construct from an iterable of exponents. Internal use only."""
        exponents = tuple(exponents)
        assert len(exponents) == _LEN, f'Expected {_LEN} exponents, got {len(exponents)}.'
        assert all(type(exp) is Fraction for exp in exponents), 'All exponents must be Fraction.'
        if not any(exponents):
            return DIMENSIONLESS
        obj = object.__new__(cls)
        object.__setattr__(obj, '_exponents', exponents)
        return obj

    def __repr__(self) -> str:
        arguments = ', '.join(
            f'{sym}={as_int_if_integral(exp)!r}'
            for sym, exp in zip(_ASCII_SYMBOLS, self._exponents)
            if exp
        )
        return f'{type(self).__name__}({arguments})'

    def __str__(self) -> str:
        formatted_expression = ''.join(
            sym + superscript(exp) for sym, exp in zip(_SYMBOLS, self._exponents) if exp
        )
        return formatted_expression or '1'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dimension):
            return NotImplemented
        return self._exponents == other._exponents

    def __hash__(self) -> int:
        return hash(self._exponents)

    def __copy__(self) -> Dimension:
        return self  # Dimensions are immutable

    def __deepcopy__(self, memo: dict[int, object]) -> Dimension:
        return self

    def __reduce__(self) -> tuple[type[Dimension], tuple[Fraction, ...]]:
        return type(self), self._exponents

    def __mul__(self, other: Dimension) -> Dimension:
        if not isinstance(other, Dimension):
            return NotImplemented
        return self._from_iter(a + b for a, b in zip(self._exponents, other._exponents))

    def __truediv__(self, other: Dimension) -> Dimension:
        if not isinstance(other, Dimension):
            return NotImplemented
        return self._from_iter(a - b for a, b in zip(self._exponents, other._exponents))

    def __pow__(self, power: Fractionable) -> Dimension:
        power_as_fraction = as_fraction(power)
        return self._from_iter(exp * power_as_fraction for exp in self._exponents)

    def __rtruediv__(self, other: Literal[1]) -> Dimension:
        if type(other) is int and other == 1:
            return self._from_iter(-exp for exp in self._exponents)
        return NotImplemented

    @property
    def T(self) -> Fraction:
        return self._exponents[0]

    @property
    def L(self) -> Fraction:
        return self._exponents[1]

    @property
    def M(self) -> Fraction:
        return self._exponents[2]

    @property
    def I(self) -> Fraction:  # noqa: E743
        return self._exponents[3]

    @property
    def Θ(self) -> Fraction:
        return self._exponents[4]

    @property
    def N(self) -> Fraction:
        return self._exponents[5]

    @property
    def J(self) -> Fraction:
        return self._exponents[6]

    Theta = Θ  # Alias for Θ
    time, length, mass, current, temperature, amount, luminous = T, L, M, I, Θ, N, J

    def as_tuple(self) -> tuple[Fraction, ...]:
        return self._exponents

    def as_dict(self) -> dict[str, Fraction]:
        return {sym: exp for sym, exp in zip(_ASCII_SYMBOLS, self._exponents) if exp}


DIMENSIONLESS = object.__new__(Dimension)
object.__setattr__(DIMENSIONLESS, '_exponents', (Fraction(0),) * _LEN)
