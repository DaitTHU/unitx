from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from typing import Literal, final

from .utils.number import ZERO, common_fraction
from .utils.special_char import superscript

__all__ = ['DIMENSIONLESS', 'Dimension']

_SYMBOL = ('T', 'L', 'M', 'I', 'Θ', 'N', 'J')
_SYMBOL_ASCII = ('T', 'L', 'M', 'I', 'Theta', 'N', 'J')
_LEN = len(_SYMBOL)
_ALLZERO = (ZERO,) * _LEN


@final
class Dimension:
    '''
    Implementation of physical dimensions.
    - `T`: time
    - `L`: length
    - `M`: mass
    - `I`: current
    - `Θ`: temperature (use ASCII alias `Theta` for code)
    - `N`: amount
    - `J`: luminous

    Construct
    ---
    >>> dim_velocity = Dimension(L=1, T=-1)
    >>> dim_force = Dimension(M=1, L=1, T=-2)

    Attributes
    ---
    You can access the exponents of each dimension by symbols or full names.
    >>> dim_force.T
    -2
    >>> dim_force.mass
    1

    Operations
    ---
    You can perform multiplication, division, and exponentiation on dimensions.
    >>> dim_force * dim_velocity
    Dimension(T=-3, L=2, M=1)
    >>> dim_velocity / Dimension(T=1)
    Dimension(T=-2, L=1)
    >>> dim_force ** 2
    Dimension(T=-4, L=2, M=2)
    >>> 1 / dim_force
    Dimension(T=2, L=-1, M=-1)
    '''

    __slots__ = ('_exponents',)
    _exponents: tuple[Fraction, ...]  # Exponents for T, L, M, I, Θ, N, J

    def __new__(cls, T=0, L=0, M=0, I=0, Theta=0, N=0, J=0) -> Dimension:
        exponents = tuple(map(common_fraction, (T, L, M, I, Theta, N, J)))
        if not any(exponents):
            return DIMENSIONLESS
        obj = object.__new__(cls)
        object.__setattr__(obj, '_exponents', exponents)
        return obj

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError('Dimension is immutable and cannot set attribute.')

    def __delattr__(self, name: str) -> None:
        raise AttributeError('Dimension is immutable and cannot delete attribute.')

    @classmethod
    def _from_iter(cls, iterable: Iterable[Fraction], /) -> Dimension:
        '''Constructor from an iterable of exponents. Internal use only.'''
        exponents = tuple(iterable)
        assert len(exponents) == _LEN, f'Expected {_LEN} exponents, got {len(exponents)}.'
        assert all(isinstance(v, Fraction) for v in exponents), 'All exponents must be of type Fraction.'
        if not any(exponents):
            return DIMENSIONLESS
        obj = object.__new__(cls)
        object.__setattr__(obj, '_exponents', exponents)
        return obj

    def __mul__(self, other: Dimension) -> Dimension:
        if not isinstance(other, Dimension):
            return NotImplemented
        return self._from_iter(a + b for a, b in zip(self._exponents, other._exponents))

    def __truediv__(self, other: Dimension) -> Dimension:
        if not isinstance(other, Dimension):
            return NotImplemented
        return self._from_iter(a - b for a, b in zip(self._exponents, other._exponents))

    def __pow__(self, power: Fraction | float) -> Dimension:
        power = common_fraction(power, stacklevel=2)
        if not isinstance(power, (int, Fraction)):
            return NotImplemented
        return self._from_iter(a * power for a in self._exponents)

    def __rtruediv__(self, other: Literal[1]) -> Dimension:
        if other == 1:
            return self._from_iter(-a for a in self._exponents)
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dimension):
            return NotImplemented
        return self._exponents == other._exponents

    def __hash__(self) -> int: return hash(self._exponents)

    def as_tuple(self) -> tuple[Fraction, ...]: return self._exponents

    @staticmethod
    def __unpack_exponents():
        def __getter(index: int):
            return lambda self: self._exponents[index]
        return (property(__getter(i)) for i in range(_LEN))

    T, L, M, I, Θ, N, J = __unpack_exponents()
    Theta = Θ  # Alias for Θ
    time, length, mass, current, temperature, amount, luminous = T, L, M, I, Θ, N, J

    def __repr__(self) -> str:
        para = ', '.join(f'{s}={e}' for s, e in zip(_SYMBOL_ASCII, self._exponents) if e)
        return f'{type(self).__name__}({para})'

    def __str__(self) -> str:
        return ''.join(s + superscript(e) for s, e in zip(_SYMBOL, self._exponents) if e) or '1'


DIMENSIONLESS = object.__new__(Dimension)
object.__setattr__(DIMENSIONLESS, '_exponents', _ALLZERO)
