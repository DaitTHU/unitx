from __future__ import annotations

from typing import TYPE_CHECKING

from ._numerics import Fractionable, Magnitude
from .dimension import DIMENSIONLESS, Dimension
from .factor import ONE, Factor
from .monomial import Monomial
from .singleunit import SingleUnit
from .unitparser import UnitParser

if TYPE_CHECKING:
    from .quantity import Quantity


class Unit:
    """A physical unit represented by a formal product of single units.

    Accept a string in `UnitParser` syntax, such as `'kg m/s2'`,
    `'m/(s A)'`, or `'m^(1/2)'`. Registered symbols, singular English names,
    prefixes, and aliases are resolved by `SingleUnit` and are case-sensitive.
    The default empty string, whitespace-only strings, and `'1'` represent
    the multiplicative identity. See `UnitParser` for the full grammar.

    Multiplication and division combine Unit operands, and powers accept int,
    Fraction, Decimal, or float exponents. Operations return new units and
    cancel matching formal factors. Decimal strings, Fraction, and Decimal
    exponents remain exact; float exponents may be approximated with an
    `InexactFloatWarning`.

    A supported scalar magnitude multiplied by a Unit constructs a Quantity
    with that unit. Dividing a magnitude by a Unit constructs a Quantity with
    the reciprocal unit. Both preserve the original magnitude and its type
    without scaling. NumPy scalars, arrays, and NumPy operations are not
    supported yet; NumPy integration is planned.

    Equality and hashing compare canonical unit identities and rational
    exponents, regardless of factor insertion order. Aliases compare equal,
    while prefixed units such as m and cm remain distinct. Equal dimensions
    and scaling factors alone do not imply equal formal unit expressions.

    `factor` is the exact multiplicative scale from this unit to its
    corresponding coherent SI unit. Its `value` is an int or Fraction when
    exactly rational, otherwise a float approximation. For units of the same
    dimension, `(source.factor / target.factor)(magnitude)` converts a
    magnitude from source to target. Mass is referenced to kg.

    `is_dimensionless()` checks only the physical dimension. A dimensionless
    expression can retain a scale other than one, as in `'km/m'`.
    `symbol` and str() provide readable Unicode display; repr() provides a
    reconstructible expression using the canonical unit symbols.

    >>> unit = Unit('kilogram metre / second2')
    >>> unit.symbol, unit.factor.value, unit.dimension
    ('kg⋅m/s²', 1, Dimension(T=-2, L=1, M=1))
    >>> Unit('m') / Unit('s')
    Unit('m * s**-1')
    >>> Unit('m*s') == Unit('s*m')
    True
    >>> Unit('m') == Unit('cm')
    False
    >>> from fractions import Fraction
    >>> root = Unit('mm') ** Fraction(1, 3)
    >>> root.factor.value
    Fraction(1, 10)
    >>> root**3 == Unit('mm')
    True
    >>> scaled = Unit('km/m')
    >>> scaled.is_dimensionless(), scaled.factor.value
    (True, 1000)
    >>> source, target = Unit('g/cm3'), Unit('kg/m3')
    >>> (source.factor / target.factor)(2)
    2000
    >>> eval(repr(unit)) == unit
    True
    >>> 2 * Unit('cm')
    Quantity(2, Unit('cm'))
    """

    __slots__ = ('_monomial', '_dimension', '_factor', '_symbol')
    _monomial: Monomial[SingleUnit]
    _symbol: str
    _factor: Factor
    _dimension: Dimension

    def __new__(cls, symbol: str = '', /) -> Unit:
        if not isinstance(symbol, str):
            raise TypeError(f'Unit symbol must be str, got {type(symbol)}.')
        return cls._from_monomial(UnitParser(symbol).parse())

    @classmethod
    def _from_monomial(cls, monomial: Monomial[SingleUnit], /) -> Unit:
        """Direct constructor from Monomial without copy. Internal use only."""
        obj = object.__new__(cls)
        obj._monomial = monomial
        obj._symbol = str(monomial)
        obj._factor = ONE
        obj._dimension = DIMENSIONLESS
        for unit, exponent in monomial.components():
            obj._factor *= unit.factor**exponent
            obj._dimension *= unit.dimension**exponent
        return obj

    def __repr__(self) -> str:
        expression = ' * '.join(
            f'{unit}'
            if exponent == 1
            else f'{unit}**{exponent}'
            if exponent.denominator == 1
            else f'{unit}**({exponent})'
            for unit, exponent in self._monomial.components()
        )
        return f'{type(self).__name__}({expression!r})'

    def __str__(self) -> str:
        return self.symbol

    def __hash__(self) -> int:
        return hash(self._monomial)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Unit):
            return NotImplemented
        return self._monomial == other._monomial

    def __mul__(self, other: Unit) -> Unit:
        if not isinstance(other, Unit):
            return NotImplemented
        return self._from_monomial(self._monomial * other._monomial)

    def __truediv__(self, other: Unit) -> Unit:
        if not isinstance(other, Unit):
            return NotImplemented
        return self._from_monomial(self._monomial / other._monomial)

    def __rmul__[T: Magnitude](self, other: T) -> Quantity[T]:
        from .quantity import Quantity

        return Quantity(other, self)

    def __rtruediv__[T: Magnitude](self, other: T) -> Quantity[T]:
        from .quantity import Quantity

        return Quantity(other, self**-1)

    def __pow__(self, exponent: Fractionable) -> Unit:
        if not isinstance(exponent, Fractionable):
            return NotImplemented
        return self._from_monomial(self._monomial**exponent)

    @property
    def symbol(self) -> str:
        """Return the readable unit expression with Unicode multiplication and powers."""
        return self._symbol

    @property
    def factor(self) -> Factor:
        """Return the exact multiplicative scale to the corresponding coherent SI unit."""
        return self._factor

    @property
    def dimension(self) -> Dimension:
        """Return the physical dimensions of the formal unit product."""
        return self._dimension

    def is_dimensionless(self) -> bool:
        """Return whether all physical dimension exponents are zero, regardless of scale."""
        return self.dimension is DIMENSIONLESS
