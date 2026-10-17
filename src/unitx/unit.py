from __future__ import annotations

from typing import TYPE_CHECKING, Self, overload

from ._numerics import Fractionable, Magnitude, as_fraction
from .dimension import DIMENSIONLESS, Dimension
from .factor import ONE, Factor, SymbolicConstant
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
    without scaling. A Factor or SymbolicConstant instead constructs a
    ScaledUnit and preserves its exact scale. These construction operators
    require the magnitude or factor on the left; multiplying or dividing a
    Unit by a raw magnitude or factor is unsupported. NumPy scalars, arrays,
    and NumPy operations are not supported yet; NumPy integration is planned.

    Equality and hashing compare canonical unit identities and rational
    exponents, regardless of factor insertion order. Aliases compare equal,
    while prefixed units such as m and cm remain distinct. Equal dimensions
    and scaling factors alone do not imply equal formal unit expressions.
    ScaledUnit additionally compares its explicit scale; a scale of one
    compares equal to the corresponding Unit.

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

    def __new__(cls, symbol: str = '', /) -> Self:
        if not isinstance(symbol, str):
            raise TypeError(f'Unit symbol must be str, got {type(symbol)}.')
        return cls._from_monomial(UnitParser(symbol).parse())

    @classmethod
    def _from_monomial(cls, monomial: Monomial[SingleUnit], /) -> Self:
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
        return f'{type(self).__name__}({self.expression()!r})'

    def __str__(self) -> str:
        return self.symbol

    def __hash__(self) -> int:
        return hash(self._monomial)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Unit):
            return NotImplemented
        return self._monomial == other._monomial

    @overload
    def __mul__(self, other: ScaledUnit) -> ScaledUnit: ...

    @overload
    def __mul__(self, other: Unit) -> Unit: ...

    def __mul__(self, other: Unit) -> Unit:
        if not isinstance(other, Unit):
            return NotImplemented
        return Unit._from_monomial(self._monomial * other._monomial)

    @overload
    def __truediv__(self, other: ScaledUnit) -> ScaledUnit: ...

    @overload
    def __truediv__(self, other: Unit) -> Unit: ...

    def __truediv__(self, other: Unit) -> Unit:
        if not isinstance(other, Unit):
            return NotImplemented
        return Unit._from_monomial(self._monomial / other._monomial)

    @overload
    def __rmul__[T: Magnitude](self, other: T) -> Quantity[T]: ...

    @overload
    def __rmul__(self, other: SymbolicConstant | Factor) -> ScaledUnit: ...

    def __rmul__(
        self, other: Magnitude | SymbolicConstant | Factor
    ) -> Quantity[Magnitude] | ScaledUnit:
        from .quantity import Quantity

        if isinstance(other, (Factor, SymbolicConstant)):
            return ScaledUnit(self, other)
        return Quantity(other, self)

    @overload
    def __rtruediv__[T: Magnitude](self, other: T) -> Quantity[T]: ...

    @overload
    def __rtruediv__(self, other: SymbolicConstant | Factor) -> ScaledUnit: ...

    def __rtruediv__(
        self, other: Magnitude | SymbolicConstant | Factor
    ) -> Quantity[Magnitude] | ScaledUnit:
        from .quantity import Quantity

        if isinstance(other, (Factor, SymbolicConstant)):
            return ScaledUnit(self**-1, other)
        return Quantity(other, self**-1)

    def __pow__(self, exponent: Fractionable) -> Unit:
        if not isinstance(exponent, Fractionable):
            return NotImplemented
        return Unit._from_monomial(self._monomial**exponent)

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

    def expression(self) -> str:
        """Return only the formal unit product in parser syntax."""
        return ' * '.join(
            f'{unit}'
            if exponent == 1
            else f'{unit}**{exponent.numerator}'
            if exponent.denominator == 1
            else f'{unit}**({exponent})'
            for unit, exponent in self._monomial.components()
        )


class ScaledUnit(Unit):
    """A formal unit product with an additional exact positive scale.

    `scale` is relative to the unscaled formal product; `factor` includes both
    that scale and the product's own SI factor. Either can equal one. Nested
    scales are combined exactly. Unit multiplication, division, and powers
    involving a ScaledUnit return ScaledUnit, even when the scale becomes one.
    Equality compares the formal product and explicit scale, so a scale of one
    compares equal to the unscaled Unit. `symbol` describes only the product;
    str() also includes its scale, and repr() reconstructs both.

    Copying and pickling preserve the formal units and exact scale. Restoring
    a unit requires its referenced unit definitions to be registered.

    >>> minute = Factor(60) * Unit('s')
    >>> minute
    ScaledUnit('s', scale=Factor.from_dict({2: 2, 3: 1, 5: 1}))
    >>> str(minute)
    '60 s'
    >>> minute == Unit('s')
    False
    """

    __slots__ = ('_scale',)
    _scale: Factor

    def __new__(cls, base: Unit | str, /, scale: SymbolicConstant | Factor) -> ScaledUnit:
        if isinstance(base, str):
            base = Unit(base)
        elif not isinstance(base, Unit):
            error_message = f'Base unit must be Unit or str, got {type(base).__name__}.'
            raise TypeError(error_message)
        if not isinstance(scale, (SymbolicConstant, Factor)):
            error_message = f'Scale must be SymbolicConstant or Factor, got {type(scale).__name__}.'
            raise TypeError(error_message)
        scale = scale if isinstance(scale, Factor) else Factor(scale)
        obj = object.__new__(cls)
        obj._monomial = base._monomial
        obj._symbol = base.symbol
        obj._scale = base.scale * scale if isinstance(base, ScaledUnit) else scale
        obj._factor = base.factor * scale
        obj._dimension = base.dimension
        return obj

    @classmethod
    def _from_monomial(cls, monomial: Monomial[SingleUnit], /, scale: Factor) -> ScaledUnit:
        """Direct constructor from Monomial without copy. Internal use only."""
        obj = super()._from_monomial(monomial)
        obj._scale = scale
        obj._factor *= scale
        return obj

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.expression()!r}, scale={self.scale!r})'

    def __reduce__(self) -> tuple[type[ScaledUnit], tuple[str, Factor]]:
        return type(self), (self.expression(), self.scale)

    def __str__(self) -> str:
        if self.scale == ONE:
            return self.symbol
        if self.symbol.startswith('1'):
            return f'{self.scale}{self.symbol[1:]}'
        return f'{self.scale} {self.symbol}'

    def __hash__(self) -> int:
        if self.scale == ONE:
            return super().__hash__()
        return hash((self._monomial, self.scale))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Unit):
            return NotImplemented
        other_scale = other.scale if isinstance(other, ScaledUnit) else ONE
        return self._monomial == other._monomial and self.scale == other_scale

    def __mul__(self, other: Unit) -> ScaledUnit:
        if not isinstance(other, Unit):
            return NotImplemented
        scale = self.scale * other.scale if isinstance(other, ScaledUnit) else self.scale
        return ScaledUnit._from_monomial(self._monomial * other._monomial, scale)

    def __truediv__(self, other: Unit) -> ScaledUnit:
        if not isinstance(other, Unit):
            return NotImplemented
        scale = self.scale / other.scale if isinstance(other, ScaledUnit) else self.scale
        return ScaledUnit._from_monomial(self._monomial / other._monomial, scale)

    @overload
    def __rmul__[T: Magnitude](self, other: T) -> Quantity[T]: ...

    @overload
    def __rmul__(self, other: Unit | SymbolicConstant | Factor) -> ScaledUnit: ...

    def __rmul__(
        self, other: Unit | Magnitude | SymbolicConstant | Factor
    ) -> Quantity[Magnitude] | ScaledUnit:
        if isinstance(other, Unit):
            scale = other.scale * self.scale if isinstance(other, ScaledUnit) else self.scale
            return ScaledUnit._from_monomial(other._monomial * self._monomial, scale)
        return super().__rmul__(other)

    @overload
    def __rtruediv__[T: Magnitude](self, other: T) -> Quantity[T]: ...

    @overload
    def __rtruediv__(self, other: Unit | SymbolicConstant | Factor) -> ScaledUnit: ...

    def __rtruediv__(
        self, other: Unit | Magnitude | SymbolicConstant | Factor
    ) -> Quantity[Magnitude] | ScaledUnit:
        if isinstance(other, Unit):
            other_scale = other.scale if isinstance(other, ScaledUnit) else ONE
            return ScaledUnit._from_monomial(
                other._monomial / self._monomial, other_scale / self.scale
            )
        return super().__rtruediv__(other)

    def __pow__(self, exponent: Fractionable) -> ScaledUnit:
        if not isinstance(exponent, Fractionable):
            return NotImplemented
        power = as_fraction(exponent)
        return ScaledUnit._from_monomial(self._monomial**power, self.scale**power)

    @property
    def scale(self) -> Factor:
        """Return the additional scale relative to the formal unit product."""
        return self._scale
