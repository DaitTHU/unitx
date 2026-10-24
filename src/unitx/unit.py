from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, Self, cast, overload

from ._immutable import _Immutable
from ._numerics import Array, Fractionable, Magnitude, as_exponent, is_numpy_object
from ._numpy_typing import NumpyValue
from ._registry.prefixes import NO_PREFIX
from .dimension import DIMENSIONLESS, Dimension
from .exceptions import UnitSymbolError, UnitSyntaxError
from .factor import Factor, SymbolicConstant, one
from .monomial import Monomial
from .singleunit import SingleUnit
from .unitdefinition import PrefixPolicy, UnitDefinition, _ConversionData, _UnitProduct
from .unitparser import UnitParser
from .unitregistry import _definition_target

if TYPE_CHECKING:
    from .affinequantity import AffineQuantity
    from .logunit import LogKind, LogUnit
    from .quantity import Quantity


class Unit(_Immutable):
    """A physical unit represented by a formal product of single units.

    Unit always represents a linear scale, including temperature differences
    such as degC and K. Absolute readings use AffineQuantity.

    Accept a string in `UnitParser` syntax, such as `'kg m/s2'`,
    `'m/(s A)'`, or `'m^(1/2)'`. Registered symbols, singular English names,
    prefixes, and aliases are resolved by `SingleUnit` and are case-sensitive.
    Registered multiword names are accepted when they occupy the entire
    input; use `[registered name]` or their symbols inside compound
    expressions, or combine Unit objects instead.
    The default empty string, whitespace-only strings, and `'1'` represent
    the multiplicative identity. See `UnitParser` for the full grammar.

    Multiplication and division combine Unit operands, and powers accept int,
    Fraction, Decimal, or float exponents. Operations return new units and
    cancel matching formal factors. Decimal strings, Fraction, and Decimal
    exponents remain exact; float exponents use their shortest decimal text,
    consistently with Monomial, Dimension, and Factor. Use Fraction for an
    exact rational such as one third.

    Multiplying a Unit by an int, Fraction, Decimal, or float constructs a
    Quantity, just as multiplying the magnitude by the Unit does. Construction
    preserves the magnitude and its type. Dividing a magnitude by a Unit also
    preserves it and uses the reciprocal unit. Dividing a Unit by an int,
    Fraction, or finite Decimal constructs a Quantity with an exact Fraction
    reciprocal; float divisors use native floating-point division. Booleans
    are not accepted as right-hand numeric operands.
    Factor and SymbolicConstant operands add an exact scale to the unit.
    Optional NumPy support constructs
    one Quantity for the entire numeric array, preserving the input object.

    Equality and hashing compare canonical unit identities and rational
    exponents, regardless of factor insertion order. Aliases compare equal,
    while prefixed units such as m and cm remain distinct. Equal dimensions
    and scaling factors alone do not imply equal formal unit expressions.
    The exact scale is part of the unit identity.

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
    >>> print(Unit('cm') * 2)
    2 cm
    >>> print(Unit('m') / 2)
    1/2 m
    """

    __slots__ = ('_monomial', '_scale', '_conversion')
    _monomial: Monomial[SingleUnit]
    _conversion: _ConversionData
    _scale: Factor

    def __new__(
        cls,
        base: str | Unit = '',
        /,
        *,
        scale: SymbolicConstant | Factor = one,
    ) -> Self:
        if isinstance(base, str):
            monomial = UnitParser(base).parse()
            base_scale = one
        elif isinstance(base, Unit):
            monomial = base._monomial
            base_scale = base.scale
        else:
            error_message = f'Unit base must be str or Unit, got {type(base).__name__}.'
            raise TypeError(error_message)
        if not isinstance(scale, (SymbolicConstant, Factor)):
            error_message = (
                f'Unit scale must be Factor or SymbolicConstant, got {type(scale).__name__}.'
            )
            raise TypeError(error_message)
        exact_scale = scale if isinstance(scale, Factor) else Factor(scale)
        return cls._from_monomial(monomial, scale=base_scale * exact_scale)

    @classmethod
    def _from_monomial(cls, monomial: Monomial[SingleUnit], /, *, scale: Factor = one) -> Self:
        """Direct constructor from Monomial without copy. Internal use only."""
        obj = object.__new__(cls)
        factor = one
        dimension = DIMENSIONLESS
        for unit, exponent in monomial.components():
            factor *= unit.factor**exponent
            dimension *= unit.dimension**exponent
        object.__setattr__(obj, '_monomial', monomial)
        object.__setattr__(obj, '_scale', scale)
        object.__setattr__(obj, '_conversion', _ConversionData(factor * scale, dimension))
        return obj

    @classmethod
    def _from_definition(cls, definition: UnitDefinition) -> Unit:
        return Unit._from_monomial(
            Monomial({SingleUnit._from_definitions(NO_PREFIX, definition): 1})
        )

    def _as_product(self) -> _UnitProduct:
        return _UnitProduct(
            self.scale,
            tuple(
                (base._prefix, base._unit, exponent)
                for base, exponent in self._monomial.components()
            ),
        )

    @classmethod
    def _from_product(cls, product: _UnitProduct) -> Unit:
        return Unit._from_monomial(
            Monomial(
                {
                    SingleUnit._from_definitions(prefix, definition): exponent
                    for prefix, definition, exponent in product.terms
                }
            ),
            scale=product.scale,
        )

    def __repr__(self) -> str:
        try:
            resolved = Unit(self.expression())
        except (UnitSymbolError, UnitSyntaxError):
            resolved = None
        if resolved is not None and resolved._monomial == self._monomial:
            if self.scale == one:
                return f'Unit({self.expression()!r})'
            return f'Unit({self.expression()!r}, scale={self.scale!r})'
        terms = tuple(
            (
                base._unit._key.int,
                base.prefix_symbol,
                exponent.numerator,
                exponent.denominator,
            )
            for base, exponent in self._monomial.components()
        )
        representation = f'Unit._from_terms({terms!r}'
        if self.scale != one:
            representation += f', scale={self.scale!r}'
        return representation + ')'

    @classmethod
    def _from_terms(
        cls, terms: tuple[tuple[int, str, int, int], ...], *, scale: Factor = one
    ) -> Unit:
        return Unit._from_monomial(
            Monomial(
                {
                    SingleUnit._from_identity(identity, prefix): Fraction(numerator, denominator)
                    for identity, prefix, numerator, denominator in terms
                }
            ),
            scale=scale,
        )

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        scale = tuple(self.scale._monomial.components())
        return _restore_unit, (self._monomial, scale)

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        return self

    def __call__(
        self,
        reference: Unit | Quantity[Magnitude],
        /,
        *,
        kind: LogKind | None = None,
    ) -> LogUnit:
        """Construct a referenced dB coordinate; other units raise TypeError.

        A Unit reference means one of that unit. Watts and volts (including
        SI prefixes and exact scales) default to power and amplitude conventions;
        other references require an explicit `kind`. Common references reuse
        the named dBm, dBW and dBV descriptors without registering new units.
        """
        from .logunit import _decibel_coordinate

        return _decibel_coordinate(self, reference, kind=kind)

    def __str__(self) -> str:
        if self.scale == one:
            return self.symbol
        components: dict[SingleUnit | int | str, Fraction] = dict(
            self.scale._display_monomial().components()
        )
        components.update(self._monomial.components())
        return str(Monomial(components))

    def __format__(self, format_spec: str) -> str:
        """Select a display format or apply Python string alignment to the unit."""
        from ._display import format_unit

        return format_unit(self, format_spec)

    def to_latex(self, *, fraction: bool = True) -> str:
        """Return a LaTeX fragment with slash fractions or signed unit powers."""
        from ._display import unit_body

        return unit_body(self, 'latex', fraction=fraction)

    def to_siunitx(self) -> str:
        """Return a siunitx v3 unit command, without math delimiters.

        Simple ASCII symbols use native literal unit syntax. Extra scale or
        other symbols use escaped LaTeX with unit parsing explicitly disabled.
        """
        from ._display import siunitx_unit

        return siunitx_unit(self)

    def to_html(self) -> str:
        """Return an escaped HTML fragment, independent of MathJax."""
        from ._display import html_span, unit_body

        return html_span(unit_body(self, 'html'))

    def _repr_latex_(self) -> str:
        return '$' + self.to_latex() + '$'

    def _repr_html_(self) -> str:
        return self.to_html()

    def __hash__(self) -> int:
        return hash((self._monomial, self.scale))

    def __eq__(self, other: object) -> bool:

        if not isinstance(other, Unit):
            return NotImplemented
        return self._monomial == other._monomial and self.scale == other.scale

    @overload
    def __mul__[A: Array](self, other: A) -> Quantity[A]: ...

    @overload
    def __mul__[T: Fractionable](self, other: T) -> Quantity[T]: ...

    @overload
    def __mul__(self, other: SymbolicConstant | Factor) -> Unit: ...

    @overload
    def __mul__(self, other: Unit) -> Unit: ...

    def __mul__(self, other: object) -> Unit | Quantity[Magnitude]:

        from .quantity import Quantity

        if is_numpy_object(other):
            return Quantity(cast(Array, other), self)
        if isinstance(other, Fractionable):
            if isinstance(other, bool):
                raise TypeError('Boolean numeric operands are not supported.')
            return Quantity(other, self)
        if isinstance(other, SymbolicConstant | Factor):
            scale = other if isinstance(other, Factor) else Factor(other)
            return Unit(self, scale=scale)
        if not isinstance(other, Unit):
            return NotImplemented
        return Unit._from_monomial(self._monomial * other._monomial, scale=self.scale * other.scale)

    @overload
    def __truediv__(self, other: Array) -> Quantity[NumpyValue]: ...

    @overload
    def __truediv__(self, other: int | Fraction | Decimal) -> Quantity[Fraction]: ...

    @overload
    def __truediv__(self, other: float) -> Quantity[float]: ...

    @overload
    def __truediv__(self, other: SymbolicConstant | Factor) -> Unit: ...

    @overload
    def __truediv__(self, other: Unit) -> Unit: ...

    def __truediv__(self, other: object) -> Unit | Quantity[Magnitude]:

        from .quantity import Quantity

        if is_numpy_object(other):
            from ._numpy import apply_ufunc, validate_magnitude

            validate_magnitude(other)
            return apply_ufunc('true_divide', self, other)
        if isinstance(other, Fractionable):
            if isinstance(other, bool):
                raise TypeError('Boolean numeric operands are not supported.')
            reciprocal = 1 / other if isinstance(other, float) else 1 / Fraction(other)
            return Quantity(reciprocal, self)
        if isinstance(other, SymbolicConstant | Factor):
            scale = 1 / other if isinstance(other, Factor) else 1 / Factor(other)
            return Unit(self, scale=scale)
        if not isinstance(other, Unit):
            return NotImplemented
        return Unit._from_monomial(self._monomial / other._monomial, scale=self.scale / other.scale)

    @overload
    def __rmul__[T: Magnitude](self, other: T) -> Quantity[T]: ...

    @overload
    def __rmul__(self, other: SymbolicConstant | Factor) -> Unit: ...

    def __rmul__(self, other: Magnitude | SymbolicConstant | Factor) -> Quantity[Magnitude] | Unit:
        from .quantity import Quantity

        if isinstance(other, (Factor, SymbolicConstant)):
            return Unit(self, scale=other)
        return Quantity(other, self)

    @overload
    def __rtruediv__[T: Magnitude](self, other: T) -> Quantity[T]: ...

    @overload
    def __rtruediv__(self, other: SymbolicConstant | Factor) -> Unit: ...

    def __rtruediv__(
        self, other: Magnitude | SymbolicConstant | Factor
    ) -> Quantity[Magnitude] | Unit:
        from .quantity import Quantity

        if isinstance(other, (Factor, SymbolicConstant)):
            return Unit(self**-1, scale=other)
        return Quantity(other, self**-1)

    def __pow__(self, exponent: Fractionable) -> Unit:
        if not isinstance(exponent, Fractionable):
            return NotImplemented
        power = as_exponent(exponent)
        return Unit._from_monomial(self._monomial**power, scale=self.scale**power)

    def __rmatmul__[T: Magnitude](self, value: T, /) -> AffineQuantity[T]:
        """Construct an absolute reading: value at this unit's coordinate."""
        from .affinequantity import AffineQuantity

        return AffineQuantity(value, self)

    def __array_ufunc__(
        self, ufunc: object, method: str, *inputs: object, **kwargs: object
    ) -> object:
        from ._numpy import dispatch_ufunc

        return dispatch_ufunc(ufunc, method, inputs, kwargs)

    @property
    def symbol(self) -> str:
        """Return the readable unit expression with Unicode multiplication and powers."""
        return str(self._monomial)

    @property
    def scale(self) -> Factor:
        """Return the exact scale relative to the formal unit product."""
        return self._scale

    @property
    def factor(self) -> Factor:
        """Return the exact scale of a multiplicative unit.

        Temperature spellings describe differences; no offset is included.
        """
        return self._conversion.factor

    @property
    def dimension(self) -> Dimension:
        """Return the physical dimensions of the formal unit product."""
        return self._conversion.dimension

    def is_dimensionless(self) -> bool:
        """Return whether all physical dimension exponents are zero, regardless of scale."""
        return self.dimension.physical is DIMENSIONLESS

    def expression(self) -> str:
        """Return only the formal unit product in parser syntax."""
        return ' * '.join(
            unit._expression()
            if exponent == 1
            else f'{unit._expression()}**{exponent.numerator}'
            if exponent.denominator == 1
            else f'{unit._expression()}**({exponent})'
            for unit, exponent in self._monomial.components()
        )

    def define(
        self,
        symbol: str,
        /,
        name: str | None = None,
        *name_aliases: str,
        symbol_aliases: str | tuple[str, ...] = (),
        prefixable: PrefixPolicy | None = None,
    ) -> Unit:
        """Create an independent named unit without changing self.

        A Registry.defining() block binds this definition to its selected registry.
        Otherwise no parser tables change. UnitCatalog collects public unit
        attributes after its class body finishes.

        The new unit has this unit's exact SI factor and physical dimension,
        but a distinct formal identity. Symbols and symbol aliases must each
        be one printable UnitParser unit token. Names and name aliases may
        contain multiple such tokens separated by single ASCII spaces or hyphens, without
        leading or trailing spaces. Decimal digits and grammar operators are
        not allowed in any token; square brackets and integer superscripts are
        reserved. The symbol must be nonempty; an omitted name defaults to it. Empty aliases
        are ignored.

        The symbol is positional-only. The primary name accepts positional or
        keyword input. Additional names are optional positional arguments;
        pass the primary name positionally when supplying them.
        Symbol aliases accept a string or tuple of strings and,
        like prefixable, must be passed by keyword.

        Omitted prefixable uses the defining registry default, or True outside a scope.
        With prefixable=True, SI prefixes also apply to symbols and names;
        prefix names attach directly to the first word of a multiword name.
        All spellings must be internally unambiguous. Binding to a Registry
        also checks existing units, prefixed forms and implicit unit products.
        Complex names work standalone or in `[name]` references in expressions.
        `prefixable` accepts True for all spellings, False for none, a single
        spelling string, or a tuple of exact symbols, names, or aliases, such as
        `('arcsec', 'arcsecond')`. Only listed spellings accept prefixes;
        `'arcsec'` is equivalent to `('arcsec',)`. An empty tuple disables
        all prefixes. Unknown or empty spellings are errors.
        Invalid definitions raise TypeError, ValueError, or UnitRegistryError
        before returning the definition or changing an explicit registry.
        """
        target = _definition_target()
        definition = UnitDefinition._derived(
            symbol,
            name,
            *name_aliases,
            conversion=self._conversion,
            symbol_aliases=symbol_aliases,
            prefixable=(target._prefixable if target is not None else True)
            if prefixable is None
            else prefixable,
        )
        if target is not None:
            target._bind((definition,))
        return Unit._from_definition(definition)


def _restore_unit(
    monomial: Monomial[SingleUnit],
    scale: tuple[tuple[int | SymbolicConstant, Fraction], ...],
) -> Unit:
    return Unit._from_monomial(monomial, scale=Factor.from_dict(dict(scale)))
