"""Affine coordinate descriptors with explicit physical mappings."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from functools import partial
from typing import TYPE_CHECKING, Any, Self, cast, overload

from ._coordinates.affine import _AffineMapping
from ._coordinates.definition import _CoordinateDefinition
from ._immutable import _Immutable
from ._numerics import Fractionable, Magnitude
from ._operands import _LinearUnitSource, linear_unit
from ._registry.reconstruction import _definition_by_id
from ._scaling import _coordinate_scalar
from .dimension import DIMENSIONLESS, Dimension
from .exceptions import DimensionError, UnitSymbolError
from .factor import Factor, SymbolicConstant, one
from .unit import Unit
from .unitregistry import _definition_target

if TYPE_CHECKING:
    from .affinequantity import AffineQuantity
    from .quantity import Quantity

__all__ = ['AffineUnit']


class AffineUnit(_Immutable, _LinearUnitSource):
    """A linear coordinate whose reading `at` represents the physical `reference`.

    The difference scale defaults to the reference's unit. Construction retains
    a normalized physical origin and the complete difference Unit. A nonzero
    defining reading that would require a symbolic sum for the origin is rejected.
    Multiplication constructs a difference; `value @ coordinate` constructs an
    AffineQuantity reading. Naming and registry membership do not change mapping.
    """

    __slots__ = ('_mapping', '_definition')
    _mapping: _AffineMapping
    _definition: _CoordinateDefinition | None
    _quantity_name = 'AffineQuantity'

    def __new__(
        cls,
        reference: Quantity[Magnitude] | AffineQuantity[Magnitude],
        /,
        *,
        at: int | float | Fraction | Decimal = 0,
        delta_unit: str | Unit | None = None,
    ) -> Self:
        from .affinequantity import AffineQuantity
        from .quantity import Quantity

        if isinstance(reference, AffineQuantity):
            if delta_unit is None:
                delta_unit = reference.unit.delta_unit
            reference = reference.unit._mapping._exact_physical(
                _coordinate_scalar(reference.value, 'Reference')
            )
        elif not isinstance(reference, Quantity):
            error_message = 'Reference must be a Quantity or AffineQuantity.'
            raise TypeError(error_message)
        delta = reference.unit if delta_unit is None else delta_unit
        if isinstance(delta, str):
            delta = Unit(delta)
        obj = object.__new__(cls)
        object.__setattr__(obj, '_mapping', _AffineMapping(reference, at=at, delta_unit=delta))
        object.__setattr__(obj, '_definition', None)
        return obj

    @classmethod
    def _from_definition(cls, definition: _CoordinateDefinition) -> Self:
        if not isinstance(definition._mapping, _AffineMapping):
            error_message = 'Coordinate definition has an incompatible mapping.'
            raise TypeError(error_message)
        obj = object.__new__(cls)
        object.__setattr__(obj, '_mapping', definition._mapping)
        object.__setattr__(obj, '_definition', definition)
        return obj

    @property
    def origin(self) -> Quantity[Fraction]:
        """The physical quantity at reading zero."""
        return self._mapping.origin

    @property
    def delta_unit(self) -> Unit:
        """The complete linear Unit for differences, including any exact scale."""
        return self._mapping.delta_unit

    @property
    def dimension(self) -> Dimension:
        return self._mapping.dimension

    def _kind(self) -> str | None:
        return None

    def _as_linear_unit(self) -> Unit:
        return (
            Unit._from_definition(self._definition)
            if self._definition is not None
            else self.delta_unit
        )

    def __rmatmul__[T: Magnitude](self, value: T, /) -> AffineQuantity[T]:
        from .affinequantity import AffineQuantity

        return AffineQuantity(value, self)

    def define(
        self,
        symbol: str,
        /,
        name: str | None = None,
        *name_aliases: str,
        symbol_aliases: str | tuple[str, ...] = (),
    ) -> Self:
        """Create a distinct named coordinate; explicit scopes bind its spellings."""
        definition = _CoordinateDefinition(
            self._mapping, symbol, name, *name_aliases, symbol_aliases=symbol_aliases
        )
        target = _definition_target()
        if target is not None:
            target._bind((definition,))
        return type(self)._from_definition(definition)

    @classmethod
    def _from_identity(cls, identity: int) -> Self:
        definition = _definition_by_id(identity)
        if not isinstance(definition, _CoordinateDefinition):
            error_message = 'Definition is not a coordinate.'
            raise TypeError(error_message)
        return cls._from_definition(definition)

    def _compatible_with(self, other: AffineUnit) -> bool:
        return (
            type(self) is type(other)
            and self.dimension == other.dimension
            and self._kind() == other._kind()
        )

    @property
    def symbol(self) -> str:
        if self._definition is not None:
            return self._definition.symbol
        from ._display import _display_unit

        display = _display_unit(self)
        if display is not self:
            return str(display)
        return f'{type(self).__name__}({self.origin})'

    @property
    def name(self) -> str:
        return self._definition.name if self._definition is not None else self.symbol

    def __repr__(self) -> str:
        if self._definition is not None:
            expression = self._default_expression()
            if expression is not None:
                return f'{self._quantity_name}(0, {expression!r}).unit'
            return f'{type(self).__name__}._from_identity({self._definition._key.int!r})'
        kind = f', kind={self._kind()!r}' if self._kind() is not None else ''
        return f'{type(self).__name__}({self.origin!r}, delta_unit={self.delta_unit!r}{kind})'

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        from ._coordinates.resolve import coordinate_from_definition

        if self._definition is not None:
            return coordinate_from_definition, (self._definition,)
        return partial(type(self), delta_unit=self.delta_unit), (self.origin,)

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        return self

    def __str__(self) -> str:
        return self.symbol

    def __format__(self, format_spec: str) -> str:
        from ._display import format_unit

        return format_unit(self, format_spec)

    def to_latex(self, *, fraction: bool = True) -> str:
        """Render the coordinate symbol as escaped LaTeX text."""
        from ._display import unit_body

        return unit_body(self, 'latex', fraction=fraction)

    def to_html(self) -> str:
        """Render the coordinate symbol as escaped HTML."""
        from ._display import html_span, unit_body

        return html_span(unit_body(self, 'html'))

    def to_siunitx(self) -> str:
        """Render the coordinate symbol with siunitx unit parsing disabled."""
        from ._display import siunitx_unit

        return siunitx_unit(self)

    def _repr_latex_(self) -> str:
        return '$' + self.to_latex() + '$'

    def _repr_html_(self) -> str:
        return self.to_html()

    def is_dimensionless(self) -> bool:
        return self.dimension.physical is DIMENSIONLESS

    def __hash__(self) -> int:
        return hash((type(self), self._mapping, self._definition))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AffineUnit):
            return NotImplemented
        return self._mapping == other._mapping and self._definition is other._definition

    @overload
    def __mul__[T: Magnitude](self, other: Quantity[T], /) -> Quantity[T]: ...

    @overload
    def __mul__(self, other: Unit | _LinearUnitSource, /) -> Unit: ...

    @overload
    def __mul__(self, other: Factor | SymbolicConstant, /) -> Unit: ...

    @overload
    def __mul__[T: Magnitude](self, other: T, /) -> Quantity[T]: ...

    def __mul__(self, other: object, /) -> Unit | Quantity[Magnitude]:

        linear = linear_unit(self)
        return linear * cast(
            Any, linear_unit(other) if isinstance(other, _LinearUnitSource) else other
        )

    @overload
    def __rmul__[T: Magnitude](self, other: Quantity[T], /) -> Quantity[T]: ...

    @overload
    def __rmul__(self, other: Unit | _LinearUnitSource, /) -> Unit: ...

    @overload
    def __rmul__(self, other: Factor | SymbolicConstant, /) -> Unit: ...

    @overload
    def __rmul__[T: Magnitude](self, other: T, /) -> Quantity[T]: ...

    def __rmul__(self, other: object, /) -> Unit | Quantity[Magnitude]:

        linear = linear_unit(self)
        return (
            cast(Any, linear_unit(other) if isinstance(other, _LinearUnitSource) else other)
            * linear
        )

    @overload
    def __truediv__(self, other: Unit | _LinearUnitSource, /) -> Unit: ...

    @overload
    def __truediv__(self, other: int | Fraction | Decimal, /) -> Quantity[Fraction]: ...

    @overload
    def __truediv__(self, other: float, /) -> Quantity[float]: ...

    @overload
    def __truediv__(self, other: Factor | SymbolicConstant, /) -> Unit: ...

    @overload
    def __truediv__(self, other: object, /) -> Unit | Quantity[Magnitude]: ...

    def __truediv__(self, other: object, /) -> Unit | Quantity[Magnitude]:

        return linear_unit(self) / cast(
            Any, linear_unit(other) if isinstance(other, _LinearUnitSource) else other
        )

    @overload
    def __rtruediv__(self, other: Unit | _LinearUnitSource, /) -> Unit: ...

    @overload
    def __rtruediv__(self, other: Factor | SymbolicConstant, /) -> Unit: ...

    @overload
    def __rtruediv__[T: Magnitude](self, other: T, /) -> Quantity[T]: ...

    def __rtruediv__(self, other: object, /) -> Unit | Quantity[Magnitude]:

        return cast(
            Any, linear_unit(other) if isinstance(other, _LinearUnitSource) else other
        ) / linear_unit(self)

    def __pow__(self, exponent: Fractionable, /) -> Unit:

        return linear_unit(self) ** exponent

    def __array_ufunc__(
        self, ufunc: object, method: str, *inputs: object, **kwargs: object
    ) -> object:
        from ._numpy import dispatch_ufunc

        return dispatch_ufunc(ufunc, method, inputs, kwargs)

    def expression(self) -> str:
        """Return the registered coordinate spelling, without inventing a unit product."""
        if self._definition is not None:
            return self.symbol
        if (
            self._kind() is None
            and self.origin.value == 0
            and self.origin.unit == self.delta_unit
            and self.delta_unit.scale == one
        ):
            return self.delta_unit.expression()
        error_message = 'Register an anonymous coordinate before requesting its expression.'
        raise TypeError(error_message)

    def _default_expression(self) -> str | None:
        """Return a repr shorthand only when the default binding matches this coordinate."""
        from ._coordinates.resolve import as_coordinate

        try:
            expression = self.expression()
            resolved = as_coordinate(expression)
        except (TypeError, UnitSymbolError, DimensionError):
            return None
        return expression if resolved == self else None


def _as_affine_unit(unit: str | Unit | _LinearUnitSource) -> AffineUnit:
    from ._coordinates.resolve import as_coordinate

    coordinate = as_coordinate(unit)
    if not isinstance(coordinate, AffineUnit):
        error_message = 'Use LogQuantity for logarithmic coordinates.'
        raise TypeError(error_message)
    return coordinate
