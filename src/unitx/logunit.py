"""Log coordinate descriptors with explicit physical mappings."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from functools import partial
from typing import TYPE_CHECKING, Any, Self, cast, overload

from ._coordinates.definition import _CoordinateDefinition
from ._coordinates.logarithmic import _LogMapping
from ._coordinates.logarithms import LogKind
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
    from .logquantity import LogQuantity
    from .quantity import Quantity

__all__ = ['LogUnit', 'LogKind']


class LogUnit(_Immutable, _LinearUnitSource):
    """A logarithmic coordinate whose reading `at` represents `reference`.

    The positive physical reference, logarithmic difference Unit and explicit
    power/amplitude convention determine the mapping. The defining `reference` and `at`
    are retained without evaluating the zero reading. Multiplication constructs
    a difference; `value @ coordinate` constructs a LogQuantity reading.
    """

    __slots__ = ('_mapping', '_definition')
    _mapping: _LogMapping
    _definition: _CoordinateDefinition | None
    _quantity_name = 'LogQuantity'

    def __new__(
        cls,
        reference: Quantity[Magnitude] | LogQuantity[Magnitude],
        /,
        *,
        at: int | float | Fraction | Decimal = 0,
        delta_unit: str | Unit,
        kind: LogKind,
    ) -> Self:
        from .logquantity import LogQuantity
        from .quantity import Quantity

        if isinstance(reference, LogQuantity):
            value = _coordinate_scalar(reference.value, 'Reference')
            physical = reference.unit.reference.unit
            reference = Quantity(reference.unit._mapping._physical_value(value, physical), physical)
        elif not isinstance(reference, Quantity):
            error_message = 'Logarithmic reference must be a Quantity or LogQuantity.'
            raise TypeError(error_message)
        delta = Unit(delta_unit) if isinstance(delta_unit, str) else delta_unit
        obj = object.__new__(cls)
        object.__setattr__(
            obj, '_mapping', _LogMapping(reference, at=at, delta_unit=delta, kind=kind)
        )
        object.__setattr__(obj, '_definition', None)
        return obj

    @classmethod
    def _from_definition(cls, definition: _CoordinateDefinition) -> Self:
        if not isinstance(definition._mapping, _LogMapping):
            error_message = 'Coordinate definition has an incompatible mapping.'
            raise TypeError(error_message)
        obj = object.__new__(cls)
        object.__setattr__(obj, '_mapping', definition._mapping)
        object.__setattr__(obj, '_definition', definition)
        return obj

    @property
    def origin(self) -> Quantity[Fraction]:
        """The positive physical reference at reading zero."""
        return self._mapping.origin

    @property
    def reference(self) -> Quantity[Fraction]:
        """The physical quantity at the defining reading `at`."""
        return self._mapping.reference

    @property
    def at(self) -> Fraction:
        """The exact defining reading paired with `reference`."""
        return self._mapping.at

    @property
    def delta_unit(self) -> Unit:
        """The complete logarithmic difference Unit."""
        return self._mapping.delta_unit

    @property
    def dimension(self) -> Dimension:
        return self._mapping.dimension

    def _kind(self) -> str | None:
        return self.kind

    def _as_linear_unit(self) -> Unit:
        return (
            Unit._from_definition(self._definition)
            if self._definition is not None
            else self.delta_unit
        )

    def __rmatmul__[T: Magnitude](self, value: T, /) -> LogQuantity[T]:
        from .logquantity import LogQuantity

        return LogQuantity(value, self)

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

    @property
    def kind(self) -> LogKind:
        return self._mapping.kind

    @classmethod
    def _from_identity(cls, identity: int) -> Self:
        definition = _definition_by_id(identity)
        if not isinstance(definition, _CoordinateDefinition):
            error_message = 'Definition is not a coordinate.'
            raise TypeError(error_message)
        return cls._from_definition(definition)

    def _compatible_with(self, other: LogUnit) -> bool:
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
        return f'{type(self).__name__}({self.reference}, at={self.at})'

    @property
    def name(self) -> str:
        return self._definition.name if self._definition is not None else self.symbol

    def __repr__(self) -> str:
        if self._definition is not None:
            expression = self._default_expression()
            if expression is not None:
                return f'{self._quantity_name}(0, {expression!r}).unit'
            return f'{type(self).__name__}._from_identity({self._definition._key.int!r})'
        return (
            f'{type(self).__name__}({self.reference!r}, delta_unit={self.delta_unit!r}, '
            f'kind={self.kind!r}, at={self.at!r})'
        )

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        from ._coordinates.resolve import coordinate_from_definition

        if self._definition is not None:
            return coordinate_from_definition, (self._definition,)
        return (
            partial(type(self), delta_unit=self.delta_unit, kind=self.kind, at=self.at),
            (self.reference,),
        )

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
        if not isinstance(other, LogUnit):
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


def _as_log_unit(unit: str | Unit | _LinearUnitSource) -> LogUnit:
    from ._coordinates.resolve import as_coordinate

    coordinate = as_coordinate(unit)
    if not isinstance(coordinate, LogUnit):
        error_message = 'A logarithmic reading requires a LogUnit with a physical reference.'
        raise TypeError(error_message)
    return coordinate
