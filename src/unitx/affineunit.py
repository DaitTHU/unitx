"""Coordinates with an origin, in a linear or logarithmic physical space."""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal, Self, cast, overload

from ._immutable import _Immutable
from ._numerics import Fractionable, Magnitude, is_numpy_object
from .dimension import DIMENSIONLESS, Dimension
from .exceptions import DimensionError
from .factor import Factor, SymbolicConstant
from .unit import ScaledUnit, Unit
from .unitregistry import UnitDefinition, _register_unit

if TYPE_CHECKING:
    from .affinequantity import AffineQuantity
    from .quantity import Quantity

__all__ = ['AffineUnit', 'LogKind']

type LogKind = Literal['power', 'amplitude']


class AffineUnit(_Immutable):
    """Define a coordinate by a known reading and its physical reference.

    ``AffineUnit(reference, at=x, delta_unit=unit)`` means reading ``x`` equals
    ``reference``. The default reading is zero. Linear coordinates default to
    the reference's difference unit; logarithmic coordinates require both an
    explicit difference unit and kind ('power' or 'amplitude').
    AffineQuantity looks up registered coordinates by spelling. Prefixes and compound
    affine expressions are unsupported. ``value @ coordinate`` constructs an
    AffineQuantity; arithmetic on an existing point requires its difference unit.
    """

    __slots__ = ('_origin', '_delta_unit', '_kind', '_definition')
    _origin: Quantity[Magnitude]
    _delta_unit: Unit
    _kind: LogKind | None
    _definition: UnitDefinition | None

    def __new__(
        cls,
        reference: Quantity[Magnitude] | AffineQuantity[Magnitude],
        /,
        *,
        at: int | float | Fraction | Decimal = 0,
        delta_unit: str | Unit | None = None,
        kind: LogKind | None = None,
    ) -> Self:
        from ._conversion import convert, scale_value, to_ratio
        from .affinequantity import AffineQuantity
        from .quantity import Quantity

        if not isinstance(reference, (Quantity, AffineQuantity)):
            raise TypeError('Reference must be a Quantity or AffineQuantity.')
        if kind not in (None, 'power', 'amplitude'):
            raise ValueError("kind must be 'power', 'amplitude', or None.")
        value = _coordinate_scalar(reference.value, 'Reference')
        reading = _coordinate_scalar(at, 'at')
        reference_unit = reference.unit
        if delta_unit is None:
            if kind is not None:
                raise TypeError('A logarithmic coordinate requires an explicit delta_unit.')
            delta_unit = (
                reference_unit.delta_unit
                if isinstance(reference_unit, AffineUnit)
                else reference_unit
            )
        delta = Unit(delta_unit) if isinstance(delta_unit, str) else delta_unit
        if not isinstance(delta, Unit) or isinstance(delta, AffineUnit):
            raise TypeError('delta_unit must be a multiplicative Unit.')
        if isinstance(reference_unit, AffineUnit):
            physical_unit = reference_unit.origin.unit
            value = _coordinate_scalar(convert(value, reference_unit, physical_unit), 'Reference')
        else:
            physical_unit = reference_unit
        if physical_unit.log_degree:
            raise DimensionError(
                'Reference must be a physical quantity, not a logarithmic difference.'
            )
        if kind is None:
            if delta.dimension != reference.dimension or delta.log_degree:
                raise DimensionError(
                    'Linear reference and difference must have the same dimension.'
                )
            # Rationalize before arithmetic, independent of the Decimal context.
            step = _coordinate_scalar(
                scale_value(reading, delta.factor / physical_unit.factor), 'Coordinate step'
            )
            origin_value = value - step
        else:
            if not delta.is_dimensionless() or delta.log_degree != 1:
                raise DimensionError(
                    'A logarithmic coordinate requires a pure logarithmic difference unit.'
                )
            if value <= 0:
                raise ValueError('Logarithmic reference must be positive.')
            origin_value = value / to_ratio(reading, delta, kind)
        origin_value = _coordinate_scalar(origin_value, 'Origin')
        if kind is not None and origin_value <= 0:
            raise ValueError('Logarithmic origin must be positive.')
        obj = object.__new__(cls)
        object.__setattr__(obj, '_origin', Quantity(origin_value, physical_unit))
        object.__setattr__(obj, '_delta_unit', delta)
        object.__setattr__(obj, '_kind', kind)
        object.__setattr__(obj, '_definition', None)
        return obj

    @classmethod
    def _from_definition(cls, definition: UnitDefinition) -> Self:
        from .quantity import Quantity

        assert definition.reference is not None and definition.delta is not None
        obj = cls(
            Quantity(*definition.reference),
            delta_unit=definition.delta,
            kind=cast(LogKind | None, definition.kind),
        )
        object.__setattr__(obj, '_definition', definition)
        return obj

    @property
    def origin(self) -> Quantity[Magnitude]:
        """Physical quantity at reading zero, in a multiplicative unit.

        Reference points are normalized at construction; no coordinate chain
        or copy of the original defining reading is retained.
        """
        return self._origin

    @property
    def delta_unit(self) -> Unit:
        return self._delta_unit

    @property
    def kind(self) -> LogKind | None:
        return self._kind

    @property
    def dimension(self) -> Dimension:
        return self.origin.dimension

    @property
    def log_degree(self) -> Fraction:
        return self.delta_unit.log_degree

    @property
    def symbol(self) -> str:
        if self._definition is not None:
            return self._definition.symbol
        if self.kind is None and self.origin.value == 0 and self.origin.unit == self.delta_unit:
            return self.delta_unit.symbol
        return f'affine({self.origin})'

    @property
    def name(self) -> str:
        return self._definition.name if self._definition is not None else self.symbol

    def __repr__(self) -> str:
        if self._definition is not None:
            return f'AffineQuantity(0, {self.symbol!r}).unit'
        return f'AffineUnit({self.origin!r}, delta_unit={self.delta_unit!r}, kind={self.kind!r})'

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        if self._definition is not None:
            return _restore_coordinate, (self.symbol,)
        return _restore_affine, (self.origin, self.delta_unit, self.kind)

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
        return self.dimension is DIMENSIONLESS

    def __hash__(self) -> int:
        return hash((self.delta_unit, self.origin, self.kind, self.symbol))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AffineUnit):
            return NotImplemented
        return (
            self.delta_unit == other.delta_unit
            and self.kind == other.kind
            and bool(self.origin == other.origin)
            and self.symbol == other.symbol
        )

    def __rmatmul__[T: Magnitude](self, value: T, /) -> AffineQuantity[T]:
        """Construct an absolute reading in this custom coordinate."""
        from .affinequantity import AffineQuantity

        return AffineQuantity(value, self)

    @overload
    def __mul__[T: Magnitude](self, other: Quantity[T], /) -> Quantity[T]: ...

    @overload
    def __mul__(self, other: Unit | AffineUnit, /) -> Unit: ...

    @overload
    def __mul__(self, other: Factor | SymbolicConstant, /) -> ScaledUnit: ...

    @overload
    def __mul__[T: Magnitude](self, other: T, /) -> Quantity[T]: ...

    def __mul__(self, other: object, /) -> Unit | Quantity[Magnitude]:
        from ._conversion import linear_unit

        linear = linear_unit(self)
        return linear * cast(Any, linear_unit(other) if isinstance(other, AffineUnit) else other)

    @overload
    def __rmul__[T: Magnitude](self, other: Quantity[T], /) -> Quantity[T]: ...

    @overload
    def __rmul__(self, other: Unit | AffineUnit, /) -> Unit: ...

    @overload
    def __rmul__(self, other: Factor | SymbolicConstant, /) -> ScaledUnit: ...

    @overload
    def __rmul__[T: Magnitude](self, other: T, /) -> Quantity[T]: ...

    def __rmul__(self, other: object, /) -> Unit | Quantity[Magnitude]:
        from ._conversion import linear_unit

        linear = linear_unit(self)
        return cast(Any, linear_unit(other) if isinstance(other, AffineUnit) else other) * linear

    @overload
    def __truediv__(self, other: Unit | AffineUnit, /) -> Unit: ...

    @overload
    def __truediv__(self, other: int | Fraction | Decimal, /) -> Quantity[Fraction]: ...

    @overload
    def __truediv__(self, other: float, /) -> Quantity[float]: ...

    @overload
    def __truediv__(self, other: Factor | SymbolicConstant, /) -> ScaledUnit: ...

    @overload
    def __truediv__(self, other: object, /) -> Unit | Quantity[Magnitude]: ...

    def __truediv__(self, other: object, /) -> Unit | Quantity[Magnitude]:
        from ._conversion import linear_unit

        return linear_unit(self) / cast(
            Any, linear_unit(other) if isinstance(other, AffineUnit) else other
        )

    @overload
    def __rtruediv__(self, other: Unit | AffineUnit, /) -> Unit: ...

    @overload
    def __rtruediv__(self, other: Factor | SymbolicConstant, /) -> ScaledUnit: ...

    @overload
    def __rtruediv__[T: Magnitude](self, other: T, /) -> Quantity[T]: ...

    def __rtruediv__(self, other: object, /) -> Unit | Quantity[Magnitude]:
        from ._conversion import linear_unit

        return cast(
            Any, linear_unit(other) if isinstance(other, AffineUnit) else other
        ) / linear_unit(self)

    def __pow__(self, exponent: Fractionable, /) -> Unit:
        from ._conversion import linear_unit

        return linear_unit(self) ** exponent

    def __array_ufunc__(
        self, ufunc: object, method: str, *inputs: object, **kwargs: object
    ) -> object:
        from ._numpy import dispatch_ufunc

        return dispatch_ufunc(ufunc, method, inputs, kwargs)

    def define(
        self,
        symbol: str,
        /,
        name: str,
        *name_aliases: str,
        symbol_aliases: str | tuple[str, ...] = (),
        prefixable: Literal[False] = False,
    ) -> AffineUnit:
        """Register this coordinate under a distinct immutable identity, without prefixes."""
        if prefixable is not False:
            raise TypeError('Affine coordinate prefixes are unsupported.')
        reference_unit = self.origin.unit
        assert isinstance(reference_unit, Unit)
        definition = UnitDefinition(
            symbol,
            name,
            *name_aliases,
            delta=self.delta_unit.expression(),
            reference=(
                cast(int | float | Fraction | Decimal, self.origin.value),
                reference_unit.expression(),
            ),
            factor=self.delta_unit.factor,
            dimension=self.delta_unit.dimension,
            log_degree=self.delta_unit.log_degree,
            prefixable=False,
            kind=self.kind,
            symbol_aliases=symbol_aliases,
        )
        # Explicit scales cannot be discarded when storing parser expressions.
        from .unit import ScaledUnit

        if isinstance(self.delta_unit, ScaledUnit) or isinstance(reference_unit, ScaledUnit):
            raise TypeError(
                'Register scaled reference/difference units before defining a coordinate.'
            )
        _register_unit(definition)
        return type(self)._from_definition(definition)

    def expression(self) -> str:
        """Return the registered coordinate spelling, without inventing a unit product."""
        if self._definition is not None:
            return self.symbol
        from .unit import ScaledUnit

        if (
            self.kind is None
            and self.origin.value == 0
            and self.origin.unit == self.delta_unit
            and not isinstance(self.delta_unit, ScaledUnit)
        ):
            return self.delta_unit.expression()
        raise TypeError('Register an anonymous coordinate before requesting its expression.')


def _restore_affine(
    reference: Quantity[Magnitude] | AffineQuantity[Magnitude], delta: Unit, kind: LogKind | None
) -> AffineUnit:
    return AffineUnit(reference, delta_unit=delta, kind=kind)


def _coordinate_scalar(value: object, name: str) -> Fraction:
    """Validate a finite real defining value and preserve its exact stored value."""
    if (
        is_numpy_object(value)
        or isinstance(value, bool)
        or not isinstance(value, (int, float, Fraction, Decimal))
    ):
        raise TypeError(f'{name} must have a real Python scalar magnitude.')
    finite = (
        value.is_finite()
        if isinstance(value, Decimal)
        else (math.isfinite(value) if isinstance(value, float) else True)
    )
    if not finite:
        raise ValueError(f'{name} must be finite.')
    return Fraction(value)


def _as_affine_unit(unit: str | Unit | AffineUnit) -> AffineUnit:
    """Resolve a point coordinate; ordinary linear units have a zero origin."""
    from .exceptions import UnitSymbolError
    from .quantity import Quantity
    from .unit import ScaledUnit
    from .unitregistry import _resolve_unit

    if isinstance(unit, AffineUnit):
        return unit
    if not isinstance(unit, (str, Unit)):
        raise TypeError('A point coordinate must be an AffineUnit, linear Unit, or spelling.')
    if isinstance(unit, ScaledUnit):
        return AffineUnit(Quantity(0, unit))
    spelling = unit if isinstance(unit, str) else unit.expression()
    try:
        _, definition = _resolve_unit(' '.join(spelling.split()))
    except UnitSymbolError:
        pass
    else:
        if definition.reference is not None:
            return AffineUnit._from_definition(definition)
    linear = Unit(unit) if isinstance(unit, str) else unit
    if linear.log_degree:
        raise DimensionError('A logarithmic difference needs an explicit physical reference.')
    return AffineUnit(Quantity(0, linear))


def _restore_coordinate(symbol: str) -> AffineUnit:
    return _as_affine_unit(symbol)
