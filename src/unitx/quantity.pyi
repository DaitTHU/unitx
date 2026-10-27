from __future__ import annotations

# The public signatures reuse the package's private operand protocols.
# pyright: reportPrivateUsage=false
from collections.abc import Callable, Iterator
from decimal import Decimal
from fractions import Fraction
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Never,
    SupportsIndex,
    TypeAlias,
    TypeGuard,
    TypeVar,
    overload,
)

from ._numerics import (
    Array,
    Fractionable,
    Magnitude,
    Number,
    Real,
    Scalar,
)
from ._numerics.numpy_typing import (
    HasImag,
    HasReal,
    NumpyArray,
    NumpyDType,
    NumpyValue,
    WritableArray,
    _ArrayLeadingIndex,
    _Axis,
    _Complex64,
    _Complex128,
    _ComplexScalar,
    _DTypeValue,
    _Float32,
    _Float64,
    _FloatingScalar,
    _InexactScalar,
    _IntegerScalar,
    _NumpyArrayOf,
    _NumpyNumber,
    _NumpyScalar,
    _NumpyValueOf,
    _Shape,
    _TypedArray,
    _TypedScalar,
)
from ._operands import _LinearUnitSource, _Point
from .dimension import Dimension
from .factor import Factor, SymbolicConstant
from .unit import Unit
from .unitdefinition import PrefixPolicy

T = TypeVar('T', bound=Magnitude, covariant=True)

_NonnegativeIntegerPower: TypeAlias = Literal[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
]

_NegativeIntegerPower: TypeAlias = Literal[
    -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1
]

if TYPE_CHECKING:
    from ._coordinates.logarithms import LogKind

def _is_quantity(value: object) -> TypeGuard[Quantity[Magnitude]]: ...
def _is_nonfinite_real(value: object) -> bool: ...
def _is_float_decimal(left: object, right: object) -> bool: ...
def _scalar_arithmetic(
    op: Callable[[Scalar, Scalar], object], left: Scalar, right: Scalar
) -> Scalar: ...

class _NumberValue:
    # A descriptor in the stub allows overloads based on the instance type;
    # at runtime Quantity.n is an ordinary read-only property.
    @overload
    def __get__(self, quantity: None, owner: type[object] | None = None) -> property: ...
    @overload
    def __get__(
        self, quantity: Quantity[_TypedArray[_InexactScalar]], owner: type[object] | None = None
    ) -> _NumpyValueOf[_InexactScalar]: ...
    @overload
    def __get__(
        self, quantity: Quantity[_TypedScalar[_InexactScalar]], owner: type[object] | None = None
    ) -> _InexactScalar: ...
    @overload
    def __get__(
        self, quantity: Quantity[_DTypeValue[_InexactScalar]], owner: type[object] | None = None
    ) -> _NumpyValueOf[_InexactScalar]: ...
    @overload
    def __get__(
        self, quantity: Quantity[Array], owner: type[object] | None = None
    ) -> NumpyValue: ...
    @overload
    def __get__(
        self, quantity: Quantity[Decimal], owner: type[object] | None = None
    ) -> Decimal: ...
    @overload
    def __get__(
        self, quantity: Quantity[Fraction], owner: type[object] | None = None
    ) -> Fraction | float: ...
    @overload
    def __get__(self, quantity: Quantity[int], owner: type[object] | None = None) -> Real: ...
    @overload
    def __get__(self, quantity: Quantity[float], owner: type[object] | None = None) -> float: ...
    @overload
    def __get__(self, quantity: Quantity[Real], owner: type[object] | None = None) -> Real: ...
    @overload
    def __get__(self, quantity: Quantity[complex], owner: type[object] | None = None) -> Number: ...
    @overload
    def __get__(self, quantity: Quantity[Scalar], owner: type[object] | None = None) -> Scalar: ...
    @overload
    def __get__(
        self, quantity: Quantity[Magnitude], owner: type[object] | None = None
    ) -> Magnitude: ...
    def __set__(self, quantity: Quantity[Magnitude], value: Never) -> None: ...

class Quantity(Generic[T]):
    _value: T

    _unit: Unit

    n: _NumberValue

    def __init__(self, value: T, unit: str | Unit | _LinearUnitSource, /) -> None: ...
    @property
    def value(self) -> T: ...
    @property
    def unit(self) -> Unit: ...
    def as_pair(self) -> tuple[T, Unit]: ...
    @property
    def dimension(self) -> Dimension: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __format__(self, format_spec: str) -> str: ...
    def to_latex(self, format_spec: str = '', *, fraction: bool = True) -> str: ...
    def to_siunitx(self, format_spec: str = '') -> str: ...
    def to_html(self, format_spec: str = '') -> str: ...
    def _repr_latex_(self) -> str: ...
    def _repr_html_(self) -> str: ...
    def is_dimensionless(self) -> bool: ...
    def __bool__(self) -> bool: ...
    @overload
    def value_in(
        self: Quantity[_TypedArray[_InexactScalar]], unit: str | Unit | _LinearUnitSource, /
    ) -> _NumpyValueOf[_InexactScalar]: ...
    @overload
    def value_in(
        self: Quantity[_TypedScalar[_InexactScalar]], unit: str | Unit | _LinearUnitSource, /
    ) -> _InexactScalar: ...
    @overload
    def value_in(
        self: Quantity[_DTypeValue[_InexactScalar]], unit: str | Unit | _LinearUnitSource, /
    ) -> _NumpyValueOf[_InexactScalar]: ...
    @overload
    def value_in(self: Quantity[Array], unit: str | Unit | _LinearUnitSource, /) -> NumpyValue: ...
    @overload
    def value_in(self: Quantity[Decimal], unit: str | Unit | _LinearUnitSource, /) -> Decimal: ...
    @overload
    def value_in(
        self: Quantity[Fraction], unit: str | Unit | _LinearUnitSource, /
    ) -> Fraction | float: ...
    @overload
    def value_in[N: Real](self: Quantity[N], unit: str | Unit | _LinearUnitSource, /) -> Real: ...
    @overload
    def value_in(self: Quantity[complex], unit: str | Unit | _LinearUnitSource, /) -> Number: ...
    @overload
    def value_in(self: Quantity[Scalar], unit: str | Unit | _LinearUnitSource, /) -> Scalar: ...
    @overload
    def value_in(self, unit: str | Unit | _LinearUnitSource, /) -> Magnitude: ...
    @overload
    def to(
        self: Quantity[_TypedArray[_InexactScalar]], unit: str | Unit | _LinearUnitSource, /
    ) -> Quantity[_NumpyValueOf[_InexactScalar]]: ...
    @overload
    def to(
        self: Quantity[_TypedScalar[_InexactScalar]], unit: str | Unit | _LinearUnitSource, /
    ) -> Quantity[_InexactScalar]: ...
    @overload
    def to(
        self: Quantity[_DTypeValue[_InexactScalar]], unit: str | Unit | _LinearUnitSource, /
    ) -> Quantity[_NumpyValueOf[_InexactScalar]]: ...
    @overload
    def to(
        self: Quantity[Array], unit: str | Unit | _LinearUnitSource, /
    ) -> Quantity[NumpyValue]: ...
    @overload
    def to(
        self: Quantity[Decimal], unit: str | Unit | _LinearUnitSource, /
    ) -> Quantity[Decimal]: ...
    @overload
    def to(
        self: Quantity[Fraction], unit: str | Unit | _LinearUnitSource, /
    ) -> Quantity[Fraction | float]: ...
    @overload
    def to[N: Real](
        self: Quantity[N], unit: str | Unit | _LinearUnitSource, /
    ) -> Quantity[Real]: ...
    @overload
    def to(
        self: Quantity[complex], unit: str | Unit | _LinearUnitSource, /
    ) -> Quantity[Number]: ...
    @overload
    def to(self: Quantity[Scalar], unit: str | Unit | _LinearUnitSource, /) -> Quantity[Scalar]: ...
    @overload
    def to(self, unit: str | Unit | _LinearUnitSource, /) -> Quantity[Magnitude]: ...
    @overload
    def to_base(
        self: Quantity[_DTypeValue[_InexactScalar]],
    ) -> Quantity[_NumpyValueOf[_InexactScalar]]: ...
    @overload
    def to_base(self: Quantity[Array]) -> Quantity[NumpyValue]: ...
    @overload
    def to_base(self: Quantity[Decimal]) -> Quantity[Decimal]: ...
    @overload
    def to_base(self: Quantity[Fraction]) -> Quantity[Fraction | float]: ...
    @overload
    def to_base(self: Quantity[Real]) -> Quantity[Real]: ...
    @overload
    def to_base(self: Quantity[complex]) -> Quantity[complex]: ...
    @overload
    def to_base(self: Quantity[Scalar]) -> Quantity[Scalar]: ...
    @overload
    def to_base(self) -> Quantity[Magnitude]: ...
    @overload
    def simplify(
        self: Quantity[_DTypeValue[_InexactScalar]],
    ) -> Quantity[_NumpyValueOf[_InexactScalar]]: ...
    @overload
    def simplify(self: Quantity[Array]) -> Quantity[NumpyValue]: ...
    @overload
    def simplify(self: Quantity[Decimal]) -> Quantity[Decimal]: ...
    @overload
    def simplify(self: Quantity[Fraction]) -> Quantity[Fraction | float]: ...
    @overload
    def simplify(self: Quantity[Real]) -> Quantity[Real]: ...
    @overload
    def simplify(self: Quantity[complex]) -> Quantity[complex]: ...
    @overload
    def simplify(self: Quantity[Scalar]) -> Quantity[Scalar]: ...
    @overload
    def simplify(self) -> Quantity[Magnitude]: ...
    @overload
    def compact(
        self: Quantity[_DTypeValue[_InexactScalar]],
    ) -> Quantity[_NumpyValueOf[_InexactScalar]]: ...
    @overload
    def compact(self: Quantity[Array]) -> Quantity[NumpyValue]: ...
    @overload
    def compact(self: Quantity[Decimal]) -> Quantity[Decimal]: ...
    @overload
    def compact(self: Quantity[Fraction]) -> Quantity[Fraction | float]: ...
    @overload
    def compact(self: Quantity[Real]) -> Quantity[Real]: ...
    @overload
    def compact(self: Quantity[Scalar]) -> Quantity[Scalar]: ...
    @overload
    def compact(self) -> Quantity[Magnitude]: ...
    @overload
    def astype(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[_TypedArray[_NumpyScalar]], dtype: type[float], /
    ) -> Quantity[_NumpyArrayOf[_Float64]]: ...
    @overload
    def astype(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[_TypedScalar[_NumpyScalar]], dtype: type[float], /
    ) -> Quantity[_Float64]: ...
    @overload
    def astype(
        self: Quantity[Array], dtype: type[float], /
    ) -> Quantity[_NumpyValueOf[_Float64]]: ...
    @overload
    def astype(
        self: Quantity[_TypedArray[_NumpyScalar]], dtype: type[complex], /
    ) -> Quantity[_NumpyArrayOf[_Complex128]]: ...
    @overload
    def astype(
        self: Quantity[_TypedScalar[_NumpyScalar]], dtype: type[complex], /
    ) -> Quantity[_Complex128]: ...
    @overload
    def astype(
        self: Quantity[Array], dtype: type[complex], /
    ) -> Quantity[_NumpyValueOf[_Complex128]]: ...
    @overload
    def astype[S: _NumpyNumber](
        self: Quantity[_TypedArray[_NumpyScalar]], dtype: type[S], /
    ) -> Quantity[_NumpyArrayOf[S]]: ...
    @overload
    def astype[S: _NumpyNumber](
        self: Quantity[_TypedScalar[_NumpyScalar]], dtype: type[S], /
    ) -> Quantity[S]: ...
    @overload
    def astype(self: Quantity[Array], dtype: object, /) -> Quantity[NumpyValue]: ...
    @overload
    def astype[S: (int, float, complex, Fraction, Decimal)](
        self: Quantity[Scalar], dtype: type[S], /
    ) -> Quantity[S]: ...
    def to_ratio(self, *, kind: LogKind) -> Quantity[Magnitude]: ...
    def to_log(
        self, unit: str | Unit | _LinearUnitSource, /, *, kind: LogKind
    ) -> Quantity[Magnitude]: ...
    def define(
        self: Quantity[Fractionable],
        symbol: str,
        /,
        name: str | None = None,
        *name_aliases: str,
        symbol_aliases: str | tuple[str, ...] = (),
        prefixable: PrefixPolicy | None = None,
    ) -> Unit: ...
    def _comparison_pair(self) -> tuple[Scalar, Factor]: ...
    def _equality_parts(self) -> tuple[Scalar, Scalar, Factor]: ...
    def _equal(self, other: object, op: Callable[[Any, Any], bool]) -> bool:

        if isinstance(other, _Point): ...

    @overload
    def __eq__(self: Quantity[Array], other: object, /) -> Array: ...
    @overload
    def __eq__(  # pyright: ignore[reportOverlappingOverload]
        self, other: Quantity[Array] | Array, /
    ) -> Array: ...
    @overload
    def __eq__(self: Quantity[Scalar], other: object, /) -> bool: ...
    @overload
    def __eq__(self, other: object, /) -> bool | Array: ...
    @overload
    def __ne__(self: Quantity[Array], other: object, /) -> Array: ...
    @overload
    def __ne__(  # pyright: ignore[reportOverlappingOverload]
        self, other: Quantity[Array] | Array, /
    ) -> Array: ...
    @overload
    def __ne__(self: Quantity[Scalar], other: object, /) -> bool: ...
    @overload
    def __ne__(self, other: object, /) -> bool | Array: ...
    def __hash__(self) -> int: ...
    def _compare(self, other: object, op: Callable[[Any, Any], bool]) -> bool:

        if isinstance(other, _Point): ...

    @overload
    def __lt__(self: Quantity[Array], other: object, /) -> Array: ...
    @overload
    def __lt__(  # pyright: ignore[reportOverlappingOverload]
        self, other: Quantity[Array] | Array, /
    ) -> Array: ...
    @overload
    def __lt__(self: Quantity[Scalar], other: object, /) -> bool: ...
    @overload
    def __lt__(self, other: object, /) -> bool | Array: ...
    @overload
    def __le__(self: Quantity[Array], other: object, /) -> Array: ...
    @overload
    def __le__(  # pyright: ignore[reportOverlappingOverload]
        self, other: Quantity[Array] | Array, /
    ) -> Array: ...
    @overload
    def __le__(self: Quantity[Scalar], other: object, /) -> bool: ...
    @overload
    def __le__(self, other: object, /) -> bool | Array: ...
    @overload
    def __gt__(self: Quantity[Array], other: object, /) -> Array: ...
    @overload
    def __gt__(  # pyright: ignore[reportOverlappingOverload]
        self, other: Quantity[Array] | Array, /
    ) -> Array: ...
    @overload
    def __gt__(self: Quantity[Scalar], other: object, /) -> bool: ...
    @overload
    def __gt__(self, other: object, /) -> bool | Array: ...
    @overload
    def __ge__(self: Quantity[Array], other: object, /) -> Array: ...
    @overload
    def __ge__(  # pyright: ignore[reportOverlappingOverload]
        self, other: Quantity[Array] | Array, /
    ) -> Array: ...
    @overload
    def __ge__(self: Quantity[Scalar], other: object, /) -> bool: ...
    @overload
    def __ge__(self, other: object, /) -> bool | Array: ...
    def _value_for_isclose(self) -> Fraction | float: ...
    @overload
    def isclose(
        self: Quantity[Array],
        other: object,
        /,
        *,
        rel_tol: object = 1e-9,
        abs_tol: object = 0,
    ) -> Array: ...
    @overload
    def isclose(
        self,
        other: Quantity[Array] | Array,
        /,
        *,
        rel_tol: object = 1e-9,
        abs_tol: object = 0,
    ) -> Array: ...
    @overload
    def isclose(
        self, other: object, /, *, rel_tol: Array | Quantity[Array], abs_tol: object = 0
    ) -> Array: ...
    @overload
    def isclose(
        self, other: object, /, *, rel_tol: object = 1e-9, abs_tol: Array | Quantity[Array]
    ) -> Array: ...
    @overload
    def isclose(
        self: Quantity[Fractionable],
        other: Quantity[Fractionable] | Fractionable,
        /,
        *,
        rel_tol: Fractionable = 1e-9,
        abs_tol: Quantity[Fractionable] | Fractionable = 0,
    ) -> bool: ...
    @overload
    def __pos__(
        self: Quantity[_DTypeValue[_InexactScalar]], /
    ) -> Quantity[_NumpyValueOf[_InexactScalar]]: ...
    @overload
    def __pos__(self: Quantity[Array], /) -> Quantity[NumpyValue]: ...
    @overload
    def __pos__[N: (int, float, Fraction, Decimal, complex)](
        self: Quantity[N], /
    ) -> Quantity[N]: ...
    @overload
    def __pos__(self: Quantity[Scalar], /) -> Quantity[Scalar]: ...
    @overload
    def __pos__(self, /) -> Quantity[Magnitude]: ...
    @overload
    def __neg__(
        self: Quantity[_DTypeValue[_InexactScalar]], /
    ) -> Quantity[_NumpyValueOf[_InexactScalar]]: ...
    @overload
    def __neg__(self: Quantity[Array], /) -> Quantity[NumpyValue]: ...
    @overload
    def __neg__[N: (int, float, Fraction, Decimal, complex)](
        self: Quantity[N], /
    ) -> Quantity[N]: ...
    @overload
    def __neg__(self: Quantity[Scalar], /) -> Quantity[Scalar]: ...
    @overload
    def __neg__(self, /) -> Quantity[Magnitude]: ...
    def _add_sub(
        self,
        other: object,
        op: Callable[[Scalar, Scalar], object],
        *,
        reverse: bool = False,
    ) -> Quantity[Scalar]:

        if isinstance(other, _Point): ...

    @overload
    def __add__[P: _Point[Any]](self, other: _Point[P], /) -> P: ...
    @overload
    def __add__(
        self: Quantity[_DTypeValue[_FloatingScalar]],
        other: Quantity[_DTypeValue[_FloatingScalar]] | _DTypeValue[_FloatingScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_FloatingScalar]]: ...
    @overload
    def __add__(
        self: Quantity[_DTypeValue[_ComplexScalar]],
        other: Quantity[_DTypeValue[_ComplexScalar]] | _DTypeValue[_ComplexScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_ComplexScalar]]: ...
    @overload
    def __add__(
        self: Quantity[_DTypeValue[_Float32]],
        other: Quantity[_DTypeValue[_Complex64]] | _DTypeValue[_Complex64],
        /,
    ) -> Quantity[_NumpyValueOf[_Complex64]]: ...
    @overload
    def __add__(
        self: Quantity[_DTypeValue[_Complex64]],
        other: Quantity[_DTypeValue[_Float32]] | _DTypeValue[_Float32],
        /,
    ) -> Quantity[_NumpyValueOf[_Complex64]]: ...
    @overload
    def __add__(
        self: Quantity[_DTypeValue[_Float64]],
        other: Quantity[_DTypeValue[_Complex128]] | _DTypeValue[_Complex128],
        /,
    ) -> Quantity[_NumpyValueOf[_Complex128]]: ...
    @overload
    def __add__(
        self: Quantity[_DTypeValue[_Complex128]],
        other: Quantity[_DTypeValue[_Float64]] | _DTypeValue[_Float64],
        /,
    ) -> Quantity[_NumpyValueOf[_Complex128]]: ...
    @overload
    def __add__(
        self: Quantity[Array], other: Quantity[Array] | Array, /
    ) -> Quantity[NumpyValue]: ...
    @overload
    def __add__(
        self: Quantity[_DTypeValue[_Float64]], other: Quantity[Fractionable] | Fractionable, /
    ) -> Quantity[_NumpyValueOf[_Float64]]: ...
    @overload
    def __add__(
        self: Quantity[_DTypeValue[_Float64]], other: Quantity[complex] | complex, /
    ) -> Quantity[_NumpyValueOf[_Complex128]]: ...
    @overload
    def __add__(
        self: Quantity[Array], other: Quantity[Magnitude] | Magnitude, /
    ) -> Quantity[NumpyValue]: ...
    @overload
    def __add__(self, other: Quantity[Array] | Array, /) -> Quantity[NumpyValue]: ...
    @overload
    def __add__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[Decimal], other: Quantity[Decimal | int] | Decimal | int, /
    ) -> Quantity[Decimal]: ...
    @overload
    def __add__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[int], other: Quantity[Decimal] | Decimal, /
    ) -> Quantity[Decimal]: ...
    @overload
    def __add__(self: Quantity[Decimal], other: Quantity[float] | float, /) -> Quantity[float]: ...
    @overload
    def __add__(
        self: Quantity[float], other: Quantity[Decimal] | Decimal, /
    ) -> Quantity[float]: ...
    @overload
    def __add__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[int], other: Quantity[int] | int, /
    ) -> Quantity[Real]: ...
    @overload
    def __add__(self: Quantity[float], other: Quantity[float] | float, /) -> Quantity[float]: ...
    @overload
    def __add__[L: Real, R: Real](
        self: Quantity[L], other: Quantity[R] | R, /
    ) -> Quantity[Real]: ...
    @overload
    def __add__(
        self: Quantity[complex], other: Quantity[complex] | complex, /
    ) -> Quantity[complex]: ...
    @overload
    def __add__(
        self: Quantity[Number], other: Quantity[Number] | Number, /
    ) -> Quantity[Number]: ...
    @overload
    def __add__(
        self: Quantity[Scalar], other: Quantity[Scalar] | Scalar, /
    ) -> Quantity[Scalar]: ...
    @overload
    def __add__(self, other: Quantity[Magnitude] | Magnitude, /) -> Quantity[Magnitude]: ...
    @overload
    def __radd__(
        self: Quantity[_DTypeValue[_FloatingScalar]],
        other: Quantity[_DTypeValue[_FloatingScalar]] | _DTypeValue[_FloatingScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_FloatingScalar]]: ...
    @overload
    def __radd__(
        self: Quantity[_DTypeValue[_ComplexScalar]],
        other: Quantity[_DTypeValue[_ComplexScalar]] | _DTypeValue[_ComplexScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_ComplexScalar]]: ...
    @overload
    def __radd__(
        self: Quantity[Array], other: Quantity[Array] | Array, /
    ) -> Quantity[NumpyValue]: ...
    @overload
    def __radd__(
        self: Quantity[Array], other: Quantity[Magnitude] | Magnitude, /
    ) -> Quantity[NumpyValue]: ...
    @overload
    def __radd__(self, other: Quantity[Array] | Array, /) -> Quantity[NumpyValue]: ...
    @overload
    def __radd__(self: Quantity[Decimal], other: Decimal | int, /) -> Quantity[Decimal]: ...
    @overload
    def __radd__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[int], other: Decimal, /
    ) -> Quantity[Decimal]: ...
    @overload
    def __radd__(self: Quantity[Decimal], other: float, /) -> Quantity[float]: ...
    @overload
    def __radd__(self: Quantity[float], other: Decimal, /) -> Quantity[float]: ...
    @overload
    def __radd__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[int], other: int, /
    ) -> Quantity[Real]: ...
    @overload
    def __radd__(self: Quantity[float], other: float, /) -> Quantity[float]: ...
    @overload
    def __radd__[N: Real](self: Quantity[N], other: Real, /) -> Quantity[Real]: ...
    @overload
    def __radd__(self: Quantity[complex], other: complex, /) -> Quantity[complex]: ...
    @overload
    def __radd__(self: Quantity[Number], other: Number, /) -> Quantity[Number]: ...
    @overload
    def __radd__(self: Quantity[Scalar], other: Scalar, /) -> Quantity[Scalar]: ...
    @overload
    def __radd__(self, other: Magnitude, /) -> Quantity[Magnitude]: ...
    @overload
    def __sub__(
        self: Quantity[_DTypeValue[_FloatingScalar]],
        other: Quantity[_DTypeValue[_FloatingScalar]] | _DTypeValue[_FloatingScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_FloatingScalar]]: ...
    @overload
    def __sub__(
        self: Quantity[_DTypeValue[_ComplexScalar]],
        other: Quantity[_DTypeValue[_ComplexScalar]] | _DTypeValue[_ComplexScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_ComplexScalar]]: ...
    @overload
    def __sub__(
        self: Quantity[_DTypeValue[_Float32]],
        other: Quantity[_DTypeValue[_Complex64]] | _DTypeValue[_Complex64],
        /,
    ) -> Quantity[_NumpyValueOf[_Complex64]]: ...
    @overload
    def __sub__(
        self: Quantity[_DTypeValue[_Complex64]],
        other: Quantity[_DTypeValue[_Float32]] | _DTypeValue[_Float32],
        /,
    ) -> Quantity[_NumpyValueOf[_Complex64]]: ...
    @overload
    def __sub__(
        self: Quantity[_DTypeValue[_Float64]],
        other: Quantity[_DTypeValue[_Complex128]] | _DTypeValue[_Complex128],
        /,
    ) -> Quantity[_NumpyValueOf[_Complex128]]: ...
    @overload
    def __sub__(
        self: Quantity[_DTypeValue[_Complex128]],
        other: Quantity[_DTypeValue[_Float64]] | _DTypeValue[_Float64],
        /,
    ) -> Quantity[_NumpyValueOf[_Complex128]]: ...
    @overload
    def __sub__(
        self: Quantity[Array], other: Quantity[Array] | Array, /
    ) -> Quantity[NumpyValue]: ...
    @overload
    def __sub__(
        self: Quantity[_DTypeValue[_Float64]], other: Quantity[Fractionable] | Fractionable, /
    ) -> Quantity[_NumpyValueOf[_Float64]]: ...
    @overload
    def __sub__(
        self: Quantity[_DTypeValue[_Float64]], other: Quantity[complex] | complex, /
    ) -> Quantity[_NumpyValueOf[_Complex128]]: ...
    @overload
    def __sub__(
        self: Quantity[Array], other: Quantity[Magnitude] | Magnitude, /
    ) -> Quantity[NumpyValue]: ...
    @overload
    def __sub__(self, other: Quantity[Array] | Array, /) -> Quantity[NumpyValue]: ...
    @overload
    def __sub__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[Decimal], other: Quantity[Decimal | int] | Decimal | int, /
    ) -> Quantity[Decimal]: ...
    @overload
    def __sub__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[int], other: Quantity[Decimal] | Decimal, /
    ) -> Quantity[Decimal]: ...
    @overload
    def __sub__(self: Quantity[Decimal], other: Quantity[float] | float, /) -> Quantity[float]: ...
    @overload
    def __sub__(
        self: Quantity[float], other: Quantity[Decimal] | Decimal, /
    ) -> Quantity[float]: ...
    @overload
    def __sub__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[int], other: Quantity[int] | int, /
    ) -> Quantity[Real]: ...
    @overload
    def __sub__(self: Quantity[float], other: Quantity[float] | float, /) -> Quantity[float]: ...
    @overload
    def __sub__[L: Real, R: Real](
        self: Quantity[L], other: Quantity[R] | R, /
    ) -> Quantity[Real]: ...
    @overload
    def __sub__(
        self: Quantity[complex], other: Quantity[complex] | complex, /
    ) -> Quantity[complex]: ...
    @overload
    def __sub__(
        self: Quantity[Number], other: Quantity[Number] | Number, /
    ) -> Quantity[Number]: ...
    @overload
    def __sub__(
        self: Quantity[Scalar], other: Quantity[Scalar] | Scalar, /
    ) -> Quantity[Scalar]: ...
    @overload
    def __sub__(self, other: Quantity[Magnitude] | Magnitude, /) -> Quantity[Magnitude]: ...
    @overload
    def __rsub__(
        self: Quantity[_DTypeValue[_FloatingScalar]],
        other: Quantity[_DTypeValue[_FloatingScalar]] | _DTypeValue[_FloatingScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_FloatingScalar]]: ...
    @overload
    def __rsub__(
        self: Quantity[_DTypeValue[_ComplexScalar]],
        other: Quantity[_DTypeValue[_ComplexScalar]] | _DTypeValue[_ComplexScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_ComplexScalar]]: ...
    @overload
    def __rsub__(
        self: Quantity[Array], other: Quantity[Array] | Array, /
    ) -> Quantity[NumpyValue]: ...
    @overload
    def __rsub__(
        self: Quantity[Array], other: Quantity[Magnitude] | Magnitude, /
    ) -> Quantity[NumpyValue]: ...
    @overload
    def __rsub__(self, other: Quantity[Array] | Array, /) -> Quantity[NumpyValue]: ...
    @overload
    def __rsub__(self: Quantity[Decimal], other: Decimal | int, /) -> Quantity[Decimal]: ...
    @overload
    def __rsub__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[int], other: Decimal, /
    ) -> Quantity[Decimal]: ...
    @overload
    def __rsub__(self: Quantity[Decimal], other: float, /) -> Quantity[float]: ...
    @overload
    def __rsub__(self: Quantity[float], other: Decimal, /) -> Quantity[float]: ...
    @overload
    def __rsub__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[int], other: int, /
    ) -> Quantity[Real]: ...
    @overload
    def __rsub__(self: Quantity[float], other: float, /) -> Quantity[float]: ...
    @overload
    def __rsub__[N: Real](self: Quantity[N], other: Real, /) -> Quantity[Real]: ...
    @overload
    def __rsub__(self: Quantity[complex], other: complex, /) -> Quantity[complex]: ...
    @overload
    def __rsub__(self: Quantity[Number], other: Number, /) -> Quantity[Number]: ...
    @overload
    def __rsub__(self: Quantity[Scalar], other: Scalar, /) -> Quantity[Scalar]: ...
    @overload
    def __rsub__(self, other: Magnitude, /) -> Quantity[Magnitude]: ...
    @overload
    def __mul__(
        self, other: Unit | _LinearUnitSource | Factor | SymbolicConstant, /
    ) -> Quantity[T]: ...
    @overload
    def __mul__(
        self: Quantity[_DTypeValue[_FloatingScalar]],
        other: Quantity[_DTypeValue[_FloatingScalar]] | _DTypeValue[_FloatingScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_FloatingScalar]]: ...
    @overload
    def __mul__(
        self: Quantity[_DTypeValue[_ComplexScalar]],
        other: Quantity[_DTypeValue[_ComplexScalar]] | _DTypeValue[_ComplexScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_ComplexScalar]]: ...
    @overload
    def __mul__(
        self: Quantity[_DTypeValue[_Float32]],
        other: Quantity[_DTypeValue[_Complex64]] | _DTypeValue[_Complex64],
        /,
    ) -> Quantity[_NumpyValueOf[_Complex64]]: ...
    @overload
    def __mul__(
        self: Quantity[_DTypeValue[_Complex64]],
        other: Quantity[_DTypeValue[_Float32]] | _DTypeValue[_Float32],
        /,
    ) -> Quantity[_NumpyValueOf[_Complex64]]: ...
    @overload
    def __mul__(
        self: Quantity[_DTypeValue[_Float64]],
        other: Quantity[_DTypeValue[_Complex128]] | _DTypeValue[_Complex128],
        /,
    ) -> Quantity[_NumpyValueOf[_Complex128]]: ...
    @overload
    def __mul__(
        self: Quantity[_DTypeValue[_Complex128]],
        other: Quantity[_DTypeValue[_Float64]] | _DTypeValue[_Float64],
        /,
    ) -> Quantity[_NumpyValueOf[_Complex128]]: ...
    @overload
    def __mul__(
        self: Quantity[_DTypeValue[_Float64]],
        other: Quantity[_DTypeValue[_IntegerScalar]] | _DTypeValue[_IntegerScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_Float64]]: ...
    @overload
    def __mul__(
        self: Quantity[_DTypeValue[_Complex128]],
        other: Quantity[_DTypeValue[_IntegerScalar]] | _DTypeValue[_IntegerScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_Complex128]]: ...
    @overload
    def __mul__(
        self: Quantity[Array], other: Quantity[Array] | Array, /
    ) -> Quantity[NumpyValue]: ...
    @overload
    def __mul__(
        self: Quantity[_DTypeValue[_InexactScalar]], other: Quantity[Fractionable] | Fractionable, /
    ) -> Quantity[_NumpyValueOf[_InexactScalar]]: ...
    @overload
    def __mul__(
        self: Quantity[Fractionable],
        other: Quantity[_DTypeValue[_InexactScalar]] | _DTypeValue[_InexactScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_InexactScalar]]: ...
    @overload
    def __mul__(
        self: Quantity[_DTypeValue[_Float32]], other: Quantity[complex] | complex, /
    ) -> Quantity[_NumpyValueOf[_Complex64]]: ...
    @overload
    def __mul__(
        self: Quantity[_DTypeValue[_Float64]], other: Quantity[complex] | complex, /
    ) -> Quantity[_NumpyValueOf[_Complex128]]: ...
    @overload
    def __mul__(
        self: Quantity[_DTypeValue[_ComplexScalar]], other: Quantity[complex] | complex, /
    ) -> Quantity[_NumpyValueOf[_ComplexScalar]]: ...
    @overload
    def __mul__(
        self: Quantity[Array], other: Quantity[Magnitude] | Magnitude, /
    ) -> Quantity[NumpyValue]: ...
    @overload
    def __mul__(self, other: Quantity[Array] | Array, /) -> Quantity[NumpyValue]: ...
    @overload
    def __mul__(self: Quantity[int], other: Quantity[int] | int, /) -> Quantity[int]: ...
    @overload
    def __mul__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[Decimal], other: Quantity[Decimal | int] | Decimal | int, /
    ) -> Quantity[Decimal]: ...
    @overload
    def __mul__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[int], other: Quantity[Decimal] | Decimal, /
    ) -> Quantity[Decimal]: ...
    @overload
    def __mul__(self: Quantity[float], other: Quantity[float] | float, /) -> Quantity[float]: ...
    @overload
    def __mul__(self: Quantity[Decimal], other: Quantity[float] | float, /) -> Quantity[float]: ...
    @overload
    def __mul__(
        self: Quantity[float], other: Quantity[Decimal] | Decimal, /
    ) -> Quantity[float]: ...
    @overload
    def __mul__(
        self: Quantity[Fraction], other: Quantity[int | Fraction] | int | Fraction, /
    ) -> Quantity[Fraction]: ...
    @overload
    def __mul__(
        self: Quantity[int], other: Quantity[Fraction] | Fraction, /
    ) -> Quantity[Fraction]: ...
    @overload
    def __mul__[L: Real, R: Real](
        self: Quantity[L], other: Quantity[R] | R, /
    ) -> Quantity[Real]: ...
    @overload
    def __mul__(
        self: Quantity[complex], other: Quantity[complex] | complex, /
    ) -> Quantity[complex]: ...
    @overload
    def __mul__(
        self: Quantity[Number], other: Quantity[Number] | Number, /
    ) -> Quantity[Number]: ...
    @overload
    def __mul__(
        self: Quantity[Scalar], other: Quantity[Scalar] | Scalar, /
    ) -> Quantity[Scalar]: ...
    @overload
    def __mul__(self, other: Quantity[Magnitude] | Magnitude, /) -> Quantity[Magnitude]: ...
    @overload
    def __rmul__(
        self, other: Unit | _LinearUnitSource | Factor | SymbolicConstant, /
    ) -> Quantity[T]: ...
    @overload
    def __rmul__(
        self: Quantity[_DTypeValue[_FloatingScalar]],
        other: Quantity[_DTypeValue[_FloatingScalar]] | _DTypeValue[_FloatingScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_FloatingScalar]]: ...
    @overload
    def __rmul__(
        self: Quantity[_DTypeValue[_ComplexScalar]],
        other: Quantity[_DTypeValue[_ComplexScalar]] | _DTypeValue[_ComplexScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_ComplexScalar]]: ...
    @overload
    def __rmul__(
        self: Quantity[Array], other: Quantity[Array] | Array, /
    ) -> Quantity[NumpyValue]: ...
    @overload
    def __rmul__(
        self: Quantity[_DTypeValue[_InexactScalar]], other: Quantity[Fractionable] | Fractionable, /
    ) -> Quantity[_NumpyValueOf[_InexactScalar]]: ...
    @overload
    def __rmul__(
        self: Quantity[Fractionable],
        other: Quantity[_DTypeValue[_InexactScalar]] | _DTypeValue[_InexactScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_InexactScalar]]: ...
    @overload
    def __rmul__(
        self: Quantity[_DTypeValue[_Float32]], other: Quantity[complex] | complex, /
    ) -> Quantity[_NumpyValueOf[_Complex64]]: ...
    @overload
    def __rmul__(
        self: Quantity[_DTypeValue[_Float64]], other: Quantity[complex] | complex, /
    ) -> Quantity[_NumpyValueOf[_Complex128]]: ...
    @overload
    def __rmul__(
        self: Quantity[_DTypeValue[_ComplexScalar]], other: Quantity[complex] | complex, /
    ) -> Quantity[_NumpyValueOf[_ComplexScalar]]: ...
    @overload
    def __rmul__(
        self: Quantity[Array], other: Quantity[Magnitude] | Magnitude, /
    ) -> Quantity[NumpyValue]: ...
    @overload
    def __rmul__(self, other: Quantity[Array] | Array, /) -> Quantity[NumpyValue]: ...
    @overload
    def __rmul__(self: Quantity[int], other: int, /) -> Quantity[int]: ...
    @overload
    def __rmul__(self: Quantity[Decimal], other: Decimal | int, /) -> Quantity[Decimal]: ...
    @overload
    def __rmul__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[int], other: Decimal, /
    ) -> Quantity[Decimal]: ...
    @overload
    def __rmul__(self: Quantity[float], other: float, /) -> Quantity[float]: ...
    @overload
    def __rmul__(self: Quantity[Decimal], other: float, /) -> Quantity[float]: ...
    @overload
    def __rmul__(self: Quantity[float], other: Decimal, /) -> Quantity[float]: ...
    @overload
    def __rmul__(self: Quantity[Fraction], other: int | Fraction, /) -> Quantity[Fraction]: ...
    @overload
    def __rmul__[N: Real](self: Quantity[N], other: Real, /) -> Quantity[Real]: ...
    @overload
    def __rmul__(self: Quantity[complex], other: complex, /) -> Quantity[complex]: ...
    @overload
    def __rmul__(self: Quantity[Number], other: Number, /) -> Quantity[Number]: ...
    @overload
    def __rmul__(self: Quantity[Scalar], other: Scalar, /) -> Quantity[Scalar]: ...
    @overload
    def __rmul__(self, other: Magnitude, /) -> Quantity[Magnitude]: ...
    @overload
    def __truediv__(
        self, other: Unit | _LinearUnitSource | Factor | SymbolicConstant, /
    ) -> Quantity[T]: ...
    @overload
    def __truediv__(
        self: Quantity[_DTypeValue[_FloatingScalar]],
        other: Quantity[_DTypeValue[_FloatingScalar]] | _DTypeValue[_FloatingScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_FloatingScalar]]: ...
    @overload
    def __truediv__(
        self: Quantity[_DTypeValue[_ComplexScalar]],
        other: Quantity[_DTypeValue[_ComplexScalar]] | _DTypeValue[_ComplexScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_ComplexScalar]]: ...
    @overload
    def __truediv__(
        self: Quantity[_DTypeValue[_Float32]],
        other: Quantity[_DTypeValue[_Complex64]] | _DTypeValue[_Complex64],
        /,
    ) -> Quantity[_NumpyValueOf[_Complex64]]: ...
    @overload
    def __truediv__(
        self: Quantity[_DTypeValue[_Complex64]],
        other: Quantity[_DTypeValue[_Float32]] | _DTypeValue[_Float32],
        /,
    ) -> Quantity[_NumpyValueOf[_Complex64]]: ...
    @overload
    def __truediv__(
        self: Quantity[_DTypeValue[_Float64]],
        other: Quantity[_DTypeValue[_Complex128]] | _DTypeValue[_Complex128],
        /,
    ) -> Quantity[_NumpyValueOf[_Complex128]]: ...
    @overload
    def __truediv__(
        self: Quantity[_DTypeValue[_Complex128]],
        other: Quantity[_DTypeValue[_Float64]] | _DTypeValue[_Float64],
        /,
    ) -> Quantity[_NumpyValueOf[_Complex128]]: ...
    @overload
    def __truediv__(
        self: Quantity[_DTypeValue[_Float64]],
        other: Quantity[_DTypeValue[_IntegerScalar]] | _DTypeValue[_IntegerScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_Float64]]: ...
    @overload
    def __truediv__(
        self: Quantity[_DTypeValue[_Complex128]],
        other: Quantity[_DTypeValue[_IntegerScalar]] | _DTypeValue[_IntegerScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_Complex128]]: ...
    @overload
    def __truediv__(
        self: Quantity[Array], other: Quantity[Array] | Array, /
    ) -> Quantity[NumpyValue]: ...
    @overload
    def __truediv__(
        self: Quantity[_DTypeValue[_InexactScalar]], other: Quantity[Fractionable] | Fractionable, /
    ) -> Quantity[_NumpyValueOf[_InexactScalar]]: ...
    @overload
    def __truediv__(
        self: Quantity[Fractionable],
        other: Quantity[_DTypeValue[_InexactScalar]] | _DTypeValue[_InexactScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_InexactScalar]]: ...
    @overload
    def __truediv__(
        self: Quantity[_DTypeValue[_Float32]], other: Quantity[complex] | complex, /
    ) -> Quantity[_NumpyValueOf[_Complex64]]: ...
    @overload
    def __truediv__(
        self: Quantity[_DTypeValue[_Float64]], other: Quantity[complex] | complex, /
    ) -> Quantity[_NumpyValueOf[_Complex128]]: ...
    @overload
    def __truediv__(
        self: Quantity[_DTypeValue[_ComplexScalar]], other: Quantity[complex] | complex, /
    ) -> Quantity[_NumpyValueOf[_ComplexScalar]]: ...
    @overload
    def __truediv__(
        self: Quantity[Array], other: Quantity[Magnitude] | Magnitude, /
    ) -> Quantity[NumpyValue]: ...
    @overload
    def __truediv__(self, other: Quantity[Array] | Array, /) -> Quantity[NumpyValue]: ...
    @overload
    def __truediv__(self: Quantity[int], other: Quantity[int] | int, /) -> Quantity[float]: ...
    @overload
    def __truediv__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[Decimal], other: Quantity[Decimal | int] | Decimal | int, /
    ) -> Quantity[Decimal]: ...
    @overload
    def __truediv__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[int], other: Quantity[Decimal] | Decimal, /
    ) -> Quantity[Decimal]: ...
    @overload
    def __truediv__(
        self: Quantity[float], other: Quantity[float] | float, /
    ) -> Quantity[float]: ...
    @overload
    def __truediv__(
        self: Quantity[Decimal], other: Quantity[float] | float, /
    ) -> Quantity[float]: ...
    @overload
    def __truediv__(
        self: Quantity[float], other: Quantity[Decimal] | Decimal, /
    ) -> Quantity[float]: ...
    @overload
    def __truediv__(
        self: Quantity[Fraction], other: Quantity[int | Fraction] | int | Fraction, /
    ) -> Quantity[Fraction]: ...
    @overload
    def __truediv__(
        self: Quantity[int], other: Quantity[Fraction] | Fraction, /
    ) -> Quantity[Fraction]: ...
    @overload
    def __truediv__[L: Real, R: Real](
        self: Quantity[L], other: Quantity[R] | R, /
    ) -> Quantity[Real]: ...
    @overload
    def __truediv__(
        self: Quantity[complex], other: Quantity[complex] | complex, /
    ) -> Quantity[complex]: ...
    @overload
    def __truediv__(
        self: Quantity[Number], other: Quantity[Number] | Number, /
    ) -> Quantity[Number]: ...
    @overload
    def __truediv__(
        self: Quantity[Scalar], other: Quantity[Scalar] | Scalar, /
    ) -> Quantity[Scalar]: ...
    @overload
    def __truediv__(self, other: Quantity[Magnitude] | Magnitude, /) -> Quantity[Magnitude]: ...
    @overload
    def __rtruediv__(
        self: Quantity[_DTypeValue[_FloatingScalar]],
        other: Quantity[_DTypeValue[_FloatingScalar]] | _DTypeValue[_FloatingScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_FloatingScalar]]: ...
    @overload
    def __rtruediv__(
        self: Quantity[_DTypeValue[_ComplexScalar]],
        other: Quantity[_DTypeValue[_ComplexScalar]] | _DTypeValue[_ComplexScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_ComplexScalar]]: ...
    @overload
    def __rtruediv__(
        self: Quantity[Array], other: Quantity[Array] | Array, /
    ) -> Quantity[NumpyValue]: ...
    @overload
    def __rtruediv__(
        self: Quantity[_DTypeValue[_InexactScalar]], other: Quantity[Fractionable] | Fractionable, /
    ) -> Quantity[_NumpyValueOf[_InexactScalar]]: ...
    @overload
    def __rtruediv__(
        self: Quantity[Fractionable],
        other: Quantity[_DTypeValue[_InexactScalar]] | _DTypeValue[_InexactScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_InexactScalar]]: ...
    @overload
    def __rtruediv__(
        self: Quantity[_DTypeValue[_Float32]], other: Quantity[complex] | complex, /
    ) -> Quantity[_NumpyValueOf[_Complex64]]: ...
    @overload
    def __rtruediv__(
        self: Quantity[_DTypeValue[_Float64]], other: Quantity[complex] | complex, /
    ) -> Quantity[_NumpyValueOf[_Complex128]]: ...
    @overload
    def __rtruediv__(
        self: Quantity[_DTypeValue[_ComplexScalar]], other: Quantity[complex] | complex, /
    ) -> Quantity[_NumpyValueOf[_ComplexScalar]]: ...
    @overload
    def __rtruediv__(
        self: Quantity[Array],
        other: Unit
        | _LinearUnitSource
        | Factor
        | SymbolicConstant
        | Quantity[Magnitude]
        | Magnitude,
        /,
    ) -> Quantity[NumpyValue]: ...
    @overload
    def __rtruediv__(self, other: Quantity[Array] | Array, /) -> Quantity[NumpyValue]: ...
    @overload
    def __rtruediv__(
        self: Quantity[int], other: Unit | _LinearUnitSource | Factor | SymbolicConstant | int, /
    ) -> Quantity[float]: ...
    @overload
    def __rtruediv__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[Decimal],
        other: Unit | _LinearUnitSource | Factor | SymbolicConstant | Decimal | int,
        /,
    ) -> Quantity[Decimal]: ...
    @overload
    def __rtruediv__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[int], other: Decimal, /
    ) -> Quantity[Decimal]: ...
    @overload
    def __rtruediv__(
        self: Quantity[float],
        other: Unit | _LinearUnitSource | Factor | SymbolicConstant | float,
        /,
    ) -> Quantity[float]: ...
    @overload
    def __rtruediv__(self: Quantity[Decimal], other: float, /) -> Quantity[float]: ...
    @overload
    def __rtruediv__(self: Quantity[float], other: Decimal, /) -> Quantity[float]: ...
    @overload
    def __rtruediv__(self: Quantity[Fraction], other: int | Fraction, /) -> Quantity[Fraction]: ...
    @overload
    def __rtruediv__[N: Real](
        self: Quantity[N], other: Unit | _LinearUnitSource | Factor | SymbolicConstant | Real, /
    ) -> Quantity[Real]: ...
    @overload
    def __rtruediv__(self: Quantity[complex], other: complex, /) -> Quantity[complex]: ...
    @overload
    def __rtruediv__(self: Quantity[Number], other: Number, /) -> Quantity[Number]: ...
    @overload
    def __rtruediv__(
        self: Quantity[Scalar],
        other: Unit | _LinearUnitSource | Factor | SymbolicConstant | Scalar,
        /,
    ) -> Quantity[Scalar]: ...
    @overload
    def __rtruediv__(
        self, other: Unit | _LinearUnitSource | Factor | SymbolicConstant | Magnitude, /
    ) -> Quantity[Magnitude]: ...
    @overload
    def __pow__(
        self: Quantity[Array], exponent: Array | Quantity[Array], /
    ) -> Quantity[NumpyValue]: ...
    @overload
    def __pow__(
        self: Quantity[_DTypeValue[_InexactScalar]], exponent: int | float, /
    ) -> Quantity[_NumpyValueOf[_InexactScalar]]: ...
    @overload
    def __pow__(self: Quantity[Array], exponent: object, /) -> Quantity[NumpyValue]: ...
    @overload
    def __pow__(self, exponent: Array | Quantity[Array], /) -> Quantity[NumpyValue]: ...
    @overload
    def __pow__(self: Quantity[Decimal], exponent: int | Decimal, /) -> Quantity[Decimal]: ...
    @overload
    def __pow__(self: Quantity[int], exponent: Decimal, /) -> Quantity[Decimal]: ...
    @overload
    def __pow__(self: Quantity[int], exponent: _NonnegativeIntegerPower, /) -> Quantity[int]: ...
    @overload
    def __pow__(self: Quantity[int], exponent: _NegativeIntegerPower, /) -> Quantity[float]: ...
    @overload
    def __pow__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[int], exponent: int, /
    ) -> Quantity[Real]: ...
    @overload
    def __pow__(self: Quantity[float], exponent: int, /) -> Quantity[float]: ...
    @overload
    def __pow__[N: Real](self: Quantity[N], exponent: Real, /) -> Quantity[Number]: ...
    @overload
    def __pow__(self: Quantity[Scalar], exponent: Fractionable, /) -> Quantity[Scalar]: ...
    @overload
    def __pow__(self, exponent: Fractionable, /) -> Quantity[Magnitude]: ...
    @overload
    def __iadd__[P: _Point[Any]](self, other: _Point[P], /) -> P: ...
    @overload
    def __iadd__[A: WritableArray](self: Quantity[A], other: object, /) -> Quantity[A]: ...
    @overload
    def __iadd__(
        self: Quantity[Array], other: Quantity[Magnitude] | Magnitude, /
    ) -> Quantity[NumpyValue]: ...
    @overload
    def __iadd__(self, other: Quantity[Array] | Array, /) -> Quantity[NumpyValue]: ...
    @overload
    def __iadd__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[Decimal], other: Quantity[Decimal | int] | Decimal | int, /
    ) -> Quantity[Decimal]: ...
    @overload
    def __iadd__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[int], other: Quantity[Decimal] | Decimal, /
    ) -> Quantity[Decimal]: ...
    @overload
    def __iadd__(self: Quantity[Decimal], other: Quantity[float] | float, /) -> Quantity[float]: ...
    @overload
    def __iadd__(
        self: Quantity[float], other: Quantity[Decimal] | Decimal, /
    ) -> Quantity[float]: ...
    @overload
    def __iadd__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[int], other: Quantity[int] | int, /
    ) -> Quantity[Real]: ...
    @overload
    def __iadd__(self: Quantity[float], other: Quantity[float] | float, /) -> Quantity[float]: ...
    @overload
    def __iadd__[L: Real, R: Real](
        self: Quantity[L], other: Quantity[R] | R, /
    ) -> Quantity[Real]: ...
    @overload
    def __iadd__(
        self: Quantity[Scalar], other: Quantity[Scalar] | Scalar, /
    ) -> Quantity[Scalar]: ...
    @overload
    def __iadd__(self, other: Quantity[Magnitude] | Magnitude, /) -> Quantity[Magnitude]: ...
    @overload
    def __isub__[A: WritableArray](self: Quantity[A], other: object, /) -> Quantity[A]: ...
    @overload
    def __isub__(
        self: Quantity[Array], other: Quantity[Magnitude] | Magnitude, /
    ) -> Quantity[NumpyValue]: ...
    @overload
    def __isub__(self, other: Quantity[Array] | Array, /) -> Quantity[NumpyValue]: ...
    @overload
    def __isub__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[Decimal], other: Quantity[Decimal | int] | Decimal | int, /
    ) -> Quantity[Decimal]: ...
    @overload
    def __isub__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[int], other: Quantity[Decimal] | Decimal, /
    ) -> Quantity[Decimal]: ...
    @overload
    def __isub__(self: Quantity[Decimal], other: Quantity[float] | float, /) -> Quantity[float]: ...
    @overload
    def __isub__(
        self: Quantity[float], other: Quantity[Decimal] | Decimal, /
    ) -> Quantity[float]: ...
    @overload
    def __isub__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[int], other: Quantity[int] | int, /
    ) -> Quantity[Real]: ...
    @overload
    def __isub__(self: Quantity[float], other: Quantity[float] | float, /) -> Quantity[float]: ...
    @overload
    def __isub__[L: Real, R: Real](
        self: Quantity[L], other: Quantity[R] | R, /
    ) -> Quantity[Real]: ...
    @overload
    def __isub__(
        self: Quantity[Scalar], other: Quantity[Scalar] | Scalar, /
    ) -> Quantity[Scalar]: ...
    @overload
    def __isub__(self, other: Quantity[Magnitude] | Magnitude, /) -> Quantity[Magnitude]: ...
    @overload
    def __imul__[A: WritableArray](self: Quantity[A], other: object, /) -> Quantity[A]: ...
    @overload
    def __imul__(
        self, other: Unit | _LinearUnitSource | Factor | SymbolicConstant, /
    ) -> Quantity[T]: ...
    @overload
    def __imul__(
        self: Quantity[_DTypeValue[_FloatingScalar]],
        other: Quantity[_DTypeValue[_FloatingScalar]] | _DTypeValue[_FloatingScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_FloatingScalar]]: ...
    @overload
    def __imul__(
        self: Quantity[_DTypeValue[_ComplexScalar]],
        other: Quantity[_DTypeValue[_ComplexScalar]] | _DTypeValue[_ComplexScalar],
        /,
    ) -> Quantity[_NumpyValueOf[_ComplexScalar]]: ...
    @overload
    def __imul__(
        self: Quantity[Array], other: Quantity[Magnitude] | Magnitude, /
    ) -> Quantity[NumpyValue]: ...
    @overload
    def __imul__(self, other: Quantity[Array] | Array, /) -> Quantity[NumpyValue]: ...
    @overload
    def __imul__(self: Quantity[int], other: Quantity[int] | int, /) -> Quantity[int]: ...
    @overload
    def __imul__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[Decimal], other: Quantity[Decimal | int] | Decimal | int, /
    ) -> Quantity[Decimal]: ...
    @overload
    def __imul__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[int], other: Quantity[Decimal] | Decimal, /
    ) -> Quantity[Decimal]: ...
    @overload
    def __imul__(self: Quantity[float], other: Quantity[float] | float, /) -> Quantity[float]: ...
    @overload
    def __imul__(self: Quantity[Decimal], other: Quantity[float] | float, /) -> Quantity[float]: ...
    @overload
    def __imul__(
        self: Quantity[float], other: Quantity[Decimal] | Decimal, /
    ) -> Quantity[float]: ...
    @overload
    def __imul__[L: Real, R: Real](
        self: Quantity[L], other: Quantity[R] | R, /
    ) -> Quantity[Real]: ...
    @overload
    def __imul__(
        self: Quantity[Scalar], other: Quantity[Scalar] | Scalar, /
    ) -> Quantity[Scalar]: ...
    @overload
    def __imul__(self, other: Quantity[Magnitude] | Magnitude, /) -> Quantity[Magnitude]: ...
    @overload
    def __itruediv__[A: WritableArray](self: Quantity[A], other: object, /) -> Quantity[A]: ...
    @overload
    def __itruediv__(
        self, other: Unit | _LinearUnitSource | Factor | SymbolicConstant, /
    ) -> Quantity[T]: ...
    @overload
    def __itruediv__(
        self: Quantity[Array], other: Quantity[Magnitude] | Magnitude, /
    ) -> Quantity[NumpyValue]: ...
    @overload
    def __itruediv__(self, other: Quantity[Array] | Array, /) -> Quantity[NumpyValue]: ...
    @overload
    def __itruediv__(self: Quantity[int], other: Quantity[int] | int, /) -> Quantity[float]: ...
    @overload
    def __itruediv__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[Decimal], other: Quantity[Decimal | int] | Decimal | int, /
    ) -> Quantity[Decimal]: ...
    @overload
    def __itruediv__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[int], other: Quantity[Decimal] | Decimal, /
    ) -> Quantity[Decimal]: ...
    @overload
    def __itruediv__(
        self: Quantity[float], other: Quantity[float] | float, /
    ) -> Quantity[float]: ...
    @overload
    def __itruediv__(
        self: Quantity[Decimal], other: Quantity[float] | float, /
    ) -> Quantity[float]: ...
    @overload
    def __itruediv__(
        self: Quantity[float], other: Quantity[Decimal] | Decimal, /
    ) -> Quantity[float]: ...
    @overload
    def __itruediv__[L: Real, R: Real](
        self: Quantity[L], other: Quantity[R] | R, /
    ) -> Quantity[Real]: ...
    @overload
    def __itruediv__(
        self: Quantity[Scalar], other: Quantity[Scalar] | Scalar, /
    ) -> Quantity[Scalar]: ...
    @overload
    def __itruediv__(self, other: Quantity[Magnitude] | Magnitude, /) -> Quantity[Magnitude]: ...
    @overload
    def __ipow__[A: WritableArray](  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[A], exponent: object, /
    ) -> Quantity[A]: ...
    @overload
    def __ipow__(self: Quantity[Array], exponent: object, /) -> Quantity[NumpyValue]: ...
    @overload
    def __ipow__(self, exponent: Array | Quantity[Array], /) -> Quantity[NumpyValue]: ...
    @overload
    def __ipow__(self: Quantity[Decimal], exponent: int | Decimal, /) -> Quantity[Decimal]: ...
    @overload
    def __ipow__(self: Quantity[int], exponent: Decimal, /) -> Quantity[Decimal]: ...
    @overload
    def __ipow__(self: Quantity[int], exponent: _NonnegativeIntegerPower, /) -> Quantity[int]: ...
    @overload
    def __ipow__(self: Quantity[int], exponent: _NegativeIntegerPower, /) -> Quantity[float]: ...
    @overload
    def __ipow__(  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[int], exponent: int, /
    ) -> Quantity[Real]: ...
    @overload
    def __ipow__(self: Quantity[float], exponent: int, /) -> Quantity[float]: ...
    @overload
    def __ipow__[N: Real](self: Quantity[N], exponent: Real, /) -> Quantity[Number]: ...
    @overload
    def __ipow__(self: Quantity[Scalar], exponent: Fractionable, /) -> Quantity[Scalar]: ...
    @overload
    def __ipow__(self, exponent: Fractionable, /) -> Quantity[Magnitude]: ...
    @overload
    def __imatmul__[A: WritableArray](self: Quantity[A], other: object, /) -> Quantity[A]: ...
    @overload
    def __imatmul__(self, other: object, /) -> Quantity[NumpyValue]: ...
    def _inplace_result(self, result: Quantity[Magnitude]) -> Quantity[Magnitude]: ...
    def _has_numpy_operand(self, *others: object) -> bool: ...
    def __array__(self, dtype: object = None, copy: object = None) -> Any: ...
    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any: ...
    def __array_function__(
        self, func: Any, types: tuple[type, ...], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any: ...
    def _array_value(self) -> Any: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def dtype(self: Quantity[Array]) -> NumpyDType: ...
    @property
    def T[A: Array](self: Quantity[A]) -> Quantity[A]: ...
    @property
    def real[A: Array](self: Quantity[HasReal[A]]) -> Quantity[A]: ...
    @property
    def imag[A: Array](self: Quantity[HasImag[A]]) -> Quantity[A]: ...
    @overload
    def __getitem__[S: _NumpyNumber](
        self: Quantity[_DTypeValue[S]], key: _ArrayLeadingIndex
    ) -> Quantity[_NumpyArrayOf[S]]: ...
    @overload
    def __getitem__[S: _NumpyNumber](
        self: Quantity[_TypedArray[S]], key: Any
    ) -> Quantity[_NumpyValueOf[S]]: ...
    @overload
    def __getitem__(self: Quantity[Array], key: Any) -> Quantity[NumpyValue]: ...
    def __setitem__(
        self: Quantity[Array], key: Any, value: Quantity[Magnitude] | Magnitude
    ) -> None: ...
    def __len__(self: Quantity[Array]) -> int: ...
    @overload
    def __iter__[S: _NumpyNumber](
        self: Quantity[_TypedArray[S]],
    ) -> Iterator[Quantity[_NumpyValueOf[S]]]: ...
    @overload
    def __iter__(self: Quantity[Array]) -> Iterator[Quantity[NumpyValue]]: ...
    @overload
    def transpose[S: _NumpyScalar](
        self: Quantity[_TypedArray[S]], axes: _Axis = None, /, *more_axes: SupportsIndex
    ) -> Quantity[_NumpyArrayOf[S]]: ...
    @overload
    def transpose[S: _NumpyNumber](
        self: Quantity[_TypedScalar[S]], axes: _Axis = None, /, *more_axes: SupportsIndex
    ) -> Quantity[S]: ...
    @overload
    def transpose(
        self: Quantity[Array], axes: _Axis = None, /, *more_axes: SupportsIndex
    ) -> Quantity[NumpyValue]: ...
    @overload
    def squeeze[S: _NumpyScalar](
        self: Quantity[_TypedArray[S]], axis: _Axis = None
    ) -> Quantity[_NumpyArrayOf[S]]: ...
    @overload
    def squeeze[S: _NumpyNumber](
        self: Quantity[_TypedScalar[S]], axis: _Axis = None
    ) -> Quantity[S]: ...
    @overload
    def squeeze(self: Quantity[Array], axis: _Axis = None) -> Quantity[NumpyValue]: ...
    @overload
    def reshape[S: _NumpyScalar](
        self: Quantity[_TypedArray[S]],
        shape: _Shape,
        /,
        *dimensions: SupportsIndex,
        order: Literal['C', 'F', 'A'] = 'C',
        copy: bool | None = None,
    ) -> Quantity[_NumpyArrayOf[S]]: ...
    @overload
    def reshape(
        self: Quantity[Array],
        shape: _Shape,
        /,
        *dimensions: SupportsIndex,
        order: Literal['C', 'F', 'A'] = 'C',
        copy: bool | None = None,
    ) -> Quantity[NumpyValue]: ...
    @overload
    def ravel[S: _NumpyScalar](
        self: Quantity[_TypedArray[S]],
        order: Literal['C', 'F', 'A', 'K'] = 'C',
    ) -> Quantity[_NumpyArrayOf[S]]: ...
    @overload
    def ravel(
        self: Quantity[Array],
        order: Literal['C', 'F', 'A', 'K'] = 'C',
    ) -> Quantity[NumpyArray]: ...
    @overload
    def copy[S: _NumpyScalar](
        self: Quantity[_TypedArray[S]],
        order: Literal['C', 'F', 'A', 'K'] = 'C',
    ) -> Quantity[_NumpyArrayOf[S]]: ...
    @overload
    def copy(
        self: Quantity[Array],
        order: Literal['C', 'F', 'A', 'K'] = 'C',
    ) -> Quantity[NumpyArray]: ...
    @overload
    def sum(
        self: Quantity[_DTypeValue[_InexactScalar]],
        axis: None = None,
        dtype: None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        *,
        where: object = True,
        initial: object = 0,
    ) -> Quantity[_InexactScalar]: ...
    @overload
    def sum(
        self: Quantity[_DTypeValue[_InexactScalar]],
        axis: _Axis = None,
        dtype: None = None,
        out: None = None,
        keepdims: bool = False,
        *,
        where: object = True,
        initial: object = 0,
    ) -> Quantity[_NumpyValueOf[_InexactScalar]]: ...
    @overload
    def sum(
        self: Quantity[Array],
        axis: _Axis = None,
        dtype: object = None,
        out: None = None,
        keepdims: bool = False,
        *,
        where: object = True,
        initial: object = 0,
    ) -> Quantity[NumpyValue]: ...
    @overload
    def sum[O: WritableArray](
        self: Quantity[Array],
        axis: _Axis,
        dtype: object,
        out: Quantity[O],
        keepdims: bool = False,
        *,
        where: object = True,
        initial: object = 0,
    ) -> Quantity[O]: ...
    @overload
    def sum[O: WritableArray](
        self: Quantity[Array],
        axis: _Axis = None,
        dtype: object = None,
        *,
        out: Quantity[O],
        keepdims: bool = False,
        where: object = True,
        initial: object = 0,
    ) -> Quantity[O]: ...
    @overload
    def mean(
        self: Quantity[_DTypeValue[_InexactScalar]],
        axis: None = None,
        dtype: None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        *,
        where: object = True,
    ) -> Quantity[_InexactScalar]: ...
    @overload
    def mean(
        self: Quantity[_DTypeValue[_InexactScalar]],
        axis: _Axis = None,
        dtype: None = None,
        out: None = None,
        keepdims: bool = False,
        *,
        where: object = True,
    ) -> Quantity[_NumpyValueOf[_InexactScalar]]: ...
    @overload
    def mean(
        self: Quantity[Array],
        axis: _Axis = None,
        dtype: object = None,
        out: None = None,
        keepdims: bool = False,
        *,
        where: object = True,
    ) -> Quantity[NumpyValue]: ...
    @overload
    def mean[O: WritableArray](
        self: Quantity[Array],
        axis: _Axis,
        dtype: object,
        out: Quantity[O],
        keepdims: bool = False,
        *,
        where: object = True,
    ) -> Quantity[O]: ...
    @overload
    def mean[O: WritableArray](
        self: Quantity[Array],
        axis: _Axis = None,
        dtype: object = None,
        *,
        out: Quantity[O],
        keepdims: bool = False,
        where: object = True,
    ) -> Quantity[O]: ...
    @overload
    def std(
        self: Quantity[Array],
        axis: _Axis = None,
        dtype: object = None,
        out: None = None,
        ddof: object = ...,
        keepdims: bool = False,
        *,
        where: object = True,
        mean: object = ...,
        correction: object = ...,
    ) -> Quantity[NumpyValue]: ...
    @overload
    def std[O: WritableArray](
        self: Quantity[Array],
        axis: _Axis,
        dtype: object,
        out: Quantity[O],
        ddof: object = ...,
        keepdims: bool = False,
        *,
        where: object = True,
        mean: object = ...,
        correction: object = ...,
    ) -> Quantity[O]: ...
    @overload
    def std[O: WritableArray](
        self: Quantity[Array],
        axis: _Axis = None,
        dtype: object = None,
        *,
        out: Quantity[O],
        ddof: object = ...,
        keepdims: bool = False,
        where: object = True,
        mean: object = ...,
        correction: object = ...,
    ) -> Quantity[O]: ...
    def flatten(self: Quantity[Array], order: str = 'C') -> Quantity[NumpyArray]: ...
    @overload
    def __abs__(
        self: Quantity[_DTypeValue[_Complex64]], /
    ) -> Quantity[_NumpyValueOf[_Float32]]: ...
    @overload
    def __abs__(
        self: Quantity[_DTypeValue[_Complex128]], /
    ) -> Quantity[_NumpyValueOf[_Float64]]: ...
    @overload
    def __abs__(
        self: Quantity[_DTypeValue[_FloatingScalar]], /
    ) -> Quantity[_NumpyValueOf[_FloatingScalar]]: ...
    @overload
    def __abs__(self: Quantity[Array]) -> Quantity[NumpyValue]: ...
    @overload
    def __abs__(self) -> Quantity[Magnitude]: ...
    def __matmul__(self, other: object) -> Quantity[NumpyValue]: ...
    def __rmatmul__(self, other: object) -> Quantity[NumpyValue]: ...

def quantity(value: float, /, unit: str | Unit | _LinearUnitSource) -> Quantity[Decimal]: ...
