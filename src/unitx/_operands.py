"""Minimal operand boundaries, independent of coordinate implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from .dimension import Dimension
    from .unit import Unit

Result = TypeVar('Result', covariant=True)


class _Point(Generic[Result]):
    """Identify readings; Result is the type produced by adding a difference.

    Concrete families select their widened numerical result type. This avoids
    treating a reading as a structurally compatible scalar in linear overloads.
    The marker stores no state and implements no arithmetic.
    """

    __slots__ = ()

    if TYPE_CHECKING:

        @property
        def value(self) -> Any: ...

        @property
        def unit(self) -> Any: ...

        @property
        def dimension(self) -> Dimension: ...


class _LinearUnitSource:
    """Provide a linear projection without exposing a coordinate mapping."""

    __slots__ = ()

    def _as_linear_unit(self) -> Unit:
        raise NotImplementedError


def linear_unit(unit: object) -> Unit:
    """Normalize unit inputs once at an explicit linear-unit boundary."""
    from .unit import Unit

    if isinstance(unit, Unit):
        return unit
    if isinstance(unit, str):
        return Unit(unit)
    if isinstance(unit, _LinearUnitSource):
        return unit._as_linear_unit()
    error_message = 'Expected a Unit, coordinate descriptor, or unit spelling.'
    raise TypeError(error_message)
