"""Optional Matplotlib axis conversion, enabled explicitly with `register()`.

Matplotlib owns the numeric plotting boundary. Unit-bearing input remains
wrapped until its converter requests values in the axis unit or coordinate.
Importing unitx itself neither imports Matplotlib nor registers converters.
"""

from typing import Any, TypeAlias

from matplotlib.units import (  # pyright: ignore[reportMissingImports]
    AxisInfo,
    ConversionInterface,
    registry,
)

from .affinequantity import AffineQuantity
from .affineunit import AffineUnit
from .exceptions import DimensionError
from .logquantity import LogQuantity
from .logunit import LogUnit
from .quantity import Quantity
from .unit import Unit

__all__ = ['register']

_AxisUnit: TypeAlias = Unit | AffineUnit | LogUnit
_QUANTITIES = (Quantity, AffineQuantity, LogQuantity)


def _axis_unit(unit: Any) -> _AxisUnit:
    if not isinstance(unit, (Unit, AffineUnit, LogUnit)):
        error_message = 'Matplotlib axis units must be Unit, AffineUnit or LogUnit objects.'
        raise TypeError(error_message)
    return unit


class _QuantityConverter(ConversionInterface):
    @staticmethod
    def axisinfo(unit: Any, axis: Any) -> AxisInfo:
        return AxisInfo(label=str(_axis_unit(unit)) or '1')

    @staticmethod
    def default_units(value: Any, axis: Any) -> _AxisUnit | None:
        if isinstance(value, _QUANTITIES):
            return value.unit
        if isinstance(value, (list, tuple)):
            for item in value:
                unit = _QuantityConverter.default_units(item, axis)
                if unit is not None:
                    return unit
        return None

    @staticmethod
    def convert(value: Any, unit: Any, axis: Any) -> Any:
        if isinstance(value, _QUANTITIES):
            target = value.unit if unit is None else _axis_unit(unit)
            if isinstance(value, Quantity):
                if not isinstance(target, Unit):
                    error_message = 'A plot axis cannot mix points and differences.'
                    raise DimensionError(error_message)
                raw = value.value_in(target)
            elif isinstance(value, AffineQuantity):
                if not isinstance(target, AffineUnit) or not value.unit._compatible_with(target):
                    error_message = (
                        'A plot axis requires compatible points with the same convention.'
                    )
                    raise DimensionError(error_message)
                raw = value.value_in(target)
            else:
                if not isinstance(target, LogUnit) or not value.unit._compatible_with(target):
                    error_message = (
                        'A plot axis requires compatible points with the same convention.'
                    )
                    raise DimensionError(error_message)
                raw = value.value_in(target)
            if (
                isinstance(raw, complex)
                or getattr(getattr(raw, 'dtype', None), 'kind', None) == 'c'
            ):
                error_message = 'Plot real values explicitly, for example np.real(q) or np.imag(q).'
                raise TypeError(error_message)
            return raw
        if isinstance(value, (list, tuple)):
            return [_QuantityConverter.convert(item, unit, axis) for item in value]
        # Bare limits and Matplotlib's own numeric defaults are already axis readings.
        return value


_CONVERTER = _QuantityConverter()


def register() -> None:
    """Enable unitx inputs at Matplotlib's unit conversion boundary.

    Registering repeatedly is safe. The first plotted quantity selects each
    axis's default unit; `xunits`/`yunits` or `axis.set_units()` can select an
    explicit Unit or coordinate. Numeric bare values remain axis readings.
    This adapter is intended for real data and does not enable arbitrary
    Matplotlib functions that coerce inputs before axis conversion.
    """
    for cls in _QUANTITIES:
        registry[cls] = _CONVERTER
