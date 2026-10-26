"""Unit-aware scalar statistics using Python's statistics numerical kernels.

Samples are Python int/float/Fraction/Decimal values or scalar quantities and
points. Each variable uses its first sample's complete unit or coordinate.
NumPy is not imported. Covariance, correlation and fmean use float results.
"""

from __future__ import annotations

import statistics as _statistics
from collections.abc import Callable, Iterable
from decimal import Decimal
from fractions import Fraction
from typing import cast, overload

from ._coordinates.resolve import _Coordinate, convert_point
from ._numerics.conversion import compatible
from .affinequantity import AffineQuantity
from .affineunit import AffineUnit
from .dimension import DIMENSIONLESS
from .exceptions import DimensionError
from .logquantity import LogQuantity
from .logunit import LogUnit
from .quantity import Quantity
from .unit import Unit

__all__ = [
    'StatisticsError',
    'mean',
    'fmean',
    'variance',
    'pvariance',
    'stdev',
    'pstdev',
    'covariance',
    'correlation',
]

StatisticsError = _statistics.StatisticsError

type _Scalar = int | float | Fraction | Decimal
type _Observation = Quantity[_Scalar] | AffineQuantity[_Scalar] | LogQuantity[_Scalar]
type _Sample = _Scalar | _Observation
type _Reference = Unit | _Coordinate


def _observation(value: _Sample) -> _Observation:
    q = (
        value
        if isinstance(value, (Quantity, AffineQuantity, LogQuantity))
        else Quantity(value, Unit())
    )
    if type(q.value) not in (int, float, Fraction, Decimal):
        error_message = 'Scalar statistics require Python int, float, Fraction or Decimal samples.'
        raise TypeError(error_message)
    return q


def _samples(data: Iterable[_Sample]) -> tuple[list[_Observation], _Reference]:
    samples = [_observation(value) for value in data]
    reference = samples[0].unit if samples else Unit()
    for sample in samples:
        if isinstance(sample.unit, Unit) and isinstance(reference, Unit):
            valid = compatible(sample.unit, reference)
        elif isinstance(sample.unit, AffineUnit) and isinstance(reference, AffineUnit):
            valid = sample.unit._compatible_with(reference)
        elif isinstance(sample.unit, LogUnit) and isinstance(reference, LogUnit):
            valid = sample.unit._compatible_with(reference)
        elif isinstance(sample.unit, _Coordinate) and isinstance(reference, _Coordinate):
            valid = False
        else:
            error_message = 'Statistics cannot mix points and differences.'
            raise TypeError(error_message)
        if not valid:
            error_message = (
                'Statistics require compatible units or coordinates within each variable.'
            )
            raise DimensionError(error_message)
    return samples, reference


def _values(samples: list[_Observation], reference: _Reference) -> list[_Scalar]:
    return [cast(_Scalar, convert_point(q.value, q.unit, reference)) for q in samples]


def _wrap(value: _Scalar, reference: _Reference) -> _Observation:
    if isinstance(reference, Unit):
        return Quantity(value, reference)
    if isinstance(reference, AffineUnit):
        return AffineQuantity(value, reference)
    return LogQuantity(value, reference)


def _delta(reference: _Reference) -> Unit:
    return reference if isinstance(reference, Unit) else reference.delta_unit


@overload
def mean(data: Iterable[_Scalar | Quantity[_Scalar]]) -> Quantity[_Scalar]: ...
@overload
def mean(data: Iterable[AffineQuantity[_Scalar]]) -> AffineQuantity[_Scalar]: ...
@overload
def mean(data: Iterable[LogQuantity[_Scalar]]) -> LogQuantity[_Scalar]: ...


def mean(data: Iterable[_Sample]) -> _Observation:
    """Return the arithmetic mean in the first sample's unit or coordinate."""
    samples, reference = _samples(data)
    function = cast(Callable[[Iterable[_Scalar]], _Scalar], _statistics.mean)
    return _wrap(function(_values(samples, reference)), reference)


@overload
def fmean(
    data: Iterable[_Scalar | Quantity[_Scalar]], weights: Iterable[_Sample] | None = None
) -> Quantity[float]: ...
@overload
def fmean(
    data: Iterable[AffineQuantity[_Scalar]], weights: Iterable[_Sample] | None = None
) -> AffineQuantity[float]: ...
@overload
def fmean(
    data: Iterable[LogQuantity[_Scalar]], weights: Iterable[_Sample] | None = None
) -> LogQuantity[float]: ...


def fmean(
    data: Iterable[_Sample], weights: Iterable[_Sample] | None = None
) -> Quantity[float] | AffineQuantity[float] | LogQuantity[float]:
    """Return a floating mean with optional coherent dimensionless weights."""
    samples, reference = _samples(data)
    raw_weights = None
    if weights is not None:
        raw_weights = []
        for value in weights:
            q = _observation(value)
            if not isinstance(q, Quantity):
                error_message = 'Weights must be linear dimensionless values.'
                raise TypeError(error_message)
            if q.dimension != DIMENSIONLESS:
                error_message = 'Weights must be dimensionless.'
                raise DimensionError(error_message)
            raw_weights.append(float(cast(_Scalar, q.value_in(Unit()))))
    result = _statistics.fmean(map(float, _values(samples, reference)), weights=raw_weights)
    return cast(
        Quantity[float] | AffineQuantity[float] | LogQuantity[float], _wrap(result, reference)
    )


def _dispersion(
    function: Callable[..., _Scalar],
    data: Iterable[_Sample],
    center: _Sample | None,
    *,
    square: bool,
) -> Quantity[_Scalar]:
    samples, reference = _samples(data)
    raw_center = None
    if center is not None:
        # Validate center semantics independently before converting to the sample reference.
        _, center_reference = _samples([samples[0], center] if samples else [center])
        raw_center = _values([_observation(center)], center_reference)[0]
    result = function(_values(samples, reference), raw_center)
    delta = _delta(reference)
    return Quantity(result, delta**2 if square else delta)


def variance(data: Iterable[_Sample], xbar: _Sample | None = None) -> Quantity[_Scalar]:
    """Return sample variance in the squared difference unit."""
    return _dispersion(cast(Callable[..., _Scalar], _statistics.variance), data, xbar, square=True)


def pvariance(data: Iterable[_Sample], mu: _Sample | None = None) -> Quantity[_Scalar]:
    """Return population variance in the squared difference unit."""
    return _dispersion(cast(Callable[..., _Scalar], _statistics.pvariance), data, mu, square=True)


def stdev(data: Iterable[_Sample], xbar: _Sample | None = None) -> Quantity[float | Decimal]:
    """Return sample standard deviation in the difference unit."""
    return cast(
        Quantity[float | Decimal],
        _dispersion(cast(Callable[..., _Scalar], _statistics.stdev), data, xbar, square=False),
    )


def pstdev(data: Iterable[_Sample], mu: _Sample | None = None) -> Quantity[float | Decimal]:
    """Return population standard deviation in the difference unit."""
    return cast(
        Quantity[float | Decimal],
        _dispersion(cast(Callable[..., _Scalar], _statistics.pstdev), data, mu, square=False),
    )


def covariance(x: Iterable[_Sample], y: Iterable[_Sample], /) -> Quantity[float]:
    """Return floating sample covariance in the product of difference units."""
    xs, xu = _samples(x)
    ys, yu = _samples(y)
    result = _statistics.covariance(
        list(map(float, _values(xs, xu))), list(map(float, _values(ys, yu)))
    )
    return Quantity(result, _delta(xu) * _delta(yu))


def correlation(
    x: Iterable[_Sample], y: Iterable[_Sample], /, *, method: str = 'linear'
) -> Quantity[float]:
    """Return Pearson or ranked correlation as a dimensionless Quantity."""
    xs, xu = _samples(x)
    ys, yu = _samples(y)
    function = cast(Callable[..., float], _statistics.correlation)
    if method == 'ranked':
        # Native ranking compares the original objects, preserving exact scalar ties/order.
        result = function(xs, ys, method=method)
    else:
        result = function(
            list(map(float, _values(xs, xu))), list(map(float, _values(ys, yu))), method=method
        )
    return Quantity(result, Unit())
