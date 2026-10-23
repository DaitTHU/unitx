"""Nanosecond clock points and explicit datetime/duration adapters.

Clock readings are AffineQuantity points; their differences are Quantity
durations. Points do not carry a clock, epoch label, process or thread tag: only
subtract or compare readings from the same clock context. Nanosecond storage
does not imply nanosecond clock resolution; platform limitations apply.
"""

from __future__ import annotations

import math
import time as _time
from datetime import datetime, timedelta, timezone, tzinfo
from decimal import Decimal
from fractions import Fraction

from ._numerics import Fractionable, Magnitude, is_numpy_object
from .affinequantity import AffineQuantity
from .quantity import Quantity
from .si import si

__all__ = [
    'time',
    'monotonic',
    'perf_counter',
    'process_time',
    'thread_time',
    'sleep',
    'from_timedelta',
    'to_timedelta',
    'from_datetime',
    'to_datetime',
]

_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def time() -> AffineQuantity[int]:
    """Return Unix time in integer nanoseconds, using time.time_ns().

    This wall clock can move backwards when the system clock is adjusted.
    """
    return AffineQuantity(_time.time_ns(), si.ns)


def monotonic() -> AffineQuantity[int]:
    """Return a monotonic reading in integer nanoseconds; its origin is undefined."""
    return AffineQuantity(_time.monotonic_ns(), si.ns)


def perf_counter() -> AffineQuantity[int]:
    """Return the performance counter in integer nanoseconds, including sleep."""
    return AffineQuantity(_time.perf_counter_ns(), si.ns)


def process_time() -> AffineQuantity[int]:
    """Return process CPU time in integer nanoseconds, excluding sleep."""
    return AffineQuantity(_time.process_time_ns(), si.ns)


def thread_time() -> AffineQuantity[int]:
    """Return thread CPU time in integer nanoseconds, where the platform supports it."""
    return AffineQuantity(_time.thread_time_ns(), si.ns)


def _seconds(duration: Quantity[Magnitude]) -> Fraction:
    if not isinstance(duration, Quantity):
        raise TypeError('Expected a scalar duration Quantity.')
    value = duration.value
    if is_numpy_object(value) or isinstance(value, bool) or not isinstance(value, Fractionable):
        raise TypeError('Duration magnitude must be int, float, Fraction, or Decimal.')
    if (isinstance(value, float) and not math.isfinite(value)) or (
        isinstance(value, Decimal) and not value.is_finite()
    ):
        raise ValueError('Duration must be finite.')
    # Rationalize before conversion, preserving Decimal input independently of
    # its arithmetic context and float input as its stored binary rational.
    seconds = Quantity(Fraction(value), duration.unit).value_in('s')
    if isinstance(seconds, float) and not math.isfinite(seconds):
        raise ValueError('Duration must be finite.')
    return Fraction(seconds)


def sleep(duration: Quantity[Fractionable], /) -> None:
    """Sleep for a finite nonnegative scalar duration.

    Seconds enter the standard library's float domain after validation. Actual
    suspension can be longer than requested, as with time.sleep().
    """
    seconds = _seconds(duration)
    if seconds < 0:
        raise ValueError('Sleep duration must be nonnegative.')
    _time.sleep(float(seconds))


def from_timedelta(duration: timedelta, /) -> Quantity[int]:
    """Convert a timedelta to an exact integer count of microseconds."""
    if not isinstance(duration, timedelta):
        raise TypeError('Expected datetime.timedelta.')
    microseconds = (duration.days * 86400 + duration.seconds) * 1_000_000 + duration.microseconds
    return Quantity(microseconds, si.us)


def to_timedelta(duration: Quantity[Fractionable], /) -> timedelta:
    """Convert a finite scalar duration, rounding to the nearest microsecond.

    Ties round to even, independently of the Decimal context. Negative
    durations are supported; values outside timedelta's range raise OverflowError.
    """
    return timedelta(microseconds=round(_seconds(duration) * 1_000_000))


def from_datetime(instant: datetime, /) -> AffineQuantity[int]:
    """Convert an aware datetime to exact integer nanoseconds since the Unix epoch.

    Naive datetimes are rejected. UTC conversion follows the supplied timezone,
    including its fold handling. The UTC instant must fit datetime's range.
    """
    if not isinstance(instant, datetime):
        raise TypeError('Expected an aware datetime.datetime.')
    if instant.utcoffset() is None:
        raise ValueError('Datetime must be timezone-aware.')
    duration = instant.astimezone(timezone.utc) - _UNIX_EPOCH
    return AffineQuantity(from_timedelta(duration).value * 1000, si.ns)


def to_datetime(instant: AffineQuantity[Fractionable], /, *, tz: tzinfo = timezone.utc) -> datetime:
    """Interpret a linear time point relative to the Unix epoch and export datetime.

    The coordinate's physical reference is interpreted as a Unix time offset;
    custom known epochs are supported. Unknown-origin and CPU clock readings
    must not be passed here: clock provenance is not checked.
    Round once to microseconds, ties to even, then convert from UTC to tz.
    UTC and target-local datetimes must both fit datetime's range.
    """
    if not isinstance(instant, AffineQuantity):
        raise TypeError('Expected a scalar AffineQuantity time point.')
    if not isinstance(tz, tzinfo):
        raise TypeError('tz must be a datetime.tzinfo instance.')
    # Rationalize the stored reading before scaling, and include the origin
    # before the single final rounding. Neither operation uses Decimal context.
    seconds = _seconds(Quantity(instant.value, instant.unit.delta_unit))
    seconds += _seconds(instant.unit.origin)
    utc = _UNIX_EPOCH + timedelta(microseconds=round(seconds * 1_000_000))
    return utc.astimezone(tz)
