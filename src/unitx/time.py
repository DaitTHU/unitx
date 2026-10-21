"""Nanosecond clocks and explicit adapters for scalar durations.

Clock readings are ordinary Quantities, without an epoch or clock tag. Only
subtract readings from the same clock. Nanosecond storage does not imply
nanosecond clock resolution; standard-library platform limitations apply.
"""

from __future__ import annotations

import math
import time as _time
from datetime import timedelta
from decimal import Decimal
from fractions import Fraction

from ._numerics import Fractionable, is_numpy_object
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
]


def time() -> Quantity[int]:
    """Return Unix time in integer nanoseconds, using time.time_ns().

    This wall clock can move backwards when the system clock is adjusted.
    """
    return Quantity(_time.time_ns(), si.ns)


def monotonic() -> Quantity[int]:
    """Return a monotonic reading in integer nanoseconds; its origin is undefined."""
    return Quantity(_time.monotonic_ns(), si.ns)


def perf_counter() -> Quantity[int]:
    """Return the performance counter in integer nanoseconds, including sleep."""
    return Quantity(_time.perf_counter_ns(), si.ns)


def process_time() -> Quantity[int]:
    """Return process CPU time in integer nanoseconds, excluding sleep."""
    return Quantity(_time.process_time_ns(), si.ns)


def thread_time() -> Quantity[int]:
    """Return thread CPU time in integer nanoseconds, where the platform supports it."""
    return Quantity(_time.thread_time_ns(), si.ns)


def _seconds(duration: Quantity[Fractionable]) -> Fraction:
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
