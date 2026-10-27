from datetime import datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
from typing import assert_type

from unitx import AffineQuantity, Quantity, math, pi, time

assert_type(time.time(), AffineQuantity[int])
assert_type(time.monotonic(), AffineQuantity[int])
assert_type(time.perf_counter(), AffineQuantity[int])
assert_type(time.process_time(), AffineQuantity[int])
assert_type(time.thread_time(), AffineQuantity[int])
assert_type(time.from_timedelta(timedelta(seconds=1)), Quantity[int])
assert_type(time.to_timedelta(Quantity(Decimal('0.5'), 's')), timedelta)
assert_type(time.sleep(Quantity(Fraction(1, 2), 'ms')), None)
assert_type(time.from_datetime(datetime(2000, 1, 1, tzinfo=timezone.utc)), AffineQuantity[int])
assert_type(time.to_datetime(time.time()), datetime)
assert_type(time.to_datetime(time.time(), tz=timezone.utc), datetime)
assert_type(math.sqrt(Quantity(4, 'm2')), Quantity[float])
assert_type(math.cbrt(Fraction(1, 3)), Quantity[float])
assert_type(math.sin(Quantity(90, 'deg')), Quantity[float])
assert_type(math.sin(pi / 2), Quantity[float])
assert_type(math.asin(Decimal('0.5')), Quantity[float])
assert_type(math.exp(1), Quantity[float])
assert_type(math.exp(1j), Quantity[complex])
assert_type(math.exp(Quantity(1j, 'percent')), Quantity[complex])
assert_type(math.conj(1j), Quantity[complex])
assert_type(math.conj(Quantity(Decimal('1.25'), 'm')), Quantity[Decimal])
assert_type(math.log(Decimal('2')), Quantity[float])
assert_type(math.hypot(Quantity(3, 'm'), Quantity(400, 'cm')), Quantity[float])
assert_type(math.atan2(Quantity(1, 'm'), Quantity(100, 'cm')), Quantity[float])
