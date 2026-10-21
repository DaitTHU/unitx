from datetime import timedelta
from decimal import Decimal
from fractions import Fraction
from typing import assert_type

from unitx import Quantity, math, pi, time

assert_type(time.time(), Quantity[int])
assert_type(time.monotonic(), Quantity[int])
assert_type(time.perf_counter(), Quantity[int])
assert_type(time.process_time(), Quantity[int])
assert_type(time.thread_time(), Quantity[int])
assert_type(time.from_timedelta(timedelta(seconds=1)), Quantity[int])
assert_type(time.to_timedelta(Quantity(Decimal('0.5'), 's')), timedelta)
assert_type(time.sleep(Quantity(Fraction(1, 2), 'ms')), None)
assert_type(math.sqrt(Quantity(4, 'm2')), Quantity[float])
assert_type(math.cbrt(Fraction(1, 3)), Quantity[float])
assert_type(math.sin(Quantity(90, 'deg')), Quantity[float])
assert_type(math.sin(pi / 2), Quantity[float])
assert_type(math.asin(Decimal('0.5')), Quantity[float])
assert_type(math.exp(1), Quantity[float])
assert_type(math.log(Decimal('2')), Quantity[float])
assert_type(math.hypot(Quantity(3, 'm'), Quantity(400, 'cm')), Quantity[float])
assert_type(math.atan2(Quantity(1, 'm'), Quantity(100, 'cm')), Quantity[float])
