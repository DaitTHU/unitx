# Standard-library enhancements

`unitx.math` provides a finite set of unit-aware real functions. `unitx.time`
provides integer nanosecond clock readings and explicit duration adapters.
Both work without NumPy installed.

## Supported modules

| Module | Use |
| --- | --- |
| `unitx.math` | Unit-aware real mathematical functions with scalar and optional NumPy backends |
| `unitx.time` | Integer ns clocks, scalar sleep, timedelta and aware datetime conversion |

These are explicit adapters, not complete replacements for Python's modules.
For numerical precision see [scalar behavior](scalars.md); for annotations
see [typing](typing.md).

## Mathematical functions

```pycon
>>> from unitx import math, pi, time
>>> from unitx.si import si
>>> math.sqrt(9 * si.cm**2)
Quantity(3.0, Unit('cm'))
>>> math.sin(90 * si.deg).value
1.0
>>> math.sin(pi / 2).value
1.0
>>> math.hypot(3 * si.m, 400 * si.cm)
Quantity(5.0, Unit('m'))
>>> round(math.atan2(1 * si.m, 100 * si.cm).value_in(si.deg))
45

```

All results are Quantities, including calls with bare numbers. Bare numbers
represent coherent dimensionless values. `Factor` and `SymbolicConstant`
arguments are evaluated numerically; these functions do not perform symbolic
identities or arbitrary-precision evaluation.

| Functions | Input and result units |
| --- | --- |
| `sqrt(x)`, `cbrt(x)` | Any ordinary linear unit; result unit is its exact 1/2 or 1/3 power |
| `sin(x)`, `cos(x)`, `tan(x)` | Dimensionless input, including deg/rad; coherent dimensionless result |
| `asin(x)`, `acos(x)`, `atan(x)` | Dimensionless input; result in rad |
| `exp(x)`, `expm1(x)` | Dimensionless input; coherent dimensionless result |
| `log(x)`, `log1p(x)`, `log2(x)`, `log10(x)` | Dimensionless input; coherent dimensionless result |
| `hypot(*coordinates)` | Compatible input dimensions; first coordinate's unit |
| `atan2(y, x)` | Compatible input dimensions; result in rad |

Dimensionless inputs are first converted to coherent values: `km/m` carries
a factor of 1000, and degrees are converted to radians. This rule also applies
to exp/log and inverse trigonometric functions, consistent with unitx's
dimensionless angle model. `log` is the natural logarithm and takes one argument.
Use `log(x) / log(base)` for a different base. `hypot()` returns dimensionless
zero; `hypot(x)` returns the absolute value in x's unit. For a bare first
coordinate, the result unit is coherent dimensionless.

Python int, float, Fraction and Decimal inputs use the standard-library math
backend, returning `Quantity[float]`. Decimal/Fraction input does not imply
arbitrary-precision output. Standard math domain, overflow, NaN and infinity
behavior applies. In particular, negative `sqrt` raises ValueError; negative
`cbrt` returns a real cube root. Complex inputs, bools, custom scalar backends,
implicit list/tuple arrays, affine points and logarithmic units are unsupported.
Explicitly convert a logarithmic quantity to a physical ratio first.

Plain real numeric NumPy arrays/scalars, bare or wrapped in Quantity, use the
optional NumPy adapter. Broadcasting, dtype, and `np.errstate` rules apply;
negative real square roots can produce NaN or raise FloatingPointError.
Array `hypot` combines coordinates with pairwise `np.hypot` calls, so rounding
may differ from the standard library's multi-coordinate implementation.
The wrappers accept positional inputs only, without `out` or `where`.
Use the supported NumPy ufuncs directly for those options.

Array overloads return `Quantity[Array]`, using the NumPy-independent magnitude
protocol so strict scalar users have complete annotations without NumPy.
At runtime, magnitudes are native NumPy arrays/scalars. The protocol exposes
shape/ndim/size/dtype and explicit array conversion, without promising a
particular result shape or dtype. For additional native array methods in
static analysis, use `np.asarray(result.value)`.

## Clock readings

```python
from unitx import time
from unitx.si import si

stamp = time.time()  # AffineQuantity[int], ns since the Unix epoch
start = time.perf_counter()  # AffineQuantity[int], ns from an undefined origin
# ... work ...
elapsed = time.perf_counter() - start
print(elapsed.to(si.ms))
time.sleep(10 * si.ms)
```

| Function | Standard-library source | Meaning |
| --- | --- | --- |
| `time()` | `time.time_ns()` | Unix wall time; can move backwards on clock adjustment |
| `monotonic()` | `time.monotonic_ns()` | Monotonic clock with an undefined origin |
| `perf_counter()` | `time.perf_counter_ns()` | Performance counter, including sleep |
| `process_time()` | `time.process_time_ns()` | Process CPU time, excluding sleep |
| `thread_time()` | `time.thread_time_ns()` | Thread CPU time, where supported |

Every reading has a Python int magnitude and unit ns. No float timestamp is
created or multiplied by 10**9, so large readings retain every stored tick.
Use these names directly; separate `_ns` aliases are unnecessary because the
unit is part of the result. Platform availability and actual clock resolution
follow the standard library: nanosecond storage does not promise nanosecond
measurement accuracy.

Readings are AffineQuantity points. Subtracting two readings returns a Quantity
duration; adding or subtracting a duration shifts a point. Adding two points or
multiplying a point is rejected, as is passing a point to `sleep()` or
`to_timedelta()`. `.to('s')` changes the reading's scale and preserves its point
class. `.value_in('s')` explicitly extracts the numeric reading in seconds.

Points do not carry a clock identity, epoch label, timezone, process or thread tag.
Only compare or subtract readings from the same clock context (and the same
process or thread for CPU clocks). Mixing contexts is not detected. The ns/s
coordinate describes a scale relative to each clock's own origin, not a
conversion between clocks. `time()` provides a Unix timestamp; the other
clocks cannot be interpreted as Unix timestamps. Use `monotonic` or
`perf_counter` for elapsed wall time.

## Datetimes and known epochs

`from_datetime(instant)` accepts an aware datetime and returns an
`AffineQuantity[int]` in ns since 1970-01-01 00:00:00 UTC, matching `time()`.
It converts to UTC and combines integer days, seconds and microseconds; it
does not pass through a float timestamp. Naive datetimes, including a tzinfo
whose `utcoffset()` returns None, raise ValueError. The UTC instant must fit
datetime's year 1–9999 range.

`to_datetime(point, *, tz=timezone.utc)` interprets a linear time coordinate's
physical reference as a Unix offset. It includes the coordinate origin,
rounds once to the nearest microsecond with ties to even, then converts from
UTC to the requested tzinfo. None is rejected rather than selecting the
machine's local zone. The UTC and target-local results must fit datetime's
range; otherwise conversion raises OverflowError. Rational coefficients and
finite Decimal readings remain exact until final rounding, independently of
the Decimal context. Irrational coefficients can require approximation.

```pycon
>>> from datetime import datetime, timezone
>>> from unitx import AffineUnit
>>> epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
>>> time.from_datetime(epoch).value
0
>>> time.to_datetime(time.from_datetime(epoch)) == epoch
True
>>> epoch_2000 = datetime(2000, 1, 1, tzinfo=timezone.utc)
>>> seconds_since_2000 = AffineUnit(time.from_datetime(epoch_2000), delta_unit='s')
>>> time.from_datetime(epoch_2000).value_in(seconds_since_2000)
Fraction(0, 1)
>>> time.to_datetime(60 @ seconds_since_2000).isoformat()
'2000-01-01T00:01:00+00:00'
>>> (60 @ seconds_since_2000).value_in('s')
Fraction(946684860, 1)

```

Known epochs use ordinary AffineUnit references; no registration is required.
For these coordinates, the zero-origin 's'/'ns' coordinates mean Unix time.
Do not pass monotonic, performance-counter or CPU readings to `to_datetime()`:
they do not have a known Unix reference, and this adapter cannot detect their
provenance. Explicit custom coordinates likewise require the caller to supply
the correct Unix reference.

Timezone offsets and DST folds follow datetime/tzinfo behavior. An aware
datetime round trip preserves its instant and microseconds, but the point does
not store its timezone; pass the desired timezone on export. Neither adapter
adds leap-second handling, calendar arithmetic, or a timezone database.

## Duration adapters

```pycon
>>> from datetime import timedelta
>>> time.from_timedelta(timedelta(seconds=2, microseconds=3))
Quantity(2000003, Unit('µs'))
>>> time.to_timedelta(1500 * si.ns)
datetime.timedelta(microseconds=2)
>>> time.to_timedelta(-1500 * si.ns)
datetime.timedelta(days=-1, seconds=86399, microseconds=999998)

```

`from_timedelta()` combines days, seconds and microseconds exactly into
`Quantity[int]` with unit microseconds. It avoids `total_seconds()` and
preserves the full range, including negative durations.

`to_timedelta()` accepts a finite scalar duration Quantity. It rounds to the
nearest microsecond with ties to even, independently of the Decimal context,
then lets timedelta enforce its range. Overflow raises OverflowError. A full
`timedelta -> Quantity -> timedelta` round trip is exact. Rational unit scales
and finite Decimal values remain exact up to the final microsecond rounding;
irrational scales use the existing Factor numerical approximation. Float
values are interpreted as their stored binary rationals, so a decimal-looking
float need not lie exactly on a rounding tie.

`sleep()` accepts a finite nonnegative scalar duration Quantity, converts it
to float seconds after validation, and calls standard-library sleep. Negative
values are checked before float conversion, even if their magnitude would
underflow to zero. Standard-library scheduling and range limits apply;
positive durations below the float range can round to zero.

`to_timedelta()` and `sleep()` accept Python int/float/Fraction/Decimal magnitudes,
with time dimension and zero logarithmic degree. Bare numbers, arrays, NumPy
scalars, complex values, bools, nonfinite magnitudes and affine points are
unsupported. These adapters do not implement calendar arithmetic, timezone
conversion, or leap-second handling.

The backends follow the Python documentation for [math](https://docs.python.org/3/library/math.html),
[time](https://docs.python.org/3/library/time.html), and
[datetime and timedelta](https://docs.python.org/3/library/datetime.html).

[Documentation index](README.md) · [NumPy support](numpy.md)
