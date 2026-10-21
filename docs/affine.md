# Linear differences and affine points

`Unit` and `Quantity` always describe linear scales and differences.
`AffineUnit` and `AffineQuantity` describe absolute coordinate readings.
The quantity class determines point/difference identity, including when the
coordinate has a zero origin such as kelvin.

```pycon
>>> from decimal import Decimal
>>> from fractions import Fraction
>>> from unitx import AffineQuantity, AffineUnit, Quantity, Unit
>>> from unitx.si import si
>>> type(Unit('degC')) is Unit
True
>>> Quantity(1, 'K').to('degC') == Quantity(1, 'degC')
True
>>> 0 * si.degC == si.degC * 0 == Quantity(0, 'degC')
True
>>> type(0 @ si.degC) is AffineQuantity
True
>>> (0 @ si.K).to('degC') == AffineQuantity(Decimal('-273.15'), 'degC')
True
>>> print((20 @ si.degC).to('degF'))
68 °F

```

`value @ unit` reads as "value at this coordinate". It constructs a point;
`value * unit` and `unit * value` construct linear Quantities. Only bare numeric
magnitudes can use the `@ Unit` construction. Matrix multiplication between
Quantity objects retains its ordinary meaning. The syntax is directional:
`unit @ value` is not a point constructor. The explicit constructor
`AffineQuantity(value, unit)` remains available. Quantity(value, descriptor)
and Quantity.to(descriptor) always take the linear side; their point counterparts
use the coordinate. These rules do not depend on the magnitude or calling context.

`si.degC`, `si.degF`, `si.dBm`, `si.dBW`, and `si.dBV` are AffineUnit
descriptors containing a coordinate origin and its difference scale.
Ordinary SI units such as `si.K` remain Unit objects. Quantity constructors,
linear conversions, and unit algebra project descriptors to their linear side;
@ and AffineQuantity use the complete coordinate. Quantity.unit always returns Unit. For temperature differences, 1 °C = 1 K
and 1 °F = 5/9 K. For logarithmic differences, the dBm/dBW/dBV spellings
carry the dB scale and logarithmic degree; their physical references apply
only when used as point coordinates. Unit spellings retain separate formal
identities even when their conversion factors agree.

## Point conversion and physical exit

| Point coordinate | Difference unit | Physical quantity at reading zero |
| --- | --- | --- |
| K | K | 0 K |
| °C | K | 273.15 K |
| °F | °R | (45967/180) K |
| dBm | dB | 1 mW |
| dBW | dB | 1 W |
| dBV | dB | 1 V |

`.to(unit)` returns an AffineQuantity and `.value_in(unit)` returns its new
reading. Conversion to K, W or V preserves point identity. Use
`.to_quantity(unit)` to explicitly obtain an ordinary physical Quantity and
perform physical arithmetic.

```pycon
>>> point = 0 @ si.degC
>>> type(point.to('K')) is AffineQuantity
True
>>> point.value_in('K')
Fraction(5463, 20)
>>> point.to_quantity('K') == Quantity(Fraction(5463, 20), 'K')
True
>>> (30 @ si.dBm).to('W').value
Fraction(1, 1)
>>> (1 @ si.W).to('dBm').value
30.0
>>> (20 @ si.dBV).to_quantity('V').value
10

```

Celsius/Fahrenheit transformations retain exact rational scales and offsets.
Common unit factors are cancelled before evaluation, including irrational
factors that cancel exactly. Decimal conversions and compatible Decimal point
arithmetic combine defining coefficients before rounding once in the active
context. Integer powers of ten and the 30 dB shift between dBm and dBW also
have exact paths. Other nonlinear evaluations can require approximation.

The target of `.to_quantity()` is always interpreted as a linear unit,
including a temperature descriptor. Consequently
`(20 @ si.degC).to('degC')` is a point reading 20 degrees Celsius, whereas
`(20 @ si.degC).to_quantity('degC')` is a linear Quantity of 293.15 Celsius
steps measured from physical zero. Prefer `.to_quantity('K')` when computing
with absolute temperature. Points and differences use the same unit symbols
in readable output; inspect their class or `repr()` when the distinction matters.

Finite float values retain their stored binary rational values in exact
comparisons. Thus the decimal literal `-273.15` is an approximation and need
not equal the exact Celsius value of 0 K. Use `Decimal('-273.15')`, Fraction,
or a difference tolerance with `.isclose()` as appropriate.

## Point and difference arithmetic

| Operation | Result |
| --- | --- |
| point - compatible point | Quantity difference, in the left coordinate's difference unit |
| point + difference | AffineQuantity in the point coordinate |
| difference + point | AffineQuantity in the point coordinate |
| point - difference | AffineQuantity in the point coordinate |
| point + point; difference - point | TypeError |
| point multiplication, division, powers, negation, absolute value | TypeError |

```pycon
>>> hot, cold = 300 @ si.K, 20 @ si.degC
>>> hot - cold == Quantity(Fraction(137, 20), 'K')
True
>>> cold + (hot - cold) == hot
True
>>> (30 @ si.degC) - (68 @ si.degF) == Quantity(10, 'K')
True
>>> print((20 @ si.degC) + 10 * si.K)
30 °C
>>> (30 @ si.dBm) - (0 @ si.dBW) == Quantity(0, 'dB')
True

```

`Quantity(300, 'K')` is a difference; `AffineQuantity(300, 'K')` is a point.
They are different objects and do not compare equal. Changing the coordinate
of a point never changes it into a difference. Negative readings can be
constructed directly, for example `(-20) @ si.degC`; negating an existing
point is undefined. Complex point readings are unsupported.

## Defining coordinates

Use `AffineUnit(reference, /, *, at=0, delta_unit=None, kind=None)` to define a
coordinate. Reading `at` in the new coordinate represents `reference`.
A Quantity reference is an ordinary physical quantity; an AffineQuantity
reference is an absolute reading in another coordinate.

```pycon
>>> celsius = AffineUnit(Decimal('273.15') * si.K)
>>> fahrenheit = AffineUnit(0 @ celsius, at=32, delta_unit=si.degR)
>>> (212 @ fahrenheit).value_in(celsius)
Fraction(100, 1)
>>> fahrenheit.origin == Quantity(Fraction(45967, 180), 'K')
True

```

Linear coordinates default to the reference's Unit, or the reference point's
`delta_unit`. Logarithmic coordinates require an explicit difference unit and
`kind='power'` or `'amplitude'`. Reference magnitudes and `at` must be finite
real Python scalars, excluding bool; logarithmic physical references must be
positive. Defining Decimal/Fraction/float values are rationalized before
arithmetic without Decimal context rounding or float denominator limiting.

The `origin` property stores the physical quantity at reading zero, normalized
to a linear Unit. No nested point-reference chain is retained. The descriptor
is immutable and hashable; it has no multiplicative `factor` property.
Use `delta_unit.factor` for the coordinate step. Multiplication, division and
powers on the descriptor operate on its linear side, like a Unit; operations
on an AffineQuantity point remain restricted.

Register with `.define(symbol, name, ...)`. Registration adds both the linear
difference spelling and its coordinate metadata. Registered descriptors retain
this spelling when projected to a linear Unit; anonymous descriptors use their
delta_unit. For example Quantity(1, si.degC), 1 * si.degC, and si.degC * 1
all store Unit('degC') and display 1 °C, with no origin offset. `.define()` returns an
AffineUnit; `Unit(symbol)` returns its linear difference unit, and
`AffineQuantity(value, symbol)` or `value @ Unit(symbol)` resolves its point
coordinate. UnitParser continues to parse linear products, so expressions such
as `degC/m`, `degC/degC`, and `degC**0` are ordinary unit algebra.

Registry UnitDefinition factor/dimension/log_degree describe the linear unit.
Its optional `reference`, `delta`, and `kind` store coordinate metadata as
primitive data, without importing Quantity. Prefixes remain governed by the
registered spelling policy; built-in shifted coordinates disable them.
Register additional ScaledUnit reference/difference units before using them
in registered coordinate definitions, to preserve explicit scales.
Custom definitions must exist in another process before reconstruction.

## Logarithmic differences and ratios

Physical Dimension retains seven rational exponents. Logarithmic difference
semantics are separately tracked by `Unit.log_degree`: dB has degree one,
dB/m degree one, dB² degree two and dB/Np degree zero. Direct linear conversion,
addition and comparison require compatible physical dimensions and degrees.
`is_dimensionless()` checks physical dimension only.

B, dB and Np retain exact structural scales: 1 B = ln(10)/2 Np and 1 dB = B/10.
`unitx.factor.ln10` represents the opaque identity ln(10) with a float
approximation; a parser does not evaluate its symbol.

```pycon
>>> (20 * si.dB).to_ratio(kind='power').value
100
>>> (20 * si.dB).to_ratio(kind='amplitude').value
10
>>> Quantity(100, '').to_log('dB', kind='power').value
20.0
>>> ((2 * Unit('dB/m')) * (10 * si.m)).to_ratio(kind='power').value
100

```

A logarithmic difference does not determine power versus amplitude convention.
Use explicit `.to_ratio(kind=...)` and `.to_log(unit, kind=...)`; `.to('')`
does not erase logarithmic degree. Negative physical values cannot be
converted to logarithmic readings. Zero maps to negative infinity.
NaNs and positive infinity propagate, and negative infinite readings restore
zero physical value. No additional nonnegative-temperature restriction is imposed.

## Comparison, tolerances and arrays

Only compatible points compare as points. Point/difference equality is False,
and ordering across those domains raises DimensionError. Linear scalar points
compare through exact coordinate alignment, retaining symbolic factors.
In particular, `AffineQuantity(180, 'deg') != AffineQuantity(math.pi, 'rad')`:
**`math.pi` is a finite binary rational**, while the degree factor retains
mathematical pi symbolically. The float stored in `unitx.factor.pi.value`
is only its numerical approximation; see [scalar precision](scalars.md).
Common origins cancel before comparison. Shifted coordinates whose reading
and origin cannot combine without a sum of different symbolic factors raise
TypeError; use `.isclose()` with a nonzero difference tolerance or explicitly
convert to one coordinate first. Array comparisons use NumPy numerical
conversion and do not promise scalar symbolic equality.
Same-convention logarithmic points have
exact comparison paths when their references can align exactly. Cross nonlinear
coordinate families require an explicit conversion to one coordinate before
comparison; approximate alignment is not advertised as exact equality.
All AffineQuantity objects are unhashable. Ordinary Quantity hashing is unchanged.

AffineQuantity.isclose defaults to `rel_tol=0, abs_tol=0`; it accepts a
Quantity difference tolerance, for example
`(20 @ si.degC).isclose(21 @ si.degC, abs_tol=1 * si.K)`.
Nonzero relative tolerances are rejected because relative coordinate errors
depend on the origin. NumPy retains its own tolerance defaults; use `rtol=0`
and an explicit compatible `atol` for affine points.

Optional NumPy supports `array @ unit` point construction, preserving the array
object; `array * unit` remains linear construction. NumPy's own left-operand
stubs do not always infer wrapper results; AffineQuantity(array, unit) supplies
precise construction typing.

- Indexing, iteration, reshape, transpose, flatten, mean, median, extrema, and
  joining compatible points preserve AffineQuantity identity and view/copy rules.
- Standard deviation returns Quantity in the difference unit; variance has its square.
  Mean dBm averages coordinate readings, corresponding to a geometric mean of power.
- Finite predicates, indices and comparisons return bare arrays/scalars.
- Sums, products, cumulative arithmetic, point multiplication and powers are rejected.
- Assignment and out require matching point/difference classes. Values convert to
  the fixed coordinate before staged same-kind writing. Failed conversion/casting
  preserves destination storage, including masks, aliases and read-only arrays.
- Array `point += difference` and `point -= difference` preserve the point and buffer.
  `point -= point` produces a difference and cannot write back into a point array.
- Implicit `np.asarray(point)` and `np.array(point)` are forbidden. Use `.value`
  or `.value_in(unit)` explicitly. NumPy remains an optional dependency.
