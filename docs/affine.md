# Linear differences and affine points

`Unit` and `Quantity` describe linear scales and differences. `AffineUnit`
describes a linear coordinate's physical origin and difference scale;
`AffineQuantity` represents a reading in that coordinate. Logarithmic coordinates
have separate [LogUnit and LogQuantity types](logarithmic.md).

```pycon
>>> from decimal import Decimal
>>> from fractions import Fraction
>>> from unitx import AffineQuantity, AffineUnit, Quantity, Unit
>>> from unitx.imperial import imperial
>>> from unitx.si import si
>>> type(Unit('degC')) is Unit
True
>>> 20 * si.degC == Quantity(20, 'K')
True
>>> type(20 @ si.degC) is AffineQuantity
True
>>> print((20 @ si.degC).to(imperial.degF))
68 °F

```

`value @ coordinate` constructs a point; `value * coordinate` constructs a
Quantity difference without applying the origin. The syntax is directional:
`coordinate @ value` is unsupported. `AffineQuantity(value, coordinate)` is the
explicit point constructor. `Quantity(value, coordinate)` and ordinary Quantity
conversions use the descriptor's linear difference identity. Quantity matrix
multiplication retains its ordinary numerical meaning.

Point identity belongs to the quantity class, including zero-origin coordinates:
`300 * si.K` is a difference and `300 @ si.K` is a point. They do not compare equal.

## Point conversion and physical exit

`.to(coordinate)` returns an AffineQuantity, including conversion to K. It accepts
only linear affine coordinates. `.value_in(coordinate)` returns the new reading.
`.to_quantity(unit)` explicitly evaluates a physical Quantity for multiplication
or other ordinary quantity arithmetic.

```pycon
>>> point = 0 @ si.degC
>>> type(point.to(si.K)) is AffineQuantity
True
>>> point.value_in(si.K)
Fraction(5463, 20)
>>> point.to_quantity(si.K) == Quantity(Fraction(5463, 20), si.K)
True
>>> point.to_quantity(si.K) * 2 == Quantity(Fraction(5463, 10), si.K)
True

```

The target of `.to_quantity()` is always interpreted as a linear difference unit,
including a temperature descriptor. Thus `(20 @ si.degC).to('degC')` is a point
reading 20 °C, whereas `.to_quantity('degC')` evaluates its physical value using
the degree-Celsius difference scale, whose step equals one kelvin.

String coordinates use bound definition metadata. `degC`, `(degC)` and
`[degree Celsius]` select the same origin. Prefixes and compound expressions
have their ordinary linear meaning; they do not create new coordinate origins.

## Point and difference arithmetic

| Operation | Result |
| --- | --- |
| point - compatible point | Quantity in the left coordinate's difference unit |
| point + difference; difference + point | AffineQuantity in the point's coordinate |
| point - difference | AffineQuantity in the point's coordinate |
| point + point; difference - point | TypeError |
| point multiplication, division, powers, negation, absolute value | TypeError |

```pycon
>>> hot, cold = 300 @ si.K, 20 @ si.degC
>>> hot - cold == Quantity(Fraction(137, 20), si.K)
True
>>> cold + (hot - cold) == hot
True
>>> (30 @ si.degC) - (68 @ imperial.degF) == Quantity(10, si.K)
True

```

Negative readings can be constructed directly, for example `(-20) @ si.degC`;
negating an existing point is undefined. No absolute-zero restriction is imposed.
Complex readings and booleans are unsupported.

## Defining coordinates

`AffineUnit(reference, /, *, at=0, delta_unit=None)` means that reading `at`
represents `reference`. A reference is a physical Quantity or an AffineQuantity
point. The difference scale defaults to the reference's difference unit.

```pycon
>>> celsius = AffineUnit(Decimal('273.15') * si.K)
>>> fahrenheit = AffineUnit(0 @ celsius, at=32, delta_unit=imperial.degR)
>>> (212 @ fahrenheit).value_in(celsius)
Fraction(100, 1)
>>> fahrenheit.origin == Quantity(Fraction(45967, 180), si.K)
True

```

The descriptor stores only its normalized physical `origin` and complete
`delta_unit`, including exact additional scales. Defining values and `at` must
be finite real Python scalars excluding bool. Stored float values are preserved
as their exact binary rationals; Decimal/Fraction definitions do not round in
the active Decimal context. No reference-coordinate chain is retained.

For nonzero `at`, the difference scale must have a rational factor relative to
the physical reference unit. Otherwise normalization would require a symbolic
sum that a single scalar Quantity cannot retain exactly. Construction raises
TypeError instead of approximating that origin. Supply the physical origin
directly with `at=0` to use an irrational difference scale.

Descriptors are immutable and hashable, with no multiplicative `factor` property.
Multiplication, division and powers on a descriptor operate on its linear side.
Name a coordinate with `.define(symbol, name, ...)`. A definition remains
standalone unless collected by a UnitCatalog or bound through a Registry.

```pycon
>>> from unitx import Registry
>>> local_kelvin = si.K.define('localK', 'local kelvin')
>>> epoch = AffineUnit(10 * local_kelvin).define('localT', 'local temperature')
>>> receiver = Registry([epoch])
>>> AffineQuantity(3, receiver.parse('localT')).value_in(si.K)
Fraction(13, 1)

```

Definitions capture bound identities, not expressions reinterpreted in the
default registry. Named projections retain their own difference identity;
anonymous coordinates use their explicit delta_unit. Reconstruction requires
matching definitions to be loaded in the receiving process.

## Comparison, tolerances and arrays

Finite scalar equality respects exact stored values and symbolic factors. For
example, float `-273.15` need not equal the exact Celsius reading of 0 K. Use
Decimal/Fraction definitions or a difference tolerance where appropriate.

`point.isclose(other, rel_tol=0, abs_tol=difference)` uses coordinate differences;
relative tolerance must be zero. The absolute tolerance is a compatible Quantity
or real zero. NaNs compare unequal; unsupported exact symbolic sums require an
explicit numerical conversion before comparison.

Numeric NumPy arrays/scalars are supported. Comparisons produce masks;
`np.asarray(point)` rejects implicit unit removal. Mean/median preserve the point
coordinate, standard deviation returns a difference, and variance returns its
squared unit. Arithmetic reductions such as summing points are unsupported.
Indexed assignment and supported output operations stage conversion and casting
before writing fixed-unit, fixed-dtype buffers. See [NumPy support](numpy.md).
