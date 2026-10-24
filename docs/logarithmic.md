# Logarithmic differences and readings

A logarithmic difference such as dB is a linear difference scale. A referenced
reading such as dBm additionally defines a physical reference and a power or
amplitude convention. `LogUnit` owns that mapping and `LogQuantity` owns a reading;
[Celsius and other affine coordinates](affine.md) use separate types.

```pycon
>>> from decimal import Decimal
>>> from fractions import Fraction
>>> from unitx import LogQuantity, LogUnit, Quantity, Unit
>>> from unitx.si import si
>>> level = 30 @ si.dBm
>>> type(level) is LogQuantity
True
>>> level.to_quantity(si.W)
Quantity(Fraction(1, 1), Unit('W'))
>>> level.to(si.dBW).value
0
>>> LogQuantity.from_quantity(1 * si.W, si.dBm).value
30.0

```

## Explicit physical boundaries

`.to()` and `.value_in()` accept logarithmic coordinates with compatible physical
dimensions. They do not convert to physical units such as W or V. Use
`.to_quantity(unit)` to evaluate the physical Quantity, and
`LogQuantity.from_quantity(quantity, coordinate)` to construct a reading from a
physical Quantity. These operations are the explicit exponential/logarithmic
boundaries. Ordinary `Quantity.to()` remains a linear difference conversion.

The constructors `LogQuantity(value, 'dBm')` and `value @ si.dBm` store a reading
directly. `AffineQuantity(value, 'dBm')` is invalid. A bare logarithmic difference
such as dB has no physical reference and cannot serve as a LogUnit by itself.

Multiplying a descriptor still constructs a difference: `20 * si.dBm` uses its
named logarithmic difference identity without applying the milliwatt reference.
Point minus compatible point returns a Quantity difference; point plus/minus a
compatible difference returns a LogQuantity. Point addition and multiplicative
point algebra are unsupported. Power/amplitude conventions must agree for point
subtraction and comparison; explicit coordinate conversion can change convention.

```pycon
>>> (30 @ si.dBm) - (0 @ si.dBW) == 0 * si.dB
True
>>> (20 @ si.dBm) + 10 * si.dB == 30 @ si.dBm
True
>>> (20 @ si.dBV).to_quantity(si.V).value
10

```

## Defining logarithmic coordinates

For a zero-reading reference, `si.dB(reference, kind=...)` is a shorthand for
`LogUnit(reference, delta_unit=si.dB, kind=...)`. A Unit argument means one of
that unit; a Quantity argument supplies the full reference value. Watts and
volts, including their SI prefixes and extra exact scales, default to
`kind='power'` and `kind='amplitude'`, respectively. This convention follows
the named unit definition, not physical dimension: other units and composite
expressions require an explicit `kind`. An explicit `kind` always takes precedence.

```pycon
>>> si.dB(si.mW) is si.dBm
True
>>> si.dB(si.W) is si.dBW
True
>>> si.dB(si.V) is si.dBV
True
>>> millivolt_level = si.dB(si.mV)
>>> (20 @ millivolt_level).to_quantity(si.mV).value
10
>>> (0 @ millivolt_level).to(si.dBV).value
-60
>>> pressure_level = si.dB(20 * si.uPa, kind='amplitude')
>>> (0 @ pressure_level).to_quantity(si.uPa).value
20

```

References with value one and unit mW, W or V, using their usual conventions,
return the existing named descriptors. Other references return
anonymous LogUnits without registering new names. The millivolt reference
corresponds to dBmV; dBm refers to a milliwatt. Voltage-to-power conversion
requires additional physical information such as impedance.

Only the formal dB unit supports this call. Reconstructed dB units behave the
same way; other Units raise TypeError when called. As a consequence of the
shared `Unit.__call__` method, `callable(unit)` is true for all Unit objects
and does not test whether this shorthand is supported.

`LogUnit(reference, /, *, delta_unit, kind, at=0)` means reading `at` represents
`reference`. A reference is a physical Quantity or LogQuantity. Both `delta_unit`
and `kind='power'|'amplitude'` are required. References must be positive finite
real Python scalars; defining readings must be finite. No user callbacks or
implicit mapping selection are involved.

```pycon
>>> power_level = LogUnit(1 * si.W, delta_unit=si.dB, kind='power', at=30)
>>> power_level.origin == 1 * si.mW
True
>>> (30 @ power_level).to_quantity(si.W).value
1
>>> voltage_level = LogUnit(10 * si.V, delta_unit=si.dB, kind='amplitude', at=20)
>>> voltage_level.origin == 1 * si.V
True

```

The defining `reference` and its reading `at` are retained exactly. A reading
`x` represents `reference * 10**((x - at) * delta.factor / coefficient)`, with the
same convention coefficients used for logarithmic differences. At `x == at`,
the exponential is one, so no approximate zero-reading reference enters the
definition. `origin` evaluates the physical quantity at reading zero and can
require an approximate exponential; it is a derived value, not stored state.
Full exact reference and difference units are retained. `.define()` creates a named descriptor;
UnitCatalog and Registry bind it using the same identity and spelling rules as
other definitions. Copy, pickle and scalar JSON exchange preserve its concrete
coordinate family and reference metadata.

Anonymous logarithmic coordinates compare their exact defining anchor,
difference Unit and convention. Evaluating `origin` and constructing another
coordinate from its approximation does not reproduce the original definition.
Equivalent readings can still compare equal across different coordinate
definitions through exact coordinate alignment.

```pycon
>>> anchored = LogUnit(1 * si.W, at=1, delta_unit=si.dB, kind='power')
>>> anchored.at, anchored.reference
(Fraction(1, 1), Quantity(Fraction(1, 1), Unit('W')))
>>> (Decimal(1) @ anchored).to_quantity(si.W).value
Decimal('1')

```

## Difference algebra and physical dimensions

Dimension owns the complete algebraic signature. Multiplication, division and
rational powers combine every exponent, so dB/m, dB² and dB/Np retain their
algebraic meanings without an independent public `log_degree` attribute.

**Complete compatibility** compares Dimension objects directly. **Physical
quantity dimension** uses `.physical`, named physical exponents, `.as_tuple()`
or `.as_dict()`, which consider only the seven physical axes. `is_dimensionless()`
also describes physical dimension only. Text formatting shows physical axes;
`repr()` and pickle retain the complete algebraic signature.

```pycon
>>> si.dB.dimension.physical == si.one.dimension
True
>>> si.dB.is_dimensionless()
True
>>> si.dB.dimension == si.one.dimension
False
>>> Unit('dB/m').dimension.physical == (1 / si.m).dimension
True
>>> Unit('dB/Np').dimension == si.one.dimension
True
>>> (si.dB.dimension ** Fraction(1, 3)) ** 3 == si.dB.dimension
True

```

B, dB and Np retain exact structural scales: 1 B = ln(10)/2 Np and 1 dB = B/10.
A logarithmic difference alone does not select the power/amplitude convention.
Use explicit `.to_ratio(kind=...)` and `.to_log(unit, kind=...)`:

```pycon
>>> (20 * si.dB).to_ratio(kind='power').value
100
>>> (20 * si.dB).to_ratio(kind='amplitude').value
10
>>> Quantity(100, '').to_log(si.dB, kind='power').value
20.0
>>> ((2 * Unit('dB/m')) * (10 * si.m)).to_ratio(kind='power').value
100

```

`.to('')` cannot erase a logarithmic difference's algebraic dimension. Ratio
conversion requires a pure logarithmic difference; compound dB/m and dB² need
to be combined into an appropriate difference before interpretation.

Negative physical inputs are invalid. Zero maps to negative infinity; negative
infinite readings restore zero physical value. NaNs and positive infinity
propagate. Integer powers of ten and the dBm/dBW 30 dB shift have exact paths;
other logarithms/exponentials may require approximate evaluation.

## Comparison and arrays

Exact comparisons retain symbolic factors and reference shifts. Coordinates
requiring approximate alignment must be converted explicitly before exact
comparison. `isclose()` uses `rel_tol=0` and an absolute Quantity difference.

Array readings use the same storage rules as affine points. Mean/median preserve
the logarithmic coordinate, standard deviation uses its difference unit, and
variance uses the squared difference unit. These are statistics of the readings,
not of the physical powers or amplitudes. Convert to physical Quantities first
when those physical statistics are intended. Writes stage conversion and dtype
validation before mutation; implicit array conversion rejects unit removal.
