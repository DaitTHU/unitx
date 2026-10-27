# Explicit unit representations

Representation changes are explicit. Arithmetic keeps the existing formal
unit rules; `.value` continues to read the stored magnitude. These operations
return new quantities and use the existing numerical conversion rules, which
can evaluate irrational scales or change the magnitude's concrete type.

## SI base units

`q.to_base()` converts a linear Quantity to a scale-one product of the seven
coherent SI base units: s, m, kg, A, K, mol and cd. Arbitrary supported rational
powers are retained. Dimensionless results use `one`, including angles.

```pycon
>>> from decimal import Decimal
>>> from fractions import Fraction
>>> from unitx import Quantity, Unit
>>> from unitx.si import si
>>> (3 * si.cm).to_base()
Quantity(Fraction(3, 100), Unit('m'))
>>> (1 * si.J).to_base().unit == si.kg * si.m**2 / si.s**2
True
>>> (1 * si.km / si.m).to_base()
Quantity(1000, Unit(''))
>>> (20 * si.degC).to_base() == 20 * si.K
True

```

`to_base()` does not search named derived units, guess physical kinds, or
perform electromagnetic SI/CGS correspondences. It rejects nonzero logarithmic
degrees such as dB, dB/m and dB². Ratios with cancelling logarithmic degrees,
such as dB/Np, can convert normally. Use the explicit logarithmic mapping before
requesting base units when a power/amplitude interpretation is required.

This method is for linear Quantity objects, including temperature differences.
Affine and logarithmic points keep their existing explicit coordinate APIs:
`(20 @ si.degC).to(si.K)` stays an affine point, while
`(30 @ si.dBm).to_quantity(si.W)` maps to a physical linear quantity.

Array results keep one unit for the whole magnitude. An identity conversion
can share the original storage; nonidentity conversion follows the existing
dtype and allocation rules. Core scalar use does not import NumPy.

## Combining like units

`q.simplify()` groups single-unit factors by their complete algebraic dimension
and combines their rational exponents. Each group uses an existing unit:
unprefixed units come first, then canonical symbols in lexical order. Independent
definitions sharing a symbol use their stable definition identity as a final
tie-breaker. Thus the choice does not depend on multiplication order.

```pycon
>>> Quantity(1, 'cm*m**2').simplify()
Quantity(Fraction(1, 100), Unit('m**3'))
>>> Quantity(1, 'km/cm').simplify()
Quantity(100000, Unit(''))
>>> (25 * si.percent).simplify()
Quantity(Fraction(1, 4), Unit(''))
>>> (1 * si.N * si.m).simplify().unit == si.N * si.m
True

```

Ordinary dimensionless factors such as rad, percent or a dimensionless custom
unit disappear from the formal product, evaluating their scale as part of the
conversion. Nonzero logarithmic degrees remain distinct: dB cannot merge with
ordinary dimensionless units, but dB and Np can combine as like factors.

An existing extra exact `Unit.scale` stays in the target unit. This preserves
symbolic factors unrelated to the merging; `q.n` or `q.to_base()` can explicitly
evaluate that scale when their input requirements are met. Fractions and
irrational factors otherwise follow ordinary conversion rules.

This operation does not search for a preferred named unit: N·m stays N·m,
and a lone prefixed dimensional unit stays as it is. Unit identity and physical
kind remain separate; grouping does not validate a physical interpretation.
The method belongs to Quantity. Unit objects remain immutable inputs; the
selection implementation is internal rather than another public conversion API.

## Readable prefixes

`q.compact()` chooses an SI prefix for a real quantity with one prefixable
unit factor. Powers, including negative and rational powers, are supported:
the prefix factor is raised to that power too.

```pycon
>>> Quantity(Fraction(1, 10**9), si.m).compact()
Quantity(Fraction(1, 1), Unit('nm'))
>>> Quantity(Fraction(1, 10**9), si.m**3).compact()
Quantity(Fraction(1, 1), Unit('mm**3'))
>>> (1000 * si.W).compact()
Quantity(Fraction(1, 1), Unit('kW'))

```

The chosen finite peak magnitude preferably lies in `[1, 1000)`. Among fitting
candidates, prefixes with exponents divisible by three come first, followed by
the smallest resulting peak. Other SI prefixes can fill gaps for powered units.
If none fits, choose the closest multiplicative distance to that interval:
`1/peak` below one, or `peak/1000` at or above 1000. Ties use the same engineering
preference, then the smaller peak and canonical prefix symbol.

Arrays use their largest absolute **finite** element, avoiding signed-integer
absolute-value overflow; NaN/Inf elements do not choose the prefix. The whole
array retains one unit. Zeros, empty/nonfinite-only inputs, nonprefixable units,
and expressions with multiple different factors keep their existing units.
Complex magnitudes and unsupported custom scalar domains raise TypeError.

The unit's prefix permissions remain authoritative, including spellings where
only a name or alias is prefixable. Gram is the prefix root for mass, so this
does not create spellings such as mkg. Additional exact Unit scales participate
in conversion to the selected prefix. This is a numerical representation
choice, with the same rounding, dtype and overflow limits as ordinary conversion;
it does not change unit identities or impose float on Decimal/Fraction values.

## Value and unit pairs

`q.as_pair()` returns the stored `(q.value, q.unit)` without conversion or copy.
The second item is the actual complete Unit, AffineUnit or LogUnit object,
including exact scales or coordinate references. Choose another representation
with `.to(target)` before obtaining the pair if needed.

```pycon
>>> original = Quantity(Decimal('1.25'), si.cm)
>>> value, unit = original.as_pair()
>>> value is original.value and unit is original.unit
True
>>> Quantity(value, unit) == original
True
>>> reading, coordinate = (20 @ si.degC).as_pair()
>>> from unitx import AffineQuantity
>>> AffineQuantity(reading, coordinate) == 20 @ si.degC
True

```

Reconstruct points with their matching class, AffineQuantity or LogQuantity:
a two-item pair does not encode the quantity family. This is an in-process
accessor, not a serialization format. Array magnitudes retain shared writable
storage just as `.value` does; obtaining a pair does not copy or protect it.
Quantity iteration is unchanged: arrays yield individual quantities and
scalar iteration raises TypeError.

[Units and quantities](units.md) · [Scalar numerical rules](scalars.md)
