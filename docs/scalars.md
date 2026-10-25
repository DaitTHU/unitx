# Scalar numerical behavior

Exact unit structure and numerical magnitude evaluation have separate
contracts. Quantity handles units; the magnitude determines the numerical
domain used by arithmetic. Array calculations use the separate
[NumPy conversion rules](numpy.md#values-storage-and-precision).

```pycon
>>> from decimal import Decimal, localcontext
>>> from fractions import Fraction
>>> from unitx import Quantity, Unit
>>> from unitx.si import si

```

## Arithmetic and result types

Construction preserves the magnitude object and type. Multiplying or dividing
by a Unit only changes the unit product. Numerical operations apply the
magnitudes' operators in operand order, preserving their promotion, rounding,
signals and exceptions.

| Operation/domain | Behavior |
| --- | --- |
| int true division | Python float result |
| Fraction arithmetic | Exact rational result when the other operand supports it |
| Decimal arithmetic | Decimal result under the active context |
| Built-in Decimal with float in `+`, `-`, `*`, `/` | Convert float with `Decimal(repr(x))`, calculate in the active Decimal context, return float |
| Decimal with Fraction arithmetic | Native TypeError; choose a common domain explicitly |
| Unary `+` and `-` | Apply the native magnitude operator; keep the unit |
| `bool(q)` | Delegate to `bool(q.value)`; a zero magnitude is false regardless of unit |

```pycon
>>> (Quantity(Fraction(1), 'm') / Quantity(3, 's')).value
Fraction(1, 3)
>>> with localcontext() as context:
...     context.prec = 2
...     q = Quantity(Decimal('1.2345'), 'm')
...     print((+q).value, (-q).value, (q * 2).value)
1.2 -1.2 2.5
>>> bool(Quantity(0, 'm'))
False

```

Mixing a built-in float with a built-in Decimal magnitude explicitly opts the
calculation into a float result, including when one operand is a bare scalar.
This lets float formulas use `SI` constants without casting every constant:

```pycon
>>> from unitx.si import SI
>>> charge = 0.5 * SI.e
>>> charge.value
8.01088317e-20
>>> type(charge.value) is float
True
>>> (Quantity(1e-300, '') * Decimal('1e400')).value
1e+100

```

For example, `x * Quantity(d, unit)` evaluates its magnitude as
`float(Decimal(repr(x)) * d)`. Unit alignment for addition/subtraction also
runs in Decimal before the final float conversion. This is an **implicit
precision downgrade**: the Decimal context controls intermediate rounding,
signals and traps; the final float can round, overflow to infinity or
underflow to zero. A very small Decimal context can reduce accuracy further.
This rule does not convert the Decimal operand to float before calculating.
It applies to the four binary operators above and their augmented forms,
not powers, comparisons, hashes, custom scalar subclasses or bare Python
`float * Decimal` expressions. Pure Decimal/int arithmetic stays Decimal.
Choose all-Decimal inputs for calculations that must retain Decimal precision.

Custom scalar magnitudes are responsible for their own operator domain and
must defer Unit operands to reflected construction. Structural protocol
compatibility alone is not a promise that every third-party number works.
The supported real inputs of `unitx.math` are narrower: its Python scalar
backend returns float magnitudes, as described in [standard-library enhancements](stdlib.md).

## Conversion and addition

Rational unit scales keep int/Fraction magnitudes exact. Decimal conversion
combines the magnitude and rational scale exactly internally, then rounds
once to Decimal under the active context. A scale of one returns the original
magnitude without arithmetic or rounding. Irrational residual scales require
numerical approximation; Decimal conversion uses the float approximation's
shortest decimal representation.

Addition and subtraction first convert the right operand to the left
quantity's unit. When an int operand is aligned with a Decimal magnitude,
it is scaled in the Decimal domain. Mixed built-in float/Decimal operands
also align in Decimal and return float; explicit Fraction operands keep
their own domain. Conversion and arithmetic can each round; changing
the calculation's unit can therefore change rounding errors.

```pycon
>>> Quantity(1, 'cm').value_in('m')
Fraction(1, 100)
>>> Quantity(Decimal('0.1'), 'm').to('cm').value
Decimal('10')
>>> (Decimal('1') * si.m + 50 * si.cm).value
Decimal('1.5')
>>> (Quantity(1, 'km/m') + 1).value_in('')
Fraction(1001, 1)

```

`.value` always means the magnitude in the stored unit, including an extra
Unit multiplier. Plain numbers used with dimensionless quantities are
coherent, unscaled values; they do not inherit the quantity's stored scale.

## Exponents

Formal exponents share one rule across Monomial, Dimension, Factor, Unit and
Quantity. int/Fraction/Decimal exponents remain exact. Float exponents use
their shortest decimal text without a denominator limit: 0.1 means 1/10,
and 1e-10 remains nonzero. Use `Fraction(1, 3)` for exactly one third;
the float `1 / 3` instead denotes the finite decimal 0.3333333333333333.

Quantity passes the original exponent object to its magnitude, so numerical
powers still follow native Python rules. For example, an int raised to 2.0
returns float; Decimal magnitudes require int or Decimal exponents. Exponents
must be finite real scalars other than bool. Numerical powers may round or
produce complex values, even though the unit exponent remains rational.

```pycon
>>> (Unit('m') ** 0.1) ** 10 == Unit('m')
True
>>> Unit('m') ** Fraction(1, 3) == Unit('m^(1/3)')
True
>>> (Quantity(2, 'm') ** 2.0).value
4.0

```

This exponent rule is distinct from Factor's numeric scale input, whose
float rationalization uses a denominator limit.

## Equality, ordering and hashing

Equality has no tolerance. Finite floats retain their stored binary rational
values; they are not replaced by intended decimal fractions. With equal unit
factors, comparison uses the native magnitude operators, including independent
`!=`, Decimal signals and complex ordering errors.

With different factors, builtin real values and complex components are
aligned exactly for equality and hashing. Ordinary numeric subclasses follow
their base types' alignment. Cross-factor rational comparison does not perform
an extra native Decimal/float comparison merely to reproduce its signals.
Decimal/Fraction comparison is supported natively. Ordering cancels common
irrational factors before approximating any remaining irrational ratio.

```pycon
>>> Quantity(0.1, '') == Quantity(Fraction(1, 10), '')
False
>>> Quantity(1, 'm') == Quantity(100, 'cm')
True
>>> len({Quantity(1, 'm'), Quantity(100, 'cm')})
1

```

Equal scalar quantities have matching hashes when their magnitudes are
hashable. Quantities with any NumPy magnitude are unhashable. Array comparisons
use numerical broadcasting, not this exact scalar comparison path.

## Close comparisons

`q.isclose(other, rel_tol=1e-9, abs_tol=0)` uses the symmetric condition in
coherent SI units:

```text
abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
```

Finite rational coefficients remain exact; residual irrational unit factors
are approximated. Zero tolerances select exact equality. Tolerances must be
finite and nonnegative. Relative tolerance is dimensionless; a nonzero
absolute tolerance for dimensional quantities must carry matching units.
Plain operands and plain absolute tolerances are coherent dimensionless
values. Different dimensions raise DimensionError. NaNs are never close;
infinities are close only when they have the same sign.

```pycon
>>> Quantity(0.1 + 0.2, 'm').isclose(Quantity(0.3, 'm'))
True
>>> a, b = Quantity(1, 'm'), Quantity(100.05, 'cm')
>>> a.isclose(b, rel_tol=0, abs_tol=1 * si.mm)
True
>>> b.isclose(a, rel_tol=0, abs_tol=1 * si.mm)
True

```

NumPy `isclose` uses its own asymmetric formula and defaults; see
[array comparisons](numpy.md#close-comparisons). AffineQuantity point tolerances
have a [separate origin-independent contract](affine.md#comparison-tolerances-and-arrays).

## Symbolic factors and pi

`Factor` represents a positive multiplicative expression with rational
exponents. `SymbolicConstant` introduces an independent positive mathematical
constant under its documented construction assumptions; it does not infer
mathematical independence from numerical values. Known products, roots and
rational scales belong in Factor, rather than new constant identities.

```pycon
>>> import math as stdmath
>>> from unitx.factor import Factor, pi
>>> pi / pi == Factor(1)
True
>>> Quantity(180, 'deg') == Quantity(stdmath.pi, 'rad')
False
>>> Quantity(180, 'deg') == Quantity(1, pi * si.rad)
True
>>> Quantity(180, 'deg').isclose(Quantity(stdmath.pi, 'rad'))
True
>>> Factor(Fraction(1, 3))(Decimal('1.5'))
Fraction(1, 2)

```

The degree factor retains symbolic pi/180. Standard-library `math.pi` is a
finite float whose stored value is rational; it is not exact mathematical pi.
`unitx.pi.value` currently supplies the same float approximation for numerical
evaluation. Symbolic cancellation remains exact; arbitrary-precision numerical
evaluation is not provided.

Factor accepts positive int/Fraction/Decimal/float scale inputs. Fraction and
finite Decimal inputs remain exact. Numeric float scale inputs use denominator
limiting and can warn, or round a tiny positive value to zero and fail.
Factor evaluation returns int/Fraction when rational and float otherwise.
Calling a Factor on a finite Decimal magnitude converts that magnitude exactly
to Fraction, independently of Decimal context; this direct Factor API differs
from Quantity conversion, which retains Decimal.

The built-in integer factorization backend is bounded. Unresolved factors
emit `IncompleteFactorWarning` and may limit simplification and comparisons.
SymPy is an optional, more capable factorization backend:

```sh
python -m pip install sympy
```

[Documentation index](README.md) · [Units and quantities](units.md) ·
[Custom units](custom-units.md)
