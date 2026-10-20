# Scalar numerical behavior

```pycon
>>> from unitx import Quantity, Unit
>>> from unitx.si import SI, si

```


Exact unit algebra and numerical magnitude evaluation have different rules:

- Formal exponents use one rule in Monomial, Dimension, Factor, Unit, and
  Quantity: int/Fraction/Decimal values stay exact; floats use their shortest
  decimal text without a denominator limit. Thus `0.1` means `1/10` and
  `1e-10` remains nonzero. Use `Fraction(1, 3)` for exactly one third:
  the float `1 / 3` means the decimal `0.3333333333333333` in formal exponents.
  Factor's numeric scale inputs retain their separate denominator-limit rules.
- Quantity handles units; the magnitudes determine numerical arithmetic.
  Operators use the magnitudes in operand order and preserve native promotion,
  rounding, signals, and exceptions. Decimal arithmetic returns Decimal and
  follows its active context. Mixing Decimal with float or Fraction raises
  `TypeError`, as it does for the bare numbers; choose a common numerical type
  explicitly when needed.
- Conversion with a rational scale keeps int/Fraction values exact. Decimal
  conversion combines the value and scale exactly internally, then rounds once
  to Decimal in the active context. Irrational unit scales are approximated as
  floats, using their shortest decimal representations for Decimal conversion.
  A scale of one preserves the magnitude without arithmetic or rounding.
  Direct `Factor(value)` retains its separate exact Fraction evaluation rules.
- Addition and subtraction align the right operand with the left quantity's
  unit before numerical arithmetic. Integers being aligned with a Decimal
  magnitude are scaled in the Decimal domain, so `Decimal('1') * si.m + 50 *
  si.cm` returns a Decimal magnitude of `1.5`. Explicit Fraction/float operands
  keep their own numerical domains. Finite-precision conversion and arithmetic
  can round; changing the result unit can therefore change rounding errors.
- Unary `+` and `-` apply the magnitude's native operators and preserve the unit.
  Decimal results remain Decimal and follow its context precision, rounding,
  signals, and exceptions; unary `+` can therefore change the numerical value.
- Integer division produces a float; Fraction arithmetic remains rational.
  Use Fraction magnitudes explicitly for exact rational results. Powers pass
  the original exponent to the magnitude: `2 ** 2.0` returns float and
  `2 ** Decimal('0.5')` returns Decimal. Decimal magnitudes require int or
  Decimal exponents, as in Python. The unit follows the common formal exponent
  rule above; numerical evaluation still uses the original exponent object.
  Exponents must be finite real scalars other than bool. Numerical powers can
  round or produce complex values, following the magnitude's rules.
- Truth values follow the magnitude, so zero quantities are false regardless
  of unit. Comparisons with equal unit factors use native magnitude operators,
  including independent `!=`, Decimal signals/exceptions and complex ordering
  errors. For different unit factors, builtin real values and complex
  components are aligned exactly to avoid rounding-dependent equality or
  hashes. Ordinary numeric subclasses use the same alignment as their base
  types. Decimal/float signals apply to native comparisons with equal factors;
  cross-factor alignment compares exact rational coefficients and does not
  perform an additional comparison merely to reproduce signals. Decimal/Fraction
  comparisons are supported natively.
- Equality is exact, without a tolerance. Finite floats are compared as their
  stored binary rational values. Floating-point conversions and irrational
  powers may round; use `isclose()` for approximate real scalar comparisons.
  It uses the symmetric condition
  `abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)` in coherent SI
  units, with defaults `rel_tol=1e-9` and `abs_tol=0`. Swapping quantities
  preserves the result. Nonzero absolute tolerances for dimensional quantities
  must carry matching units; plain operands and plain absolute tolerances
  represent unscaled dimensionless values. Tolerances must be finite and
  nonnegative. Different dimensions raise `DimensionError`. NaNs are never
  close, and infinities are close only to infinities of the same sign.
  Finite rational coefficients stay exact; residual irrational unit factors
  are approximated. With both tolerances zero, `isclose()` uses exact equality.

```pycon
>>> from decimal import Decimal, localcontext
>>> from fractions import Fraction
>>> with localcontext() as context:
...     context.prec = 2
...     q = Quantity(Decimal('1.2345'), 'm')
...     print((+q).value, (-q).value, (q * 2).value)
1.2 -1.2 2.5
>>> Quantity(Decimal('0.1'), 'm').to('cm').value
Decimal('10')
>>> (Quantity(Fraction(1), 'm') / Quantity(3, 's')).value
Fraction(1, 3)
>>> (Quantity(2, 'm') ** 2.0).value
4.0
>>> bool(Quantity(0, 'm'))
False
>>> Quantity(0.1, '') == Quantity(Fraction(1, 10), '')
False
>>> Quantity(0.1 + 0.2, 'm').isclose(Quantity(0.3, 'm'))
True
>>> a, b = Quantity(1, 'm'), Quantity(100.05, 'cm')
>>> a.isclose(b, rel_tol=0, abs_tol=Quantity(1, 'mm'))
True
>>> b.isclose(a, rel_tol=0, abs_tol=Quantity(1, 'mm'))
True

```

The built-in integer factorization backend is bounded. Unresolved factors emit
`IncompleteFactorWarning` and can limit simplification and comparisons. SymPy
is an optional, more capable factorization backend:

```sh
python -m pip install sympy
```

