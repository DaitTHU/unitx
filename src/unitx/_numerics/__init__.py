"""Scalar protocols and numerical helpers, retaining the unitx._numerics entry.

Scaling, conversion and optional NumPy support live in sibling submodules;
this entry imports none of them.

- `as_fraction`: Converts a number to a rational number (Fraction), handling
  inexact floating-point representations.
- `isprime`: Tests the primality of a number.
- `factorint`: Factors a positive integer into its prime factors,
"""

from __future__ import annotations

import math
import warnings
from decimal import Decimal
from fractions import Fraction
from typing import Any, Never, Protocol, Self, TypeAlias

from ..exceptions import _WARNING_SKIP_PREFIXES, IncompleteFactorWarning, InexactFloatWarning

__all__ = [
    'Fractionable',
    'Scalar',
    'Array',
    'Magnitude',
    'Real',
    'Number',
    'as_fraction',
    'as_int_if_integral',
    'factorint',
    'isprime',
]

# Number typing


class Scalar(Protocol):
    """Nonrecursive capability protocol for scalar magnitudes.

    Operand types depend on the concrete magnitude's numerical domain. Arithmetic
    results use object here to avoid recursive structural checks against NumPy
    overloads. Public Quantity overloads describe result types; custom magnitudes
    remain responsible for returning valid scalar results. Equality returns bool.
    Ordering is deliberately not required: valid arithmetic, including fractional powers,
    can produce complex values. NumPy magnitudes use a separate Array
    typing boundary and optional adapter, keeping scalar equality separate.

    Forward binary arithmetic has an unspecified operand domain at
    this protocol boundary. `Never` lets concrete numeric methods satisfy the
    protocol without claiming that they accept every operand. In particular,
    raw magnitudes must defer Unit operands to Unit's reflected multiplication
    and division, which construct quantities and preserve the magnitude's type.
    They must also defer Quantity operands so that Quantity can handle units.
    """

    def __pos__(self, /) -> object: ...
    def __neg__(self, /) -> object: ...
    def __add__(self, other: Never, /) -> object: ...
    def __sub__(self, other: Never, /) -> object: ...
    def __mul__(self, other: Never, /) -> object: ...
    def __truediv__(self, other: Never, /) -> object: ...
    def __pow__(self, other: Any, /) -> object: ...
    def __radd__(self, other: Any, /) -> object: ...
    def __rsub__(self, other: Any, /) -> object: ...
    def __rmul__(self, other: Any, /) -> object: ...
    def __rtruediv__(self, other: Any, /) -> object: ...
    def __eq__(self, other: Any, /) -> bool: ...


class Array(Protocol):
    """Array magnitude protocol, also satisfied by NumPy numeric scalars.

    The conversion hook intentionally avoids a mandatory NumPy type import;
    metadata remains typed and the adapter validates concrete runtime types.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def ndim(self) -> int: ...

    @property
    def size(self) -> int: ...

    @property
    def dtype(self) -> object: ...

    @property
    def __array_interface__(self) -> object: ...

    def __array__(self, /) -> Any: ...


# Check array metadata before the scalar arithmetic capability protocol.
Magnitude: TypeAlias = Array | Scalar
Real: TypeAlias = int | Fraction | float
Number: TypeAlias = Real | complex
Fractionable: TypeAlias = Real | Decimal


def is_numpy_object(value: object) -> bool:
    """Recognize NumPy objects and subclasses without importing NumPy."""
    for base in type(value).__mro__:
        module = base.__module__
        if module == 'numpy' or module.startswith('numpy.'):
            return True
    return False


def as_fraction(value: Fractionable, /, *, float_max_denominator: int = 1000000) -> Fraction:
    """Convert a number to a Fraction.

    - For `int` and `Decimal`, the conversion is always exact.
    - For `float`, the denominator is limited (under 1,000,000 by default).

    The conversion is useful for float results that are _expected_ to be
    rational numbers like 1/3 or 2/5, but may be inexact due to floating-point
    representation.

    However, such conversion may not always be as expected, especially for
    very large or small numbers like 1e30 or 1e-10. Therefore, when inexact
    conversion occurs, an `InexactFloatWarning` is issued to inform the user
    of the potential inaccuracy.

    >>> Fraction(1.47)  # exact but not expected result
    Fraction(6620291452234629, 4503599627370496)
    >>> as_fraction(1.47)  # InexactFloatWarning
    Fraction(147, 100)
    >>> as_fraction(1 / 3)  # InexactFloatWarning
    Fraction(1, 3)
    >>> as_fraction(1e-10)  # InexactFloatWarning
    Fraction(0, 1)
    >>> as_fraction(1e-10, float_max_denominator=10**12)  # InexactFloatWarning
    Fraction(1, 10000000000)
    >>> as_fraction(1e30)  # no warning, but not expected result
    Fraction(1000000000000000019884624838656, 1)
    """
    if type(value) is Fraction:
        return value
    if not isinstance(value, Fractionable) or type(value) is bool:
        error_message = f'Cannot convert {type(value).__name__} to Fraction.'
        raise TypeError(error_message)
    fraction = Fraction(value)
    if isinstance(value, float):
        limited_fraction = fraction.limit_denominator(float_max_denominator)
        if limited_fraction != fraction:
            warnings.warn(
                f'The denominator of the float number {value!r} = {fraction} is '
                'too big and may be not expected, so it was converted to '
                f'the approximate Fraction {limited_fraction}. '
                'The conversion is inexact and may not be as expected either. '
                'For exact values, use Fraction or Decimal instead.',
                InexactFloatWarning,
                skip_file_prefixes=_WARNING_SKIP_PREFIXES,
            )
        return limited_fraction
    return fraction


def as_int_if_integral(value: Fraction, /) -> int | Fraction:
    """Return an int if integral; otherwise return the original Fraction."""
    return value.numerator if value.denominator == 1 else value


def as_exponent(value: Fractionable, /) -> Fraction:
    """Normalize formal exponents; floats use their shortest decimal text."""
    if type(value) is Fraction:
        return value
    if not isinstance(value, Fractionable) or isinstance(value, bool):
        error_message = f'Cannot convert {type(value).__name__} to Fraction.'
        raise TypeError(error_message)
    # Nonfinite floats still use Fraction's standard ValueError/OverflowError.
    return Fraction(
        str(float(value)) if isinstance(value, float) and math.isfinite(value) else value
    )


# Number theory functions: prime factorization and primality testing.
# These are implemented here to avoid a dependency on `sympy`.
# However, the implementation is limited and may not be as efficient or
# complete as `sympy`. Therefore, a warning is issued when the limit is exceeded.

_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
_MILLER_RABIN_BASES = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
_PRIME_TEST_LIMIT = 2**64


def isprime(n: int) -> bool:
    """Test primality; results are definitive only for n < 2**64."""
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d & 1 == 0:
        s += 1
        d >>= 1
    for a in _MILLER_RABIN_BASES:
        a %= n
        if a == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    if n >= _PRIME_TEST_LIMIT:
        warnings.warn(
            f'Primality test for {n} is not guaranteed to be definitive. '
            'Install `sympy` for complete primality testing.',
            IncompleteFactorWarning,
            skip_file_prefixes=_WARNING_SKIP_PREFIXES,
        )
    return True


def factorint(n: int, /, *, limit: int = 1000000) -> dict[int, int]:
    """Factor a positive integer, with bounded trial division.

    Primes below 2**64 and perfect powers are handled independently of
    the trial limit. Any unresolved remainder is retained with a warning.
    >>> factorint(1)
    {}
    >>> factorint(60)
    {2: 2, 3: 1, 5: 1}
    >>> factorint(2**(2**5) + 1)
    {641: 1, 6700417: 1}
    """
    if n < 1:
        error_message = 'factorint requires a positive integer.'
        raise ValueError(error_message)
    factors = _factorint(n, limit)
    for base in factors:
        if isinstance(base, UnfactoredInteger):
            warnings.warn(
                f'Integer {base} exceeded fallback factoring limit {limit}. '
                'The remainder is stored as a single unresolved base. '
                'Install `sympy` for complete prime factorization.',
                IncompleteFactorWarning,
                skip_file_prefixes=_WARNING_SKIP_PREFIXES,
            )
    return factors


class UnfactoredInteger(int):
    """A subclass of int that represents an integer that has not been factored.

    This class is used to indicate that the integer is a remainder that could
    not be factored within the specified limit in the `factorint` function.
    It is a subclass of int, so it behaves like a regular integer in most
    contexts, but it can be distinguished from regular integers when needed.
    """

    limit: int

    def __new__(cls, value: int, /, limit: int) -> Self:
        obj = super().__new__(cls, value)
        obj.limit = limit
        return obj

    def __repr__(self) -> str:
        return f'{type(self).__name__}({int(self)}, limit={self.limit})'

    def __str__(self) -> str:
        return str(int(self))

    def __reduce__(self) -> tuple[type[Self], tuple[int, int]]:
        return type(self), (int(self), self.limit)


def _factorint(n: int, limit: int) -> dict[int, int]:
    """Internal algorithm for factorint."""
    factors: dict[int, int] = {}
    if n == 1:
        return factors
    for p in (2, 3):
        exponent = 0
        while n % p == 0:
            exponent += 1
            n //= p
        if exponent:
            factors[p] = exponent
    divisor, step = 5, 2
    while n > 1:
        # These Miller-Rabin bases are deterministic only below 2**64.
        if n < _PRIME_TEST_LIMIT and isprime(n):
            factors[n] = 1
            return factors
        # Prime degrees suffice: a composite-degree power is also a power of
        # a prime degree. Recursion extracts any remaining power of the root.
        root = math.isqrt(n)
        for degree in range(2, n.bit_length()):
            if not isprime(degree):
                continue
            power_root = root if degree == 2 else _integer_root(n, degree)
            if power_root**degree == n:
                for base, exponent in _factorint(power_root, limit).items():
                    factors[base] = factors.get(base, 0) + degree * exponent
                return factors
        while divisor <= root:
            if divisor > limit:
                factors[UnfactoredInteger(n, limit)] = 1
                return factors
            if n % divisor == 0:
                exponent = 0
                while n % divisor == 0:
                    exponent += 1
                    n //= divisor
                factors[divisor] = exponent
                # Recheck the reduced remainder before continuing trial division.
                break
            divisor += step
            step = 6 - step  # alternate between 2 and 4 to check 6k ± 1
        else:
            factors[n] = 1
            return factors
    return factors


def _integer_root(value: int, degree: int) -> int:
    """Return the floor of the `degree`-th root of a positive integer.

    Internal helper. Only intended for calls within this module where
    `degree >= 3` is guaranteed by the caller; no validation is performed
    on `value` or `degree`.
    """
    if value == 1:
        return value
    if degree >= value.bit_length():
        return 1
    lo = 1 << ((value.bit_length() - 1) // degree)
    hi = 1 << ((value.bit_length() + degree - 1) // degree)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if mid**degree <= value:
            lo = mid
        else:
            hi = mid
    return lo
