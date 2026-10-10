import math
from decimal import Decimal
from fractions import Fraction

from ..exceptions import IncompleteFactorWarning, InexactFloatWarning

ZERO = Fraction(0)


def common_fraction(value: Fraction | Decimal | float, *,
                    float_max_denominator=1000000,
                    stacklevel=1) -> Fraction:
    '''
    Convert a number to a Fraction.

    - For `int` and `Decimal`, the conversion is always exact.
    - For `float`, the denominator is limited (under 1,000,000 by default).

    The conversion is useful for float results that are _expected_ to be
    rational numbers like 1/3 or 2/5, but may be inexact due to floating-point
    representation. However, such conversion may not always be as expected,
    especially for very large or small numbers like 1e30 or 1e-10.
    Therefore, when inexact conversion occurs, a warning is issued to inform
    the user of the potential inaccuracy.

    >>> Fraction(1.47)
    Fraction(6620291452234629, 4503599627370496)
    >>> common_fraction(1.47)  # InexactFloatWarning
    Fraction(147, 100)
    >>> common_fraction(1/3)  # InexactFloatWarning
    Fraction(1, 3)
    >>> common_fraction(1e30)  # InexactFloatWarning
    Fraction(1000000000000000019884624838656, 1)
    '''
    if isinstance(value, Fraction):
        return value
    frac = Fraction(value)
    if isinstance(value, float):
        frac_limited = frac.limit_denominator(float_max_denominator)
        if frac_limited != frac:
            import warnings
            warnings.warn(
                f'The float number {value!r} = {frac} is likely inexact, so '
                f'it was converted to the approximate Fraction {frac_limited}. '
                'For exact values, use Fraction or Decimal instead.',
                InexactFloatWarning, stacklevel=stacklevel+1)
        return frac_limited
    return frac


try:
    from sympy.ntheory import factorint, isprime  # type: ignore
except ModuleNotFoundError:
    import warnings
    warnings.warn('The `sympy` package is not installed. Factorization of large integers may be incomplete.', ImportWarning)
    def factorint(n: int, /, *, limit: int = 1000000) -> dict[int, int]:
        '''Internal method to factor an integer into its prime factors.'''
        factors = {}
        if n == 1:
            return factors
        for p in (2, 3):
            while n % p == 0:
                factors[p] = factors.get(p, 0) + 1
                n //= p
        d, step = 5, 2
        sqrtn = math.isqrt(n)
        while d <= sqrtn:
            if d > limit:
                import warnings
                msg = f'Integer {n} exceeded fallback factoring limit {limit}. ' \
                    'The remainder is stored as a single composite base. ' \
                    'Install `sympy` for complete prime factorization.'
                warnings.warn(msg, IncompleteFactorWarning)
                break
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
                sqrtn = math.isqrt(n)
            d += step
            step = 6 - step
        if n > 1:
            factors[n] = 1
        return factors


    SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    BASES_FOR_MILLER_RABIN = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)


    def isprime(n: int) -> bool:
        '''Internal method to check if an integer is prime.'''
        if n < 2:
            return False
        for p in SMALL_PRIMES:
            if n % p == 0:
                return n == p
        d, s = n - 1, 0
        while d & 1 == 0:
            s += 1
            d >>= 1
        for a in BASES_FOR_MILLER_RABIN:
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
        return True


def factorfrac(n: Fraction, /) -> dict[int, int]:
    '''Factor a Fraction into its prime factors.'''
    if n.denominator == 1:
        return factorint(n.numerator)
    return factorint(n.numerator) | {k: -v for k, v in factorint(n.denominator).items()}
