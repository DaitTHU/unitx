import re
from fractions import Fraction

__all__ = ['superscript']

DIGIT = '0123456789+-=()'
SUPERSCRIPT = '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾'
SUBSCRIPT = '₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎'
SUP_TRANS = str.maketrans(SUPERSCRIPT, DIGIT)
SUB_TRANS = str.maketrans(SUBSCRIPT, DIGIT)


def superscript(ratio: int | Fraction, /, *, omit1=True) -> str:
    '''
    Convert a ratio to a string of superscript characters.
    Superscript ¹ is omitted by setting `omit1=True` (default).
    >>> superscript(2)
    '²'
    >>> superscript(-1)
    '⁻¹'
    >>> superscript(Fraction(3, 4))
    '³ᐟ⁴'
    '''
    if ratio < 0:
        return '⁻' + superscript(-ratio, omit1=False)
    if ratio.denominator == 1:
        if omit1 and ratio.numerator == 1:
            return ''
        return _sup(ratio.numerator)
    assert isinstance(ratio, Fraction)
    return _sup(ratio.numerator) + 'ᐟ' + _sup(ratio.denominator)


def _sup(number: int, /) -> str:
    return ''.join(SUPERSCRIPT[int(digit)] for digit in str(number))


def _sub(number: int, /) -> str:
    return ''.join(SUBSCRIPT[int(digit)] for digit in str(number))
