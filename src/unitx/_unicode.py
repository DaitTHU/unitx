"""Unicode utilities for unitx."""

from fractions import Fraction
from typing import Literal

__all__ = ['DOT', 'DotType', 'superscript']

DOT = '⋅'  # chr(0x22C5)
type DotType = Literal['⋅']

DIGITS = '0123456789+-'
SUPERSCRIPTS = '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻'
DIGIT_TO_SUP = str.maketrans(DIGITS, SUPERSCRIPTS)
SUP_TO_DIGIT = str.maketrans(SUPERSCRIPTS, DIGITS)
SUPERSCRIPT_SLASH = 'ᐟ'  # Canadian Syllabics Final Acute, chr(0x141F)


def superscript(ratio: int | Fraction, /, *, omit_identity: bool = True) -> str:
    """Convert a rational number to a string of superscript characters.

    - Identity `a¹ = a` is omitted by default (`omit_identity=True`).
    - As there is no superscript division in unicode, so the fraction slash
      is borrowed from Canadian Syllabics Final Acute (`ᐟ`, `chr(0x141F)`).

    >>> superscript(-2)
    '⁻²'
    >>> superscript(1, omit_identity=False)
    '¹'
    >>> superscript(Fraction(3, 4))
    '³ᐟ⁴'
    """
    if not isinstance(ratio, (int, Fraction)) or isinstance(ratio, bool):
        error_message = f'superscript() expected int | Fraction, got {type(ratio).__name__}.'
        raise TypeError(error_message)
    if ratio.denominator == 1:
        if omit_identity and ratio.numerator == 1:
            return ''
        return _sup(ratio.numerator)
    return _sup(ratio.numerator) + SUPERSCRIPT_SLASH + _sup(ratio.denominator)


def _sup(number: int, /) -> str:
    return str.translate(str(number), DIGIT_TO_SUP)
