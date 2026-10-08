from fractions import Fraction

__all__ = ['superscript']

DIGIT = '0123456789+-=()'
SUPERSCRIPT = '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾'
SUBSCRIPT = '₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎'
DIGIT_TO_SUP = str.maketrans(DIGIT, SUPERSCRIPT)
DIGIT_TO_SUB = str.maketrans(DIGIT, SUBSCRIPT)
SUP_TO_DIGIT = str.maketrans(SUPERSCRIPT, DIGIT)
SUB_TO_DIGIT = str.maketrans(SUBSCRIPT, DIGIT)
DOT = '⋅'  # chr(0x22C5)


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
    return str.translate(str(number), DIGIT_TO_SUP)


def _sub(number: int, /) -> str:
    return str.translate(str(number), DIGIT_TO_SUB)
