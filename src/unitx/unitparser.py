from __future__ import annotations

import re
from fractions import Fraction

from .monomial import Monomial
from .singleunit import SingleUnit


_TOKEN_RE = re.compile('|'.join(f'(?P<{kind}>{pattern})' for kind, pattern in {
    'POW': r'\*\*|\^',
    'MUL': r'\*',
    'DIV': r'/',
    'NUM': r'[-+]?\d+(?:\.\d+)?',  # supports integers and decimals
    'UNIT': r'[^\s\d\*\^\(\)\/\-\+]+',  # supports non-ascii
    'LPAREN': r'\(',
    'RPAREN': r'\)',
    'SIGNEDLPAREN': r'[-+]\(',  # supports leading sign for parenthesized exponent, e.g., m^-(2/3)
    'SPACE': r'[ \t]+',
    'MISMATCH': r'.',
}.items()))


class Token:
    __slots__ = ('kind', 'value')

    def __init__(self, kind: str | None, value: str) -> None:
        self.kind, self.value = kind, value
        if self.kind == 'MISMATCH':
            raise UnitSyntaxError(f'Unmatched character {self.value!r}')

    @classmethod
    def from_match(cls, match: re.Match[str]) -> Token:
        return cls(match.lastgroup, match.group())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Token):
            return NotImplemented
        return self.kind == other.kind and self.value == other.value

    def __repr__(self) -> str: return f'Token({self.kind!r}, {self.value!r})'


EOF = Token('EOF', 'End of File')
'''Sentinel token to indicate the end of the input stream'''


class UnitParser:
    def __init__(self, symbol: str):
        self.tokens = [Token.from_match(m) for m in _TOKEN_RE.finditer(symbol)]
        self.pos = 0

    @property
    def current_token(self) -> Token:
        while self.pos < len(self.tokens) and self.tokens[self.pos].kind == 'SPACE':
            self.pos += 1
        return self.tokens[self.pos] if self.pos < len(self.tokens) else EOF

    def match(self, kind: str, skip_space=True) -> Token | None:
        p = self.pos
        if skip_space:
            while p < len(self.tokens) and self.tokens[p].kind == 'SPACE':
                p += 1
        token = self.tokens[p] if p < len(self.tokens) else EOF
        if token.kind == kind:
            self.pos = p + 1  # to the next token
            return token
        return None  # fail to match

    def parse(self) -> Monomial[SingleUnit]:
        if self.current_token is EOF:
            return Monomial()
        result = self.parse_expr()
        if (token := self.current_token) is not EOF:
            raise UnitSyntaxError(f'Unparsed redundant suffix {token.value!r}.')
        return result

    def parse_expr(self) -> Monomial[SingleUnit]:
        '''Handle consecutive multiplication or division'''
        result = self.parse_term()
        while True:
            if self.match('MUL'):
                result *= self.parse_term()
            elif self.match('DIV'):
                result /= self.parse_term()
            # Implicit multiplication (space or adjacent), e.g., m s or m(s)
            elif self.current_token.kind in {'UNIT', 'LPAREN'}:
                result *= self.parse_term()
            else:
                break
        return result

    def parse_term(self) -> Monomial[SingleUnit]:
        '''Handle a single factor and its exponentiation'''
        factor_comp = self.parse_factor()
        # Explicit exponentiation check (like m^2 or m**2)
        if self.match('POW'):
            factor_comp **= self.parse_exponent_value()
        # Implicit exponentiation check (like m2 or m-2)
        elif token := self.match('NUM', skip_space=False):
            factor_comp **= Fraction(token.value)
        return factor_comp

    def parse_factor(self) -> Monomial[SingleUnit]:
        '''Handle a single unit, number, or parenthesized expression'''
        if token := self.match('UNIT'):
            return Monomial({SingleUnit(token.value): 1})
        elif token := self.match('NUM'):
            if token.value == '1':
                return Monomial()
            raise UnitSyntaxError(f"Invalid number {token.value!r}, only 1 is allowed as a numeric factor.")
        elif self.match('LPAREN'):
            res = self.parse_expr()
            if not self.match('RPAREN'):
                raise UnitSyntaxError('Unclosed parenthesis.')
            return res
        raise UnitSyntaxError(f"Expected unit or '(', got {self.current_token.value!r}.")

    def parse_exponent_value(self) -> Fraction:
        '''Parse an exponent value, which can be a number or a fraction in parentheses'''
        # digit number exponent
        if token := self.match('NUM'):
            return Fraction(token.value)
        # parenthesized exponent, possibly with a leading sign
        sign = 1
        if token := self.match('SIGNEDLPAREN'):
            sign = -1 if token.value.startswith('-') else 1
        elif not self.match('LPAREN'):
            raise UnitSyntaxError(f'Invalid exponent value {self.current_token.value!r}.')
        if not (numerator := self.match('NUM')):
            raise UnitSyntaxError('Fraction exponent missing numerator.')
        if self.match('DIV'):
            if not (denominator := self.match('NUM')):
                raise UnitSyntaxError('Fraction exponent missing denominator.')
            if not self.match('RPAREN'):
                raise UnitSyntaxError('Fraction exponent parenthesis not closed.')
            return sign * Fraction(numerator.value) / Fraction(denominator.value)
        elif self.match('RPAREN'):
            return sign * Fraction(numerator.value)
        raise UnitSyntaxError(f"Expected '/' or ')' in fraction exponent, got {self.current_token.value!r}.")


class UnitSyntaxError(ValueError):
    pass

