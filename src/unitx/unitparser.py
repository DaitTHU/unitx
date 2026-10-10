from __future__ import annotations

import re
from fractions import Fraction

from .exceptions import UnitSyntaxError
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

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.kind!r}, {self.value!r})'


EOF = Token('EOF', 'End of File')
'''Sentinel token to indicate the end of the input stream'''


class UnitParser:
    '''
    A recursive descent parser for physical unit expressions.

    This parser evaluates string representations of physical units and 
    converts them into `Monomial[SingleUnit]` objects. It supports explicit 
    and implicit multiplication, division, various forms of exponentiation, 
    grouping via parentheses, and leading division (e.g., `/m`).

    Grammar
    ---
    ```
    expr           := "/"? term ( ( "*" | "/" | <implicit_mul> ) term )*
    term           := factor exponent?
    factor         := UNIT | "1" | "(" expr ")"
    exponent       := <NUM_NO_SPACE> | ( "**" | "^" ) exponent_value
    exponent_value := NUM | ( "+(" | "-(" | "(" ) NUM ( "/" NUM )? ")"
    ```

    Lexical & Contextual Rules
    ---
    - `UNIT`           : Matches any valid unit string (supports non-ASCII chars).
    - `NUM`            : Matches integers and decimals. (Note: When used as a 
                         base factor, only '1' is permitted).
    - `<implicit_mul>` : A lookahead rule triggered when a space or no-space 
                         is followed by a `UNIT` or `"("`. (e.g., `m s` -> `m * s`).
    - `<NUM_NO_SPACE>` : A `NUM` token that immediately follows a `factor` without 
                         any intervening whitespace. (e.g., `m2` -> `m^2`).

    Supported Syntax Features & Examples:
    ---
    - Basic Units         : `m`, `s`, `kg`, `μm`
    - Multiplication      : `N * m` (explicit) or `N m` (implicit)
    - Division            : `m / s`
    - Leading Division    : `/s` or `/m^2` (Equivalent to `1 / s` and `1 / m^2`)
    - Exponentiation      : `m^2`, `m**2`, `m2` (implicit, no space)
    - Fractional Exponents: `m^(1/2)`, `s^-(2/3)`
    - Grouping            : `(m/s)^2`, `J / (kg K)`

    Raises
    ---
    UnitSyntaxError: If the expression contains unclosed parentheses, invalid 
                     characters, redundant suffixes, or invalid numeric factors.
    UnitSymbolError: If the unit symbol is invalid.
    '''

    def __init__(self, symbol: str):
        self.tokens = map(Token.from_match, _TOKEN_RE.finditer(symbol))
        self.token = next(self.tokens, EOF)

    @property
    def current_token(self) -> Token:
        while self.token is not EOF and self.token.kind == 'SPACE':
            self.token = next(self.tokens, EOF)
        return self.token

    def match(self, kind: str, skip_space=True) -> Token | None:
        token = self.current_token if skip_space else self.token
        if token.kind == kind:  # match
            self.token = next(self.tokens, EOF)  # move to the next token
            return token
        return None  # fail to match

    def parse(self) -> Monomial[SingleUnit]:
        if self.current_token is EOF:
            return Monomial()
        result = self.parse_expr()
        if self.current_token is not EOF:
            raise UnitSyntaxError(f'Unparsed redundant suffix {self.token.value!r}.')
        return result

    def parse_expr(self) -> Monomial[SingleUnit]:
        '''Handle consecutive multiplication or division'''
        # Implicit per-grammar
        result = 1 / self.parse_term() if self.match('DIV') else self.parse_term()
        while True:
            if self.match('MUL'):
                result *= self.parse_term()
            elif self.match('DIV'):
                result /= self.parse_term()
            # Implicit multiplication (space or adjacent), e.g., m s or m(s)
            elif self.current_token.kind in ('UNIT', 'LPAREN'):
                result *= self.parse_term()
            else:
                break
        return result

    def parse_term(self) -> Monomial[SingleUnit]:
        '''Handle a single factor and its exponentiation'''
        factor_comp = self.parse_factor()
        # Implicit exponentiation check (like m2 or m-2)
        if token := self.match('NUM', skip_space=False):
            factor_comp **= Fraction(token.value)
        # Explicit exponentiation check (like m^2 or m**2)
        elif self.match('POW'):
            factor_comp **= self.parse_exponent_value()
        return factor_comp

    def parse_factor(self) -> Monomial[SingleUnit]:
        '''Handle a single unit, number, or parenthesized expression'''
        if token := self.match('UNIT'):
            return Monomial({SingleUnit(token.value): 1})
        elif token := self.match('NUM'):
            if token.value == '1':
                return Monomial()
            raise UnitSyntaxError(f'Invalid number {token.value!r}, only 1 is allowed as a numeric factor.')
        elif self.match('LPAREN'):
            res = self.parse_expr()
            if not self.match('RPAREN'):
                raise UnitSyntaxError('Unclosed parenthesis.')
            return res
        raise UnitSyntaxError(f"Expected unit or '(', got {self.token.value!r}.")

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
            raise UnitSyntaxError(f'Invalid exponent value {self.token.value!r}.')
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
        raise UnitSyntaxError(f"Expected '/' or ')' in fraction exponent, got {self.token.value!r}.")
