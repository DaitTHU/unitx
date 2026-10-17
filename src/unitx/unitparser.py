"""Parse physical unit expressions into formal products of single units."""

from __future__ import annotations

import re
from collections.abc import Iterator
from fractions import Fraction

from .exceptions import UnitSymbolError, UnitSyntaxError
from .monomial import Monomial
from .singleunit import SingleUnit

_PATTERNS = {
    'POW': r'\*\*|\^',
    'MUL': r'\*',
    'DIV': r'/',
    'NUM': r'[-+]?\d+(?:\.\d+)?',  # supports integers and decimals
    'LPAREN': r'\(',
    'RPAREN': r'\)',
    'SIGN': r'[-+]',  # leading sign, e.g., m^-(2/3)
    'SPACE': r'\s+',
    'UNIT': r'[^\s\d' + re.escape('*^()/+-') + ']+',  # supports non-ascii
    'UNMATCHED': r'.',  # any single character not matched by other patterns
}

_TOKEN_RE = re.compile('|'.join(f'(?P<{kind}>{pattern})' for kind, pattern in _PATTERNS.items()))


class Token:
    """A lexical token identified by its kind and matched text.

    Parameters
    ----------
    kind : str or None
        Name of the matched regular expression group. ``None`` represents a
        match without a named group.
    value : str
        Matched text, or a description for a sentinel such as ``EOF``.

    Notes
    -----
    Equality compares both kind and value. Unit symbol validation belongs to
    ``SingleUnit`` and occurs during parsing, rather than token construction.
    """

    __slots__ = ('kind', 'value')

    def __init__(self, kind: str | None, value: str) -> None:
        self.kind, self.value = kind, value

    @classmethod
    def from_match(cls, match: re.Match[str]) -> Token:
        """Build a token from the match's last named group and matched text."""
        return cls(match.lastgroup, match.group())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Token):
            return NotImplemented
        return self.kind == other.kind and self.value == other.value

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.kind!r}, {self.value!r})'


EOF = Token('EOF', 'End of File')
"""Sentinel marking the end of the token stream."""


class UnitParser:
    """Parse a physical unit expression into an immutable formal product.

    Support multiplication, division, rational powers, and parentheses.
    Unit symbols, names, prefixes, and aliases are resolved by ``SingleUnit``.
    Exponents remain exact fractions, including those written as decimals.

    Parameters
    ----------
    expression : str
        Unit expression to parse. An empty or whitespace-only expression
        represents the multiplicative identity.

    Notes
    -----
    Parsing occurs on the first call to ``parse()``. Successful results are
    cached, and later calls return the same ``Monomial[SingleUnit]`` object.
    Failed attempts are not cached; a later call parses the original text again.

    The grammar for a nonempty expression is::

        expression     := (registered_name | product | "/" factor) EOF
        product        := factor (("*" | "/" | implicit_mul) factor)*
        factor         := base exponent?
        base           := UNIT | "1" | "(" product ")"
        exponent       := adjacent_NUM | ("**" | "^") exponent_value
        exponent_value := NUM | ("+(" | "-(" | "(") fraction ")"
        fraction       := NUM ("/" NUM)?

    ``UNIT`` is a nonempty token delimited by whitespace, decimal digits, or
    grammar operators. It must resolve to a registered symbol or singular
    English name, optionally prefixed. Inputs are case-sensitive, with aliases
    handled by ``SingleUnit``. Adjacent letters form one token: ``ms`` denotes a
    millisecond, while ``m s`` denotes metre times second.

    A registered multiword name, optionally prefixed, is recognized before
    parsing a product when it occupies the entire input. Whitespace between
    its words is treated as a single space. Inside products, parentheses,
    reciprocals, and powers, use its symbol or combine Unit objects instead;
    multiword names are not recognized as individual factors. For example,
    a registered ``astronomical unit`` can be used alone, while a compound
    expression must use its symbol, such as ``au / arcsec``.

    ``NUM`` is an optionally signed integer or decimal. Decimals require digits
    on both sides of the decimal point. Only the literal ``1`` is permitted as a
    base; other numbers are valid only as exponents or parts of an exponent.

    Whitespace, including newlines and Unicode spaces, separates tokens.
    Implicit multiplication applies when the next token is a unit, number, or
    opening parenthesis, as in ``kg m``, ``m 1/s``, or ``m(s)``. An adjacent
    number takes precedence as an exponent: ``m2`` means ``m^2``, while ``m 2``
    attempts to multiply by the invalid numeric base ``2``.

    Exponents written as ratios require parentheses, as in ``m^(1/2)`` or
    ``s^-(2/3)``. An outer sign must immediately precede the opening parenthesis;
    whitespace is allowed around the numbers and slash inside the parentheses.

    Explicit multiplication, implicit multiplication, and division have equal
    precedence and associate left to right. Thus ``m / s A`` means ``(m / s) * A``;
    use ``m / (s A)`` to put both units in the denominator.

    A leading slash accepts exactly one factor and must occupy the entire
    expression. ``/s`` is equivalent to ``1/s``, and ``/(m s)`` takes the
    reciprocal of a grouped product. Use ``1/m s`` for a product starting with a
    reciprocal. Inside parentheses, reciprocals must also be written with ``1/``.

    The result preserves canonical unit identity: ``m`` and ``mm`` remain
    distinct bases. Parsing does not perform magnitude or dimension conversion.

    Examples
    --------
    >>> parser = UnitParser('kg m/s2')
    >>> result = parser.parse()
    >>> result
    Monomial({SingleUnit('kg'): 1, SingleUnit('m'): 1, SingleUnit('s'): -2})
    >>> parser.parse() is result
    True
    >>> UnitParser('kg / (m s2)').parse()
    Monomial({SingleUnit('kg'): 1, SingleUnit('m'): -1, SingleUnit('s'): -2})
    >>> UnitParser('(m/s)^2').parse()
    Monomial({SingleUnit('m'): 2, SingleUnit('s'): -2})
    >>> UnitParser('m^(1/2)').parse()
    Monomial({SingleUnit('m'): Fraction(1, 2)})
    >>> UnitParser('m^0.1').parse().exponent_of(SingleUnit('m'))
    Fraction(1, 10)
    >>> UnitParser('m 1/s').parse()
    Monomial({SingleUnit('m'): 1, SingleUnit('s'): -1})
    >>> UnitParser('/(m s)').parse()
    Monomial({SingleUnit('m'): -1, SingleUnit('s'): -1})
    >>> UnitParser('1 1').parse()
    Monomial()
    """

    __slots__ = ('_expression', '_result', '_tokens', '_token')
    _expression: str
    _result: Monomial[SingleUnit] | None
    _tokens: Iterator[Token]
    _token: Token

    def __init__(self, expression: str) -> None:
        self._expression = expression
        self._result: Monomial[SingleUnit] | None = None

    def parse(self) -> Monomial[SingleUnit]:
        """Parse the complete expression and cache its successful result.

        Returns
        -------
        Monomial[SingleUnit]
            An immutable formal product with exact rational exponents.
            Repeated calls return the same object.

        Raises
        ------
        TypeError
            If ``expression`` is not a string.
        UnitSyntaxError
            If the expression has invalid syntax, a numeric base other than
            the literal ``1``, or a fraction exponent with a zero denominator.
            Extra valid factors after a leading reciprocal receive a dedicated
            diagnostic; other invalid suffixes report the first remaining token.
        UnitSymbolError
            If a unit symbol or name cannot be resolved by ``SingleUnit`` while
            parsing the expression.

        Warns
        -----
        UnitSymbolWarning
            If resolving a unit uses a potentially misleading prefix alias.
            Warnings can occur before a later syntax error is detected, and
            normal Python warning filters apply.

        Notes
        -----
        A failed attempt leaves no cached result. A later call starts from the
        original expression, even if failure occurred during suffix diagnosis.
        """
        if self._result is not None:
            return self._result
        self._tokens = map(Token.from_match, _TOKEN_RE.finditer(self._expression))
        name = ' '.join(self._expression.split())
        if ' ' in name:
            try:
                unit = SingleUnit(name)
            except UnitSymbolError:
                pass
            else:
                self._tokens = iter(())
                self.next_token()
                self._result = Monomial({unit: 1})
                return self._result
        self.next_token()
        if self.current_nonempty_token is EOF:
            self._result = Monomial()
            return self._result
        result = 1 / self._parse_factor() if self.match('DIV') else self._parse_product()
        if self.current_nonempty_token is not EOF:
            token = self.token
            if self.match('MUL') or self.match('DIV'):
                pass
            if self.current_nonempty_token.kind in ('UNIT', 'NUM', 'LPAREN'):
                try:
                    factor = self._parse_factor()
                except (UnitSyntaxError, UnitSymbolError):
                    factor = None
                if factor is not None:
                    error_message = (
                        "Leading '/' accepts only one factor. Use '1/' for multiple factors."
                    )
                    raise UnitSyntaxError(error_message)
            error_message = f'Unparsed redundant suffix {token.value!r}.'
            raise UnitSyntaxError(error_message)
        self._result = result
        return result

    @property
    def token(self) -> Token:
        """Return the current raw token without advancing the token stream."""
        return self._token

    def next_token(self) -> None:
        """Advance to the next raw token, or remain at ``EOF`` after exhaustion."""
        self._token = next(self._tokens, EOF)

    @property
    def current_nonempty_token(self) -> Token:
        """Consume leading whitespace and return the next non-whitespace token."""
        while self.token is not EOF and self.token.kind == 'SPACE':
            self.next_token()
        return self.token

    def match(self, kind: str, *, skip_space: bool = True) -> Token | None:
        """Consume a token of ``kind``, or return ``None`` if it does not match.

        With ``skip_space=True``, leading whitespace is consumed even on a
        mismatch. Set it to false to require the current raw token, preserving
        adjacency when parsing implicit exponents and outer signs.
        """
        token = self.current_nonempty_token if skip_space else self.token
        if token.kind == kind:  # match
            self.next_token()
            return token
        return None  # fail to match

    def _parse_product(self) -> Monomial[SingleUnit]:
        """Parse factors joined by explicit or implicit multiplication or division.

        Operators associate left to right. A numeric token also starts an
        implicit factor; ``_parse_base()`` validates that it is the literal ``1``.
        """
        result = self._parse_factor()
        while True:
            if self.match('MUL'):
                result *= self._parse_factor()
            elif self.match('DIV'):
                result /= self._parse_factor()
            elif self.token.kind in ('UNIT', 'NUM', 'LPAREN'):  # no consuming token
                result *= self._parse_factor()
            else:
                break
        return result

    def _parse_factor(self) -> Monomial[SingleUnit]:
        """Parse a base followed by at most one rational exponent.

        An implicit numeric exponent must be adjacent to the base. Explicit
        ``^`` and ``**`` operators permit intervening whitespace.
        """
        base = self._parse_base()
        if token := self.match('NUM', skip_space=False):
            return base ** Fraction(token.value)
        if self.match('POW'):
            return base ** self._parse_exponent_value()
        return base

    def _parse_base(self) -> Monomial[SingleUnit]:
        """Parse a registered unit, the literal identity, or a grouped product."""
        if token := self.match('UNIT'):
            return Monomial({SingleUnit(token.value): 1})
        elif token := self.match('NUM'):
            if token.value != '1':
                error_message = f'Invalid number {token.value!r} as a base.'
                raise UnitSyntaxError(error_message)
            return Monomial()
        elif self.match('LPAREN'):
            product = self._parse_product()
            if not self.match('RPAREN'):
                error_message = f'Product parenthesis not closed, got {self.token.value!r}.'
                raise UnitSyntaxError(error_message)
            return product
        error_message = f'Expected unit or parenthesized expression, got {self.token.value!r}.'
        raise UnitSyntaxError(error_message)

    def _parse_exponent_value(self) -> Fraction:
        """Parse an exact number or parenthesized fraction with an optional outer sign.

        An outer sign must be adjacent to the opening parenthesis. Signs on the
        numerator and denominator are part of their numeric tokens.
        """
        if token := self.match('NUM'):
            return Fraction(token.value)
        sign = 1
        if token := self.match('SIGN'):
            if token.value == '-':
                sign = -1
            if not self.match('LPAREN', skip_space=False):
                error_message = f'Expected parenthesized fraction, got {self.token.value!r}.'
                raise UnitSyntaxError(error_message)
        elif not self.match('LPAREN'):
            error_message = f'Expected number or parenthesized fraction, got {self.token.value!r}.'
            raise UnitSyntaxError(error_message)
        fraction = self._parse_fraction()
        if not self.match('RPAREN'):
            error_message = f'Fraction exponent parenthesis not closed, got {self.token.value!r}.'
            raise UnitSyntaxError(error_message)
        return sign * fraction

    def _parse_fraction(self) -> Fraction:
        """Parse a number or ratio of two numbers, preserving exact rational values.

        A missing numerator or denominator raises ``UnitSyntaxError``. A zero
        denominator is converted from ``ZeroDivisionError`` to ``UnitSyntaxError``.
        The caller consumes the closing parenthesis.
        """
        if not (numerator := self.match('NUM')):
            error_message = f'Expected numerator in fraction, got {self.token.value!r}.'
            raise UnitSyntaxError(error_message)
        if not self.match('DIV'):
            return Fraction(numerator.value)
        if not (denominator := self.match('NUM')):
            error_message = f'Expected denominator in fraction, got {self.token.value!r}.'
            raise UnitSyntaxError(error_message)
        try:
            return Fraction(numerator.value) / Fraction(denominator.value)
        except ZeroDivisionError:
            error_message = f'Division by zero: {numerator.value}/{denominator.value}.'
            raise UnitSyntaxError(error_message)
