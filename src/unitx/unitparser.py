"""Parse physical unit expressions into formal products of single units."""

from __future__ import annotations

import re
from collections.abc import Iterator
from fractions import Fraction

from ._formatting.unicode import SUP_TO_DIGIT, SUPERSCRIPTS
from .exceptions import UnitSymbolError, UnitSyntaxError
from .monomial import Monomial
from .singleunit import SingleUnit
from .unitdefinition import _UNIT_SYMBOL_PATTERN
from .unitregistry import Registry, default_registry

_PATTERNS = {
    'POW': r'\*\*|\^',
    'MUL': r'\*',
    'DIV': r'/',
    'NUM': r'[-+]?\d+(?:\.\d+)?',  # supports integers and decimals
    'LPAREN': r'\(',
    'RPAREN': r'\)',
    'SIGN': r'[-+]',  # leading sign, e.g., m^-(2/3)
    'SPACE': r'\s+',
    'NAME_REF': r'\[[^\[\]]*\]',
    'SUPERSCRIPT': '[' + SUPERSCRIPTS + ']+',
    'UNIT': _UNIT_SYMBOL_PATTERN,
    'UNMATCHED': r'.',  # any single character not matched by other patterns
}

_TOKEN_RE = re.compile('|'.join(f'(?P<{kind}>{pattern})' for kind, pattern in _PATTERNS.items()))
_INTEGER_RE = re.compile(r'[+-]?[0-9]+')


class Token:
    """A lexical token identified by its kind and matched text.

    Parameters
    ----------
    kind : str or None
        Name of the matched regular expression group. `None` represents a
        match without a named group.
    value : str
        Matched text, or a description for a sentinel such as `EOF`.

    Notes
    -----
    Equality compares both kind and value. Unit symbol validation belongs to
    `SingleUnit` and occurs during parsing, rather than token construction.
    """

    __slots__ = ('kind', 'value')

    def __init__(self, kind: str | None, value: str) -> None:
        self.kind, self.value = kind, value
        if kind == 'UNMATCHED':
            error_message = f'Unmatched character {value!r}.'
            raise UnitSyntaxError(error_message)

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
    Unit symbols, names, prefixes, and aliases resolve in the explicit `registry`,
    or the default environment when it is omitted. Returned single units bind
    immutable definitions rather than a mutable registry.
    Exponents remain exact fractions, including those written as decimals.

    Parameters
    ----------
    expression : str
        Unit expression to parse. An empty or whitespace-only expression
        represents the multiplicative identity.

    Notes
    -----
    Parsing occurs on the first call to `parse()`. Successful results are
    cached, and later calls return the same `Monomial[SingleUnit]` object.
    Failed attempts are not cached; a later call parses the original text again.
    An instance owns mutable parsing state and is not guaranteed safe for
    concurrent calls. Use separate instances or synchronize access. Concurrent
    parsers may share a Registry while its bindings remain unchanged.

    The grammar for a nonempty expression is::

        expression     := (registered_name | product | "/" factor) EOF
        product        := factor (("*" | "/" | implicit_mul) factor)*
        factor         := base exponent?
        base           := UNIT | NAME_REF | "1" | "(" product ")"
        exponent       := adjacent_NUM | adjacent_SUPERSCRIPT | ("**" | "^") exponent_value
        exponent_value := NUM | ("+(" | "-(" | "(") fraction ")"
        fraction       := NUM ("/" NUM)?

    `UNIT` is a nonempty token delimited by whitespace, decimal digits, or
    grammar operators. It must resolve to a registered symbol or singular
    English name, optionally prefixed. Inputs are case-sensitive, with aliases
    handled by `SingleUnit`. Adjacent letters form one token: `ms` denotes a
    millisecond, while `m s` denotes metre times second.

    A registered multiword or hyphenated name, optionally prefixed, is
    recognized before parsing a product when it occupies the entire input.
    Whitespace between its words is normalized to a single space. Inside
    expressions, use `[name]` to make a name one factor, for example
    `[astronomical unit] / s`, or use its symbol. NAME_REF tokens resolve
    only registered names and name aliases, never symbols or expressions.
    Prefix names belong inside the brackets and obey that spelling's policy.
    Empty, nested, and unclosed references are invalid. Quotes retain their
    existing unit-symbol meaning; there are no string escapes or interpolation.

    Adjacent Unicode superscripts denote exact integer powers, for example
    `m²`, `s⁻¹²`, and `[metre]⁺²`. Only an optional leading sign followed
    by superscript digits is accepted. Superscripts cannot follow whitespace,
    `^`, or another exponent; fractional powers use the ASCII grammar below.
    Brackets and these superscript characters are reserved, including when
    registering custom unit symbols and names.

    `NUM` is an optionally signed integer or decimal. Decimals require digits
    on both sides of the decimal point. Only the literal `1` is permitted as a
    base; other numbers are valid only as exponents or parts of an exponent.

    Whitespace, including newlines and Unicode spaces, separates tokens.
    Implicit multiplication applies when the next token is a unit, name
    reference, number, or opening parenthesis, as in `kg m`, `m[second]`,
    `m 1/s`, or `m(s)`. An adjacent
    number takes precedence as an exponent: `m2` means `m^2`, while `m 2`
    attempts to multiply by the invalid numeric base `2`.

    Exponents written as ratios require parentheses, as in `m^(1/2)` or
    `s^-(2/3)`. An outer sign must immediately precede the opening parenthesis;
    whitespace is allowed around the numbers and slash inside the parentheses.
    Both parts may be decimals: `m^(1.5/0.5)` is exactly `m^3`.

    Explicit multiplication, implicit multiplication, and division have equal
    precedence and associate left to right. Thus `m / s A` means `(m / s) * A`;
    use `m / (s A)` to put both units in the denominator.

    A leading slash accepts exactly one factor and must occupy the entire
    expression. `/s` is equivalent to `1/s`, and `/(m s)` takes the
    reciprocal of a grouped product. Use `1/m s` for a product starting with a
    reciprocal. Inside parentheses, reciprocals must also be written with `1/`.

    The result preserves canonical unit identity: `m` and `mm` remain
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

    __slots__ = ('_expression', '_result', '_tokens', '_token', '_registry')
    _expression: str
    _result: Monomial[SingleUnit] | None
    _tokens: Iterator[Token]
    _token: Token

    def __init__(self, expression: str, *, registry: Registry | None = None) -> None:
        self._registry = default_registry if registry is None else registry
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
            If `expression` is not a string.
        UnitSyntaxError
            If the expression has invalid syntax, a numeric base other than
            the literal `1`, or a fraction exponent with a zero denominator.
            Extra valid factors after a leading reciprocal receive a dedicated
            diagnostic; other invalid suffixes report the first remaining token.
        UnitSymbolError
            If a unit symbol or name cannot be resolved by `SingleUnit` while
            parsing the expression.

        Notes
        -----
        expression := (registered_name | product | "/" factor) EOF

        A failed attempt leaves no cached result. A later call starts from the
        original expression, even if failure occurred during suffix diagnosis.
        """
        if self._result is not None:
            return self._result
        if not isinstance(self._expression, str):
            raise TypeError('Unit expression must be str.')

        # Resolve a complete registered name before interpreting its words as factors.
        result = self._try_parse_name()
        if result is not None:
            self._result = result
            return result

        self._tokens = map(Token.from_match, _TOKEN_RE.finditer(self._expression))
        self.next_token()
        if self.current_nonempty_token is EOF:
            result = Monomial()
        elif self.match('DIV'):
            result = 1 / self._parse_factor()
        else:
            result = self._parse_product()

        self._expect_end()
        self._result = result
        return result

    def _try_parse_name(self) -> Monomial[SingleUnit] | None:
        """Try a complete multiword or hyphenated registered name before tokenization."""
        name = ' '.join(self._expression.split())
        if ' ' not in name and '-' not in name:
            return None
        try:
            unit = SingleUnit._from_definitions(*self._registry._resolve_name(name))
        except UnitSymbolError:
            return None
        self._tokens = iter(())
        self.next_token()
        return Monomial({unit: 1})

    def _expect_end(self) -> None:
        """Require EOF, diagnosing extra factors after a leading reciprocal.

        A complete extra factor receives the dedicated leading-slash diagnostic.
        An invalid or incomplete factor reports the original suffix instead.
        """
        if self.current_nonempty_token is EOF:
            return
        token = self.token
        if self.match('MUL') or self.match('DIV'):
            pass
        try:
            self._parse_factor()
        except (UnitSyntaxError, UnitSymbolError):
            pass
        else:
            error_message = (
                "Leading '/' followed by more than one factor is ambiguous. "
                "Use '1/(multiple_factors)' instead."
            )
            raise UnitSyntaxError(error_message)
        error_message = f'Unparsed redundant suffix {token.value!r}.'
        raise UnitSyntaxError(error_message)

    @property
    def token(self) -> Token:
        """Return the current raw token without advancing the token stream."""
        return self._token

    def next_token(self) -> None:
        """Advance to the next raw token, or remain at `EOF` after exhaustion."""
        self._token = next(self._tokens, EOF)

    @property
    def current_nonempty_token(self) -> Token:
        """Consume leading whitespace and return the next non-whitespace token."""
        while self.token is not EOF and self.token.kind == 'SPACE':
            self.next_token()
        return self.token

    def match(self, kind: str, *, skip_space: bool = True) -> Token | None:
        """Consume a token of `kind`, or return `None` if it does not match.

        With `skip_space=True`, leading whitespace is consumed even on a
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

        product := factor (("*" | "/" | implicit_mul) factor)*

        Operators associate left to right. A numeric token also starts an
        implicit factor; `_parse_base()` validates that it is the literal `1`.
        """
        result = self._parse_factor()
        while True:
            if self.match('MUL'):
                result *= self._parse_factor()
            elif self.match('DIV'):
                result /= self._parse_factor()
            elif self.token.kind in ('UNIT', 'NAME_REF', 'NUM', 'LPAREN'):  # no consuming token
                result *= self._parse_factor()
            else:
                break
        return result

    def _parse_factor(self) -> Monomial[SingleUnit]:
        """Parse a base followed by at most one rational exponent.

        factor   := base exponent?
        exponent := adjacent_NUM | adjacent_SUPERSCRIPT | ("**" | "^") exponent_value

        An implicit numeric or superscript exponent must be adjacent to the base. Explicit
        `^` and `**` operators permit intervening whitespace.
        """
        base = self._parse_base()
        if token := self.match('SUPERSCRIPT', skip_space=False):
            exponent = token.value.translate(SUP_TO_DIGIT)
            if _INTEGER_RE.fullmatch(exponent) is None:
                error_message = f'Invalid superscript exponent {token.value!r}.'
                raise UnitSyntaxError(error_message)
            return base ** Fraction(exponent)
        if token := self.match('NUM', skip_space=False):
            return base ** Fraction(token.value)
        if self.match('POW'):
            return base ** self._parse_exponent_value()
        return base

    def _parse_base(self) -> Monomial[SingleUnit]:
        """Parse a registered unit, the literal identity, or a grouped product.

        base := UNIT | NAME_REF | "1" | "(" product ")"
        """
        if token := self.match('UNIT'):
            return Monomial(
                {SingleUnit._from_definitions(*self._registry._resolve(token.value)): 1}
            )
        elif token := self.match('NAME_REF'):
            name = ' '.join(token.value[1:-1].split())
            if not name:
                error_message = 'Unit name reference must not be empty.'
                raise UnitSyntaxError(error_message)
            return Monomial({SingleUnit._from_definitions(*self._registry._resolve_name(name)): 1})
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

        exponent_value := NUM | ("+(" | "-(" | "(") fraction ")"

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

        fraction := NUM ("/" NUM)?

        NUM permits signed integers and decimals in both positions; for example,
        1.5/0.5 is exactly 3. The superscript integer restriction does not apply.
        A missing numerator or denominator raises `UnitSyntaxError`. A zero
        denominator is converted from `ZeroDivisionError` to `UnitSyntaxError`.
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
