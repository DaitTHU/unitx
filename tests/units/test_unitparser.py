"""Current unit grammar, exact algebra, diagnostics, and parser lifecycle."""

import re
import warnings
from fractions import Fraction

import pytest

from unitx.exceptions import UnitSymbolError, UnitSyntaxError
from unitx.monomial import Monomial
from unitx.singleunit import SingleUnit
from unitx.unit import Unit
from unitx.unitparser import Token, UnitParser

_WHITESPACE = [' ', '\t', '\n', '\r', '\r\n', '\v', '\f', '\xa0', '\u2003', '\u2028', '\u2029']


def test_token_equality() -> None:
    token = Token('UNIT', 'kg')
    assert token == Token('UNIT', 'kg')
    assert token != Token('UNIT', 'm')
    assert token != Token('NUM', '1')
    assert token != ('UNIT', 'kg')


def test_token_repr() -> None:
    token = Token('NUM', '-0.5')
    assert repr(token) == "Token('NUM', '-0.5')"
    assert eval(repr(token), {'Token': Token}) == token


@pytest.mark.parametrize(('pattern', 'kind'), [(r'(?P<UNIT>kg)', 'UNIT'), ('kg', None)])
def test_token_from_match(pattern: str, kind: str | None) -> None:
    match = re.fullmatch(pattern, 'kg')
    assert match is not None
    assert Token.from_match(match) == Token(kind, 'kg')


@pytest.mark.parametrize(
    'text',
    ['', ' ', '\t\n', '1', '(1)', '1*1', '/1', '1 1', '(1 1)', '1/1', '1^0', '1^(1/3)', '(1)2'],
)
def test_unit_parser_dimensionless(text: str) -> None:
    assert UnitParser(text).parse() == Monomial()


@pytest.mark.parametrize(
    ('text', 'symbol'),
    [
        ('m', 'm'),
        ('metre', 'm'),
        ('meter', 'm'),
        ('kilogram', 'kg'),
        ('µm', 'µm'),
        ('μm', 'µm'),
        ('um', 'µm'),
        ('micrometre', 'µm'),
        ('damol', 'damol'),
        ('decamole', 'damol'),
        ('second', 's'),
        ('candela', 'cd'),
        ('K', 'K'),
    ],
)
def test_unit_parser_units(text: str, symbol: str) -> None:
    result = UnitParser(text).parse()
    assert result == Monomial({SingleUnit(symbol): 1})
    assert next(iter(result.bases())).symbol == symbol


@pytest.mark.parametrize(
    'text',
    [
        'kg*m',
        'kg m',
        'kg(m)',
        '(kg)m',
        '(kg)(m)',
        '1 kg m',
        'kg*1*m',
        '((kg)*(m))',
        '1kg m',
        'kg m 1',
        'kg 1 m',
        'kg (1) m',
        'kg(m)1',
        'kg1m',
    ],
)
def test_unit_parser_multiplication(text: str) -> None:
    assert UnitParser(text).parse() == Monomial({SingleUnit('kg'): 1, SingleUnit('m'): 1})


@pytest.mark.parametrize(
    'text',
    ['m 1', 'm 1 1', 'm (1)', 'm/1', 'm*1', '1m', '1 m', '(1)m', 'm^1 1', '(m)1', 'm 1^(1/3)'],
)
def test_unit_parser_identity_product(text: str) -> None:
    assert UnitParser(text).parse() == Monomial({SingleUnit('m'): 1})


def test_unit_parser_names() -> None:
    result = UnitParser('kilogram metre / second2').parse()
    assert result == Monomial({SingleUnit('kg'): 1, SingleUnit('m'): 1, SingleUnit('s'): -2})


@pytest.mark.parametrize(
    ('text', 'components'),
    [
        ('m/s', {'m': 1, 's': -1}),
        ('/m', {'m': -1}),
        ('/m2', {'m': -2}),
        ('1/m2', {'m': -2}),
        ('/(m s)', {'m': -1, 's': -1}),
        ('m/s/A', {'m': 1, 's': -1, 'A': -1}),
        ('m / s A', {'m': 1, 's': -1, 'A': 1}),
        ('m / s * A', {'m': 1, 's': -1, 'A': 1}),
        ('m/(s A)', {'m': 1, 's': -1, 'A': -1}),
        ('m 1/s', {'m': 1, 's': -1}),
        ('m 1 / s', {'m': 1, 's': -1}),
        ('m (1/s)', {'m': 1, 's': -1}),
        ('m/(1 s)', {'m': 1, 's': -1}),
        ('1/m s', {'m': -1, 's': 1}),
        ('m/s*1', {'m': 1, 's': -1}),
    ],
)
def test_unit_parser_division(text: str, components: dict[str, int]) -> None:
    expected = Monomial({SingleUnit(symbol): exp for symbol, exp in components.items()})
    assert UnitParser(text).parse() == expected


@pytest.mark.parametrize('space', _WHITESPACE)
@pytest.mark.parametrize(
    ('expression', 'components'),
    [
        ('/m2', {'m': -2}),
        ('/m-2', {'m': 2}),
        ('/m^2', {'m': -2}),
        ('/m**2', {'m': -2}),
        ('/m^(1/2)', {'m': Fraction(-1, 2)}),
        ('/m^-(2/3)', {'m': Fraction(2, 3)}),
        ('/(m s)2', {'m': -2, 's': -2}),
        ('/1^2', {}),
    ],
)
def test_unit_parser_reciprocal_whitespace(
    space: str, expression: str, components: dict[str, int | Fraction]
) -> None:
    parser = UnitParser(f'{space}{expression}{space}')
    result = parser.parse()
    assert result == Monomial({SingleUnit(symbol): exp for symbol, exp in components.items()})
    assert parser.parse() is result


@pytest.mark.parametrize(
    ('text', 'exponent'),
    [
        ('m2', Fraction(2)),
        ('m+2', Fraction(2)),
        ('m-2', Fraction(-2)),
        ('m0', Fraction(0)),
        ('m^2', Fraction(2)),
        ('m**2', Fraction(2)),
        ('m ^ -2', Fraction(-2)),
        ('m**+2', Fraction(2)),
        ('m0.1', Fraction(1, 10)),
        ('m-0.5', Fraction(-1, 2)),
        ('m^0.1', Fraction(1, 10)),
        ('m^-0.125', Fraction(-1, 8)),
        ('m^(1/3)', Fraction(1, 3)),
        ('m**(1/3)', Fraction(1, 3)),
        ('m^(2/6)', Fraction(1, 3)),
        ('m^-(2/3)', Fraction(-2, 3)),
        ('m^+(2/3)', Fraction(2, 3)),
        ('m^(-2/3)', Fraction(-2, 3)),
        ('m^(2/-3)', Fraction(-2, 3)),
        ('m^-(2/-3)', Fraction(2, 3)),
        ('m^(+2/+3)', Fraction(2, 3)),
        ('m^(0/3)', Fraction(0)),
        ('m^(0.5/1.5)', Fraction(1, 3)),
        ('m^(1.5/0.5)', Fraction(3)),
        ('m^(1/0.1)', Fraction(10)),
        ('m^(0/-3)', Fraction(0)),
        ('m^(2)', Fraction(2)),
        ('m^-(2)', Fraction(-2)),
        ('m^(-0.25)', Fraction(-1, 4)),
        ('m^0.00000000000000000001', Fraction(1, 10**20)),
        ('m^(1/1000001)', Fraction(1, 1000001)),
        ('m^123456789012345678901234567890', Fraction(123456789012345678901234567890)),
    ],
)
def test_unit_parser_exponents(text: str, exponent: Fraction) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = UnitParser(text).parse()
    assert result == Monomial({SingleUnit('m'): exponent})
    assert type(result.exponent_of(SingleUnit('m'))) is Fraction


@pytest.mark.parametrize(
    ('text', 'components'),
    [
        ('(m/s)^2', {'m': 2, 's': -2}),
        ('(m/s)-2', {'m': -2, 's': 2}),
        ('(m/s)0.5', {'m': Fraction(1, 2), 's': Fraction(-1, 2)}),
        ('((kg m)/s2)^(1/2)', {'kg': Fraction(1, 2), 'm': Fraction(1, 2), 's': -1}),
        ('m/(s/A)', {'m': 1, 's': -1, 'A': 1}),
        ('m(s A)^-1', {'m': 1, 's': -1, 'A': -1}),
        ('(1/m)/s', {'m': -1, 's': -1}),
        ('((m))', {'m': 1}),
        ('((m/s)^(1/3))3', {'m': 1, 's': -1}),
        ('(m^0) s', {'s': 1}),
        ('m/(s^0)', {'m': 1}),
    ],
)
def test_unit_parser_groups(text: str, components: dict[str, int | Fraction]) -> None:
    expected = Monomial({SingleUnit(symbol): exp for symbol, exp in components.items()})
    assert UnitParser(text).parse() == expected


@pytest.mark.parametrize('space', _WHITESPACE)
def test_unit_parser_whitespace(space: str) -> None:
    assert UnitParser(f'{space}kg{space}m{space}').parse() == Monomial(
        {SingleUnit('kg'): 1, SingleUnit('m'): 1}
    )
    assert UnitParser(f'm{space}^{space}({space}1{space}/{space}2{space})').parse() == Monomial(
        {SingleUnit('m'): Fraction(1, 2)}
    )
    assert UnitParser(f'm{space}^{space}-({space}1{space}/{space}2{space})').parse() == Monomial(
        {SingleUnit('m'): Fraction(-1, 2)}
    )
    assert UnitParser(f'{space}/{space}m{space}').parse() == Monomial({SingleUnit('m'): -1})
    assert UnitParser(f'm{space}1{space}/{space}s').parse() == Monomial(
        {SingleUnit('m'): 1, SingleUnit('s'): -1}
    )


@pytest.mark.parametrize('space', _WHITESPACE)
def test_unit_parser_group_spacing(space: str) -> None:
    expected = Monomial({SingleUnit('m'): 2, SingleUnit('s'): -2})
    group = f'({space}m{space}/{space}s{space})'
    assert UnitParser(f'{group}2{space}').parse() == expected
    assert UnitParser(f'{group}{space}^{space}2{space}').parse() == expected


@pytest.mark.parametrize(
    ('text', 'symbols'),
    [
        ('m s / m m', ['s', 'm']),
        ('m (s / m) m', ['s', 'm']),
        ('m s / (m s) A m', ['A', 'm']),
        ('um s/μm um', ['s', 'µm']),
        ('m^(1/3) s m^(2/3) / m m', ['s', 'm']),
        ('1/m m s', ['s']),
    ],
)
def test_unit_parser_component_order(text: str, symbols: list[str]) -> None:
    result = UnitParser(text).parse()
    assert list(result.components()) == [(SingleUnit(symbol), Fraction(1)) for symbol in symbols]


def test_unit_parser_long_product() -> None:
    symbols = [
        prefix + unit
        for prefix in ('', 'k', 'm', 'µ', 'n', 'M')
        for unit in ('s', 'm', 'g', 'A', 'K', 'mol', 'cd')
    ]
    product = ' '.join(f'{symbol}^(1/3)' for symbol in symbols)
    result = UnitParser(f'{product} {product} {product}').parse()
    assert list(result.components()) == [(SingleUnit(symbol), Fraction(1)) for symbol in symbols]


def test_unit_parser_unit_boundaries() -> None:
    assert UnitParser('ms').parse() == Monomial({SingleUnit('ms'): 1})
    assert UnitParser('m s').parse() == Monomial({SingleUnit('m'): 1, SingleUnit('s'): 1})
    assert UnitParser('m2s3').parse() == Monomial({SingleUnit('m'): 2, SingleUnit('s'): 3})
    assert UnitParser('m^2s3').parse() == Monomial({SingleUnit('m'): 2, SingleUnit('s'): 3})
    assert UnitParser('m/mm').parse() == Monomial({SingleUnit('m'): 1, SingleUnit('mm'): -1})


@pytest.mark.parametrize(
    'text',
    ['m/m', 'μm/um', 'metre/meter', '(m^(1/3))3/m', '(kg m/s2)/(s-2 kg m)', 'm^0', '(1/m) m'],
)
def test_unit_parser_cancellation(text: str) -> None:
    assert UnitParser(text).parse() == Monomial()


@pytest.mark.parametrize('symbol', ['s', 'm', 'g', 'A', 'K', 'mol', 'cd'])
def test_unit_parser_invalid_kilo(symbol: str) -> None:
    with pytest.raises(UnitSymbolError):
        UnitParser('K' + symbol).parse()
    assert UnitParser('k' + symbol).parse() == Monomial({SingleUnit('k' + symbol): 1})
    expected = Monomial({SingleUnit('K'): 1}) * Monomial({SingleUnit(symbol): 1})
    assert UnitParser('K ' + symbol).parse() == expected


@pytest.mark.parametrize('text', ['', ' \n ', 'm', 'm/s2', 'm/m', '(kg m/s2)^(1/3)'])
def test_unit_parser_cached_result(text: str) -> None:
    parser = UnitParser(text)
    result = parser.parse()
    before = dict(result.components()), hash(result)
    for _ in range(3):
        assert parser.parse() is result
    assert (dict(result.components()), hash(result)) == before


@pytest.mark.parametrize('text', ['m/', 'm*', '(m/)'])
def test_unit_parser_invalid_retry(text: str) -> None:
    parser = UnitParser(text)
    for _ in range(2):
        with pytest.raises(UnitSyntaxError, match='Expected unit or parenthesized expression'):
            parser.parse()


@pytest.mark.parametrize('text', ['m)', '/m 2', '/m s^(1/0)', '/m unknown'])
def test_unit_parser_invalid_retry_suffix(text: str) -> None:
    parser = UnitParser(text)
    messages = []
    for _ in range(2):
        with pytest.raises(UnitSyntaxError, match='Unparsed redundant suffix') as caught:
            parser.parse()
        messages.append(str(caught.value))
    assert messages[0] == messages[1]


def test_unit_parser_invalid_retry_multiple() -> None:
    parser = UnitParser('/m 1')
    for _ in range(2):
        with pytest.raises(UnitSyntaxError, match="Leading '/'"):
            parser.parse()


def test_unit_parser_invalid_retry_number() -> None:
    parser = UnitParser('m 2')
    for _ in range(2):
        with pytest.raises(UnitSyntaxError, match="Invalid number '2' as a base"):
            parser.parse()


def test_unit_parser_invalid_retry_denominator() -> None:
    parser = UnitParser('m^(1/0)')
    for _ in range(2):
        with pytest.raises(UnitSyntaxError, match='Division by zero'):
            parser.parse()


@pytest.mark.parametrize('text', ['unknown', 'm/unknown'])
def test_unit_parser_invalid_retry_symbol(text: str) -> None:
    parser = UnitParser(text)
    for _ in range(2):
        with pytest.raises(UnitSymbolError, match='is not a valid unit symbol'):
            parser.parse()


@pytest.mark.parametrize('space', _WHITESPACE)
@pytest.mark.parametrize('sign', ['+', '-'])
def test_unit_parser_invalid_sign_spacing(space: str, sign: str) -> None:
    with pytest.raises(UnitSyntaxError, match='Expected parenthesized fraction'):
        UnitParser(f'm^{sign}{space}(1/2)').parse()


@pytest.mark.parametrize('text', ['m^+', 'm^-', 'm^++2', 'm^--2'])
def test_unit_parser_invalid_signed_exponent(text: str) -> None:
    with pytest.raises(UnitSyntaxError, match='Expected parenthesized fraction'):
        UnitParser(text).parse()


@pytest.mark.parametrize('expression', [None, True, 1, 1.0, [], {}])
def test_unit_parser_invalid_type(expression: object) -> None:
    with pytest.raises(TypeError):
        UnitParser(expression).parse()  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize('expression', [b'm', bytearray(b'm')])
def test_unit_parser_invalid_bytes(expression: bytes | bytearray) -> None:
    with pytest.raises(TypeError):
        UnitParser(expression).parse()  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    'text',
    ['unknown', 'k', 'mkg', 'Metre', 'metres', 'm/unknown', 'm ? s', 'm \x00 s', 'm \u200b s'],
)
def test_unit_parser_invalid_symbol(text: str) -> None:
    with pytest.raises(UnitSymbolError, match='is not a valid unit symbol'):
        UnitParser(text).parse()


@pytest.mark.parametrize(
    'text',
    [
        '0',
        '2',
        '-1',
        '+1',
        '1.0',
        '0.5',
        'm/2',
        '2 m',
        '(2)',
        '1*3',
        'm 2',
        'm^2 3',
        'm 0',
        'm +1',
        'm 1.0',
        'm 01',
    ],
)
def test_unit_parser_invalid_number(text: str) -> None:
    with pytest.raises(UnitSyntaxError, match='Invalid number .*as a base'):
        UnitParser(text).parse()


@pytest.mark.parametrize(
    'text',
    [
        '/',
        '()',
        '( )',
        '*m',
        'm/',
        'm*',
        'm//s',
        'm/*s',
        ')',
        '(m/)',
        '+',
        '-',
        '(/m)',
        'm(/s)',
        'm/(/s)',
        '(/m)/s',
        '(/m) m',
    ],
)
def test_unit_parser_invalid_factor(text: str) -> None:
    with pytest.raises(UnitSyntaxError, match='Expected unit or parenthesized expression'):
        UnitParser(text).parse()


@pytest.mark.parametrize('text', ['(m', '((m)', 'm/(s', '(m/s', '(m s'])
def test_unit_parser_invalid_parenthesis(text: str) -> None:
    with pytest.raises(UnitSyntaxError, match='Product parenthesis not closed'):
        UnitParser(text).parse()


@pytest.mark.parametrize(
    ('text', 'suffix'),
    [
        ('m)', ')'),
        ('m2^3', '^'),
        ('m^2^3', '^'),
        ('m^(1/2)^2', '^'),
        ('m + s', '+'),
        ('m - s', '-'),
    ],
)
def test_unit_parser_invalid_suffix(text: str, suffix: str) -> None:
    with pytest.raises(UnitSyntaxError) as caught:
        UnitParser(text).parse()
    assert str(caught.value) == f'Unparsed redundant suffix {suffix!r}.'


@pytest.mark.parametrize(
    'text',
    [
        '/m s',
        '/m*s',
        '/m/s',
        '/m2s',
        '/m2 s',
        '/(m s)2 A',
        '/m 1',
        '/m*1',
        '/m/1',
        '/m (1)',
        '/m s0',
        '/m 1^0',
        '/m (s/s)',
    ],
)
def test_unit_parser_invalid_multiple_factors(text: str) -> None:
    with pytest.raises(UnitSyntaxError, match="Leading '/'"):
        UnitParser(text).parse()


@pytest.mark.parametrize('space', _WHITESPACE)
@pytest.mark.parametrize('operator', ['*', '/'])
@pytest.mark.parametrize('factor', ['s', '1'])
def test_unit_parser_invalid_reciprocal_spacing(space: str, operator: str, factor: str) -> None:
    with pytest.raises(UnitSyntaxError, match="Leading '/'"):
        UnitParser(f'/m{space}{operator}{space}{factor}').parse()


@pytest.mark.parametrize(
    ('text', 'suffix'),
    [
        ('/m 2', '2'),
        ('/m2 3', '3'),
        ('/m* 2', '*'),
        ('/m/ 2', '/'),
        ('/m2(', '('),
        ('/m2)3', ')'),
        ('/m2^3', '^'),
        ('/m unknown', 'unknown'),
        ('/m*unknown', '*'),
        ('/m (s', '('),
        ('/m s^(1/0)', 's'),
        ('/m s^(1/)', 's'),
        ('/m*(s^(1/0))', '*'),
        ('/m s^', 's'),
        ('/m s^(1/2', 's'),
    ],
)
def test_unit_parser_invalid_reciprocal_suffix(text: str, suffix: str) -> None:
    with pytest.raises(UnitSyntaxError) as caught:
        UnitParser(text).parse()
    assert str(caught.value) == f'Unparsed redundant suffix {suffix!r}.'


@pytest.mark.parametrize('space', _WHITESPACE)
@pytest.mark.parametrize('expression', ['m{space}2', 'm{space}-2', '(m){space}+2', '(m/s){space}2'])
def test_unit_parser_invalid_number_spacing(space: str, expression: str) -> None:
    with pytest.raises(UnitSyntaxError, match='Invalid number .*as a base'):
        UnitParser(expression.format(space=space)).parse()


@pytest.mark.parametrize('text', ['m^', 'm**', 'm^s', 'm^^2', 'm^*2', 'm^/2', 'm^)'])
def test_unit_parser_invalid_exponent(text: str) -> None:
    with pytest.raises(UnitSyntaxError, match='Expected number or parenthesized fraction'):
        UnitParser(text).parse()


@pytest.mark.parametrize('text', ['m^()', 'm^(/2)', 'm^(s/2)', 'm^-()'])
def test_unit_parser_invalid_numerator(text: str) -> None:
    with pytest.raises(UnitSyntaxError, match='Expected numerator in fraction'):
        UnitParser(text).parse()


@pytest.mark.parametrize('text', ['m^(1/)', 'm^(1/s)', 'm^(1/', 'm^-(1/)'])
def test_unit_parser_invalid_denominator(text: str) -> None:
    with pytest.raises(UnitSyntaxError, match='Expected denominator in fraction'):
        UnitParser(text).parse()


@pytest.mark.parametrize(
    'text', ['m^(1/2', 'm^(1/2 s)', 'm^(1/2/3)', 'm^(1', 'm^(1 2)', 'm^(1 s)', 'm^-(2*3)']
)
def test_unit_parser_invalid_exponent_parenthesis(text: str) -> None:
    with pytest.raises(UnitSyntaxError, match='Fraction exponent parenthesis not closed'):
        UnitParser(text).parse()


@pytest.mark.parametrize(
    'text', ['m^(1/0)', 'm^(0/0)', 'm^-(1/-0.00)', 'm**(0.5/+0.0)', '/m^(1/0)', 'm^(0.0/-0.0)']
)
def test_unit_parser_invalid_zero_denominator(text: str) -> None:
    with pytest.raises(UnitSyntaxError, match='Division by zero'):
        UnitParser(text).parse()


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('[metre]', 'm'),
        ('[metre][second]', 'm*s'),
        ('[metre]s', 'm*s'),
        ('m[second]', 'm*s'),
        ('/[metre]', '1/m'),
        ('([metre] / s)**2', 'm2/s2'),
        ('[metre]2', 'm2'),
        ('[kilometre] / [second]', 'km/s'),
        ('[ astronomical\tunit ] / s', 'au/s'),
    ],
)
def test_unit_parser_name_reference(text: str, expected: str) -> None:
    parser = UnitParser(text)
    result = parser.parse()
    assert result is parser.parse()
    assert Unit(text) == Unit(expected)
    assert Unit(Unit(text).expression()) == Unit(text)


@pytest.mark.parametrize('text', ['[]', '[  \t\n]'])
def test_unit_parser_invalid_empty_reference(text: str) -> None:
    with pytest.raises(UnitSyntaxError, match='name reference must not be empty'):
        Unit(text)


@pytest.mark.parametrize('text', ['[metre', '[[metre]]', '[foo[bar]]', 'm/[metre'])
def test_unit_parser_invalid_open_reference(text: str) -> None:
    parser = UnitParser(text)
    for _ in range(2):
        with pytest.raises(UnitSyntaxError, match='Unmatched character'):
            parser.parse()


@pytest.mark.parametrize('text', ['[metre]]', 'metre]', '[metre][second]]'])
def test_unit_parser_invalid_close_reference(text: str) -> None:
    with pytest.raises(UnitSyntaxError):
        Unit(text)


@pytest.mark.parametrize('text', ['[m]', '[m/s]', '[metre second]', '[2]', '[unknown]', 'k[metre]'])
def test_unit_parser_invalid_name_reference(text: str) -> None:
    with pytest.raises(UnitSymbolError):
        Unit(text)


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('m²', 'm2'),
        ('m⁻²', 'm-2'),
        ('m⁺¹²', 'm12'),
        ('m⁰', ''),
        ('(m/s)²', '(m/s)2'),
        ('[metre]⁻¹²', 'm-12'),
        ('m⁻²s³', 'm-2*s3'),
    ],
)
def test_unit_parser_superscript(text: str, expected: str) -> None:
    assert Unit(text) == Unit(expected)
    assert Unit(Unit(text).expression()) == Unit(text)


@pytest.mark.parametrize('text', ['m⁻', 'm⁺', 'm⁻⁺²', 'm²⁻¹'])
def test_unit_parser_invalid_superscript(text: str) -> None:
    with pytest.raises(UnitSyntaxError, match='Invalid superscript exponent'):
        Unit(text)


@pytest.mark.parametrize('text', ['m ²', 'm^²', 'm2²', 'm²2', 'm²³⁺', 'm⁻²³⁻'])
def test_unit_parser_invalid_mixed_exponent(text: str) -> None:
    with pytest.raises(UnitSyntaxError):
        Unit(text)
