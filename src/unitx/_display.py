"""Internal structural rendering; no numerical evaluation or optional dependencies."""

from __future__ import annotations

import re
from decimal import Decimal
from fractions import Fraction
from html import escape
from numbers import Complex, Number, Real
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .affineunit import AffineUnit
    from .unit import Unit

type Style = Literal['latex', 'html']

_DECIMAL_NUMBER = re.compile(r'([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))(?:[eE]([+-]?[0-9]+))?')


def _display_unit(unit: Unit | AffineUnit) -> Unit | AffineUnit:
    """Render an unnamed zero-origin linear coordinate through its complete Unit."""
    from .affineunit import AffineUnit

    if (
        isinstance(unit, AffineUnit)
        and unit._definition is None
        and unit.kind is None
        and unit.origin.value == 0
        and unit.origin.unit == unit.delta_unit
    ):
        return unit.delta_unit
    return unit


def latex_text(text: str) -> str:
    """Escape literal text, including characters that can terminate math mode."""
    replacements = {
        '\\': r'\textbackslash{}',
        '{': r'\{',
        '}': r'\}',
        '$': r'\$',
        '&': r'\&',
        '#': r'\#',
        '%': r'\%',
        '_': r'\_',
        '^': r'\textasciicircum{}',
        '~': r'\textasciitilde{}',
        '\n': ' ',
        '\r': ' ',
        '\t': ' ',
    }
    return r'\text{' + ''.join(replacements.get(char, char) for char in text) + '}'


def product(components: list[tuple[str, Fraction]], style: Style, *, fraction: bool = True) -> str:
    numerator: list[str] = []
    denominator: list[str] = []
    for base, exponent in components:
        target = numerator if exponent > 0 or not fraction else denominator
        power = abs(exponent) if fraction else exponent
        rendered = latex_text(base) if style == 'latex' else escape(base)
        if power != 1:
            exp = str(power)
            rendered += f'^{{{exp}}}' if style == 'latex' else f'<sup>{exp}</sup>'
        target.append(rendered)
    joiner = r'\,' if style == 'latex' else '&thinsp;'
    top = joiner.join(numerator) or '1'
    if not denominator:
        return top
    bottom = joiner.join(denominator)
    if len(denominator) > 1:
        bottom = f'({bottom})'
    return f'{top}/{bottom}'


def format_unit(unit: Unit | AffineUnit, format_spec: str) -> str:
    """Select rich output or apply ordinary string formatting to a unit."""
    if format_spec == 'latex':
        return unit.to_latex()
    if format_spec == 'html':
        return unit.to_html()
    if format_spec == 'siunitx':
        return unit.to_siunitx()
    if format_spec == 'text':
        return str(unit)
    return format(str(unit), format_spec)


def siunitx_unit(unit: Unit | AffineUnit) -> str:
    body, parsed = siunitx_unit_parts(unit)
    options = '' if parsed else '[parse-units=false]'
    return rf'\unit{options}{{{body or "1"}}}'


def unit_body(unit: Unit | AffineUnit, style: Style, *, fraction: bool = True) -> str:
    from .affineunit import AffineUnit
    from .factor import one
    from .unit import ScaledUnit

    unit = _display_unit(unit)
    if isinstance(unit, AffineUnit):
        return latex_text(unit.symbol) if style == 'latex' else escape(unit.symbol)
    components = [(str(base), exp) for base, exp in unit._monomial.components()]
    body = product(components, style, fraction=fraction)
    if isinstance(unit, ScaledUnit) and unit.scale != one:
        scale = product(
            [(str(base), exp) for base, exp in unit.scale._display_monomial().components()],
            style,
            fraction=fraction,
        )
        if not components:
            return scale
        return scale + (r'\,' if style == 'latex' else '&thinsp;') + body
    return body


def format_magnitude(value: object, spec: str) -> str:
    """Delegate numeric formatting and group complex values only when necessary."""
    text = format(value, spec)
    # Native complex formatting may already group the number, with padding on
    # either side. A single fill character cannot introduce both parentheses.
    if (
        isinstance(value, Complex)
        and not isinstance(value, Real)
        and ('(' not in text or ')' not in text)
    ):
        return f'({text})'
    return text


def rich_magnitude(value: object, text: str, spec: str, style: Style) -> str:
    """Typeset recognized scalars; keep custom values and array strings literal."""
    from ._numerics import is_numpy_object

    if type(value) is Fraction and not spec and style == 'latex':
        return rf'\frac{{{value.numerator}}}{{{value.denominator}}}'
    # Do not reinterpret the output of user-defined numeric subclasses. NumPy
    # scalars are numbers, while even a zero-dimensional ndarray is not one.
    numeric = type(value) in (int, float, Decimal, Fraction) or (
        is_numpy_object(value) and isinstance(value, Number)
    )
    if numeric:
        match = _DECIMAL_NUMBER.fullmatch(text)
        if match and match[2] is not None:
            mantissa, exponent = match[1], str(int(match[2]))
            if style == 'latex':
                return rf'{mantissa}\times 10^{{{exponent}}}'
            return f'{mantissa}&times;10<sup>{exponent}</sup>'
        special = text.lower()
        if special in ('inf', '+inf', '-inf', 'infinity', '+infinity', '-infinity'):
            sign = text[0] if text[0] in '+-' else ''
            return sign + (r'\infty' if style == 'latex' else '&infin;')
        if special in ('nan', '+nan', '-nan'):
            sign = text[0] if text[0] in '+-' else ''
            return sign + (r'\mathrm{NaN}' if style == 'latex' else 'NaN')
    return latex_text(text) if style == 'latex' else escape(text)


def quantity_body(
    value: object, unit: Unit | AffineUnit, spec: str, style: Style, *, fraction: bool = True
) -> str:
    from .affineunit import AffineUnit
    from .factor import one
    from .unit import ScaledUnit

    unit = _display_unit(unit)
    # Delegate formatting without coercing Decimal, Fraction, arrays or custom scalars.
    text = format_magnitude(value, spec)
    magnitude = rich_magnitude(value, text, spec, style)
    if style == 'html':
        magnitude = f'<span style="white-space: pre-wrap">{magnitude}</span>'

    if (
        not isinstance(unit, AffineUnit)
        and not unit._monomial.components()
        and (not isinstance(unit, ScaledUnit) or unit.scale == one)
    ):
        return magnitude
    body = unit_body(unit, style, fraction=fraction)
    if isinstance(unit, ScaledUnit) and unit.scale != one:
        body = rf'\left({body}\right)' if style == 'latex' else f'({body})'
    separator = r'\,' if style == 'latex' else '&thinsp;'
    return magnitude + separator + body


def html_span(body: str) -> str:
    return f'<span class="unitx">{body}</span>'


def siunitx_unit_parts(unit: Unit | AffineUnit) -> tuple[str, bool]:
    """Return a unit argument and whether siunitx should parse it."""
    from .affineunit import AffineUnit

    unit = _display_unit(unit)
    if isinstance(unit, AffineUnit):
        return '{' + unit.to_latex() + '}', False
    from .factor import one
    from .unit import ScaledUnit

    scaled = isinstance(unit, ScaledUnit) and unit.scale != one
    components = [(str(base), exp) for base, exp in unit._monomial.components()]
    # Literal siunitx syntax uses dots as products and signed exact powers.
    # Other spellings remain literal TeX text, not invented control sequences.
    if not scaled and all(base.isascii() and base.isalpha() for base, _ in components):
        return '.'.join(base if exp == 1 else f'{base}^{{{exp}}}' for base, exp in components), True
    body = unit_body(unit, 'latex')
    # Even with parsing disabled, siunitx's literal formatter splits top-level
    # slashes. Protect the already-rendered expression as one TeX group.
    return '{' + body + '}', False


def siunitx_quantity(value: object, unit: Unit | AffineUnit, spec: str) -> str:
    from .factor import one
    from .unit import ScaledUnit

    unit = _display_unit(unit)
    text = format_magnitude(value, spec)
    options: list[str] = []
    # Recognize only ordinary ASCII decimal/scientific notation. All other
    # magnitude strings are escaped and printed without numerical parsing.
    if _DECIMAL_NUMBER.fullmatch(text):
        number = text
    else:
        options.append('parse-numbers=false')
        number = rich_magnitude(value, text, spec, 'latex')
    body, parsed = siunitx_unit_parts(unit)
    if not parsed:
        options.append('parse-units=false')
    if isinstance(unit, ScaledUnit) and unit.scale != one:
        body = rf'\left({body}\right)'
    option_text = '[' + ','.join(options) + ']' if options else ''
    if not body:
        return rf'\num{option_text}{{{number}}}'
    return rf'\qty{option_text}{{{number}}}{{{body}}}'


def text_quantity(value: object, unit: Unit | AffineUnit) -> str:
    unit = _display_unit(unit)
    unit_text = str(unit)
    if unit_text == '1':
        return str(value)
    if unit_text.startswith('1/'):
        return f'{value}{unit_text[1:]}'
    from ._unicode import DOT

    separator = DOT if unit_text[0].isdecimal() else ' '
    return f'{value}{separator}{unit_text}'


def format_quantity(value: object, unit: Unit | AffineUnit, format_spec: str) -> str:
    from .factor import one

    unit = _display_unit(unit)
    if not format_spec:
        return text_quantity(value, unit)
    # A semicolon used as a Python fill character belongs to the numeric spec.
    offset = 2 if len(format_spec) > 1 and format_spec[1] in '<>=^' else 0
    split = format_spec.find(';', offset)
    if split < 0:
        numeric, style = format_spec, 'text'
    else:
        numeric, style = format_spec[:split], format_spec[split + 1 :]
    if style == 'latex':
        return quantity_body(value, unit, numeric, 'latex')
    if style == 'html':
        return html_span(quantity_body(value, unit, numeric, 'html'))
    if style == 'siunitx':
        return siunitx_quantity(value, unit, numeric)
    if style != 'text':
        raise ValueError(f'Unknown Quantity display option: {style!r}.')
    if not numeric:
        return text_quantity(value, unit)
    text = format_magnitude(value, numeric)
    unit_text = str(unit)
    if unit_text == '1':
        return text
    from .unit import ScaledUnit

    if isinstance(unit, ScaledUnit) and unit.scale != one:
        return f'{text} ({unit_text})'
    if unit_text.startswith('1/'):
        return f'{text}{unit_text[1:]}'
    return f'{text} {unit_text}'
