"""Public display contracts, escaping and numerical independence."""

import pickle
import subprocess
import sys
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from unitx.factor import Factor
from unitx.quantity import Quantity
from unitx.unit import ScaledUnit, Unit


def test_structural_unit_rendering():
    unit = Unit('kg m / (s^2 A)')
    assert unit.to_latex() == r'\text{kg}\,\text{m}/(\text{s}^{2}\,\text{A})'
    assert unit.to_html() == '<span class="unitx">kg&thinsp;m/(s<sup>2</sup>&thinsp;A)</span>'
    assert Unit('m^(2/3)').to_latex() == r'\text{m}^{2/3}'
    assert Unit('m^(-1/2)').to_html() == '<span class="unitx">1/m<sup>1/2</sup></span>'
    assert Unit().to_latex() == '1'


@pytest.mark.parametrize('unit', [Unit(), Unit('m/s^2'), ScaledUnit('m', Factor(Fraction(2, 3)))])
def test_notebook_protocol(unit):
    for obj in (unit, Quantity(Fraction(2, 3), unit)):
        assert obj._repr_latex_() == '$' + obj.to_latex() + '$'
        assert obj._repr_html_() == obj.to_html()
        restored = pickle.loads(pickle.dumps(obj))
        assert restored == obj
        assert repr(restored) == repr(obj)


def test_scaled_unit_keeps_exact_scale():
    unit = ScaledUnit('m', Factor(Fraction(2, 3)))
    assert unit.to_latex() == r'\text{2}/\text{3}\,\text{m}'
    assert Quantity(4, unit).to_latex() == (r'\text{4}\,\left(\text{2}/\text{3}\,\text{m}\right)')
    assert Quantity(4, unit).to_html() == (
        '<span class="unitx"><span style="white-space: pre-wrap">4</span>'
        '&thinsp;(2/3&thinsp;m)</span>'
    )
    assert ScaledUnit('', Factor(2)).to_latex() == r'\text{2}'
    assert Quantity(4, ScaledUnit('', Factor(2))).to_latex() == (r'\text{4}\,\left(\text{2}\right)')
    assert ScaledUnit('m', Factor(1)).to_latex() == Unit('m').to_latex()
    assert ScaledUnit('m', Factor(2) ** Fraction(1, 2)).to_latex() == (r'\text{2}^{1/2}\,\text{m}')


@pytest.mark.parametrize('spec', ['', '.2f', '>12.3e', ';text', '.1f;latex', '.1f;html'])
def test_format_numeric_delegation(spec):
    q = Quantity(Decimal('12.345'), 'm/s')
    if not spec:
        assert format(q, spec) == str(q)
    elif ';' not in spec:
        assert format(q, spec) == format(q.value, spec) + ' m/s'
    else:
        numeric, style = spec.split(';')
        expected = {'text': str(q), 'latex': q.to_latex(numeric), 'html': q.to_html(numeric)}
        assert format(q, spec) == expected[style]


def test_native_precision_and_errors():
    q = Quantity(Decimal('12345678901234567890.123456789'), 'm')
    with localcontext() as ctx:
        ctx.prec = 2
        assert format(q, '.9f') == '12345678901234567890.123456789 m'
    assert Quantity(Fraction(2, 3), 'm').to_latex().startswith(r'\frac{2}{3}')
    assert format(Quantity(2, 'm'), ';>5') == ';;;;2 m'
    assert format(Quantity(2, 'm'), ';>5;html') == (
        '<span class="unitx"><span style="white-space: pre-wrap">;;;;2</span>&thinsp;m</span>'
    )
    with pytest.raises(ValueError, match='Unknown'):
        format(q, '.2f;unknown')
    with pytest.raises(ValueError):
        format(q, ';latex;html')
    with pytest.raises(ValueError):
        format(q, 'invalid')
    assert format(Quantity(2, ''), '04d') == '0002'
    assert format(Quantity(2, '1/s'), '04d') == '0002/s'


def test_literal_text_escaping():
    # Low-level escaping also handles characters excluded from registered tokens.
    from unitx._display import latex_text, product

    token = '<script>&"\\{}$_#%^~'
    html = product([(token, Fraction(1))], 'html')
    latex = latex_text(token)
    assert '<script>' not in html
    assert '&lt;script&gt;' in html
    assert r'\textbackslash{}' in latex
    for escaped in [r'\{', r'\}', r'\$', r'\_', r'\%', r'\#', r'\&']:
        assert escaped in latex


def test_numpy_display_keeps_array_and_native_format_boundary():
    np = pytest.importorskip('numpy')
    array = np.arange(12).reshape(3, 4)[:, ::2]
    array.flags.writeable = False
    q = Quantity(array, 'm')
    assert q.value is array
    assert q.to_html() == (
        '<span class="unitx"><span style="white-space: pre-wrap">'
        '[[ 0  2]\n [ 4  6]\n [ 8 10]]</span>&thinsp;m</span>'
    )
    assert format(q, ';latex') == q.to_latex()
    with pytest.raises(TypeError):
        format(q, '.2f')
    np.testing.assert_array_equal(array, np.arange(12).reshape(3, 4)[:, ::2])
    assert q.value is array
    for value, text in [(np.array(2.0), '2.0'), (np.array([]), '[]'), (np.float64(1.234), '1.234')]:
        item = Quantity(value, 'm')
        assert item.to_html() == (
            '<span class="unitx"><span style="white-space: pre-wrap">'
            + text
            + '</span>&thinsp;m</span>'
        )
        assert item.to_latex() == r'\text{' + text + r'}\,\text{m}'
        assert item.value is value


def test_no_optional_imports():
    code = """
import sys
from unitx.quantity import Quantity
from unitx.unit import Unit
q = Quantity(2, Unit('m'))
q.to_latex(); q.to_html(); format(q, '.2f;latex')
q.to_siunitx(); q.unit.to_siunitx(); format(q, '.2f;siunitx')
q.unit.to_latex(fraction=False); q.to_latex('.2f', fraction=False)
assert not any(x in sys.modules for x in ('numpy', 'IPython', 'sympy'))
"""
    subprocess.run([sys.executable, '-c', code], check=True)


def test_custom_magnitude_formatting_and_escaping():
    class Labeled(int):
        def __format__(self, spec):
            return '<b>&$' + spec

    q = Quantity(Labeled(2), 'm')
    assert format(q, 'custom') == '<b>&$custom m'
    assert '&lt;b&gt;&amp;$custom' in q.to_html('custom')
    assert r'\text{<b>\&\$custom}' in q.to_latex('custom')


def test_complex_grouping_and_scale_immutability():
    q = Quantity(1 + 2j, ScaledUnit('m/s', Factor(Fraction(2, 3))))
    original = repr(q)
    assert format(q, '.1f').startswith('(1.0+2.0j) ')
    assert '(1.0+2.0j)' in q.to_latex('.1f')
    assert '(1.0+2.0j)' in q.to_html('.1f')
    assert format(q, ';text') == str(q)
    assert repr(q) == original


def test_ipython_mime_dispatch_when_available():
    ipython = pytest.importorskip('IPython.core.formatters')
    formatter = ipython.DisplayFormatter()
    for obj in (Unit('m/s'), ScaledUnit('m', Factor(2)), Quantity(Fraction(2, 3), 'm')):
        data, _ = formatter.format(obj)
        assert data['text/latex'] == obj._repr_latex_()
        assert data['text/html'] == obj._repr_html_()
        assert 'text/plain' in data


@pytest.mark.parametrize(
    ('spec', 'expected'),
    [
        ('+', '(+1+2j)'),
        ('>20', ' ' * 14 + '(1+2j)'),
        ('<20', '(1+2j)' + ' ' * 14),
        ('*>20', '*' * 14 + '(1+2j)'),
        ('*<20', '(1+2j)' + '*' * 14),
        ('(>20', '(' * 14 + '(1+2j)'),
        (')<20', '(1+2j)' + ')' * 14),
        ('.2', '(1+2j)'),
        ('.2f', '(1.00+2.00j)'),
    ],
)
def test_complex_native_parentheses_and_alignment(spec, expected):
    q = Quantity(1 + 2j, 'm')
    assert format(q, spec) == expected + ' m'
    from html import escape

    assert q.to_html(spec) == (
        '<span class="unitx"><span style="white-space: pre-wrap">'
        + escape(expected)
        + '</span>&thinsp;m</span>'
    )


def test_registered_symbol_escaping():
    from unitx.unitregistry import (
        UNIT_DEFINITIONS,
        UNIT_NAME_TO_SYMBOL,
        UNIT_SYMBOL_ALIAS_TO_SYMBOL,
    )

    tables = UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL
    snapshots = [table.copy() for table in tables]
    try:
        unit = Unit('m').define('disp<&_$', 'display literal unit', prefixable=False)
        assert unit.to_html() == '<span class="unitx">disp&lt;&amp;_$</span>'
        assert unit.to_latex() == r'\text{disp<\&\_\$}'
        q = Quantity(2, unit)
        assert q.to_latex() == r'\text{2}\,\text{disp<\&\_\$}'
        assert q.to_html() == (
            '<span class="unitx"><span style="white-space: pre-wrap">2</span>'
            '&thinsp;disp&lt;&amp;_$</span>'
        )
    finally:
        for table, snapshot in zip(tables, snapshots):
            table.clear()
            table.update(snapshot)


def test_scaled_dimensionless_unit_is_not_hidden():
    q = Quantity(2, 'km/m')
    assert q.to_latex() == r'\text{2}\,\text{km}/\text{m}'
    assert q.to_html() == (
        '<span class="unitx"><span style="white-space: pre-wrap">2</span>&thinsp;km/m</span>'
    )
    assert q.value == 2
    assert q.unit == Unit('km/m')


@pytest.mark.parametrize(
    ('value', 'spec', 'latex', 'html'),
    [
        (1.234e-6, '.2e', r'1.23\times 10^{-6}', '1.23&times;10<sup>-6</sup>'),
        (-1.234e6, '+.2E', r'-1.23\times 10^{6}', '-1.23&times;10<sup>6</sup>'),
        (Decimal('1e999'), '+.3e', r'+1.000\times 10^{999}', '+1.000&times;10<sup>999</sup>'),
        (float('inf'), '', r'\infty', '&infin;'),
        (float('-inf'), '', r'-\infty', '-&infin;'),
        (Decimal('Infinity'), '+', r'+\infty', '+&infin;'),
        (float('nan'), '', r'\mathrm{NaN}', 'NaN'),
        (Decimal('-NaN'), '', r'-\mathrm{NaN}', '-NaN'),
    ],
)
def test_scientific_and_special_real_magnitudes(value, spec, latex, html):
    q = Quantity(value, 'm')
    assert q.to_latex(spec) == latex + r'\,\text{m}'
    assert q.to_html(spec) == (
        '<span class="unitx"><span style="white-space: pre-wrap">'
        + html
        + '</span>&thinsp;m</span>'
    )
    assert format(q, spec + ';latex') == q.to_latex(spec)
    assert format(q, spec + ';html') == q.to_html(spec)


def test_unit_format_selection_and_alignment():
    unit = Unit('kg m/(s^2 A)')
    assert format(unit, '') == str(unit)
    assert format(unit, 'text') == str(unit)
    assert format(unit, '>30') == format(str(unit), '>30')
    assert format(unit, 'latex') == unit.to_latex()
    assert unit.to_latex(fraction=False) == r'\text{kg}\,\text{m}\,\text{s}^{-2}\,\text{A}^{-1}'
    assert format(unit, 'html') == unit.to_html()
    assert format(unit, 'siunitx') == unit.to_siunitx()
    with pytest.raises(ValueError):
        format(unit, 'unknown')
    with pytest.raises(ValueError):
        format(unit, '.2f')


def test_inline_fractional_powers_and_extra_scale():
    unit = ScaledUnit('m^(1/3)/s', Factor(Fraction(2, 3)))
    expected = r'\text{2}\,\text{3}^{-1}\,\text{m}^{1/3}\,\text{s}^{-1}'
    assert unit.to_latex(fraction=False) == expected
    q = Quantity(Fraction(2, 3), unit)
    expected_q = r'\frac{2}{3}\,\left(' + expected + r'\right)'
    assert q.to_latex(fraction=False) == expected_q
    assert Unit().to_latex(fraction=False) == '1'
    assert Quantity(2, '').to_latex(fraction=False) == r'\text{2}'
    assert unit.scale == Factor(Fraction(2, 3))


def test_rich_numbers_respect_custom_formatting_and_padding():
    class LiteralFloat(float):
        def __format__(self, spec):
            return '1e9'

    class LiteralFraction(Fraction):
        def __format__(self, spec):
            return 'custom fraction'

    assert Quantity(LiteralFloat(2), '').to_latex() == r'\text{1e9}'
    assert Quantity(LiteralFraction(2, 3), '').to_latex() == r'\text{custom fraction}'
    assert Quantity(LiteralFraction(2, 3), '').to_siunitx() == (
        r'\num[parse-numbers=false]{\text{custom fraction}}'
    )
    assert Quantity(1.234e-6, '').to_latex('>14.2e') == r'\text{      1.23e-06}'
    assert Quantity(0.123, '').to_latex('.2%') == r'\text{12.30\%}'
    assert Quantity(0.0, '').to_latex('.1e') == r'0.0\times 10^{0}'


def test_numpy_scalars_and_arrays_have_distinct_rich_boundaries():
    np = pytest.importorskip('numpy')
    assert Quantity(np.float64(1.234e-6), '').to_latex('.2e') == r'1.23\times 10^{-6}'
    assert Quantity(np.float32('inf'), '').to_latex() == r'\infty'
    data = np.array([1.234567, 2.345678])
    q = Quantity(data, 'm')
    original_options = np.get_printoptions()
    with np.printoptions(precision=2):
        assert q.to_latex(fraction=False) == r'\text{[1.23 2.35]}\,\text{m}'
    assert np.get_printoptions() == original_options
    assert q.value is data


@pytest.mark.parametrize('unit', [Unit(), ScaledUnit('', Factor(1))])
def test_identity_unit_is_omitted_in_all_quantity_representations(unit):
    q = Quantity(4, unit)
    assert q.to_latex() == r'\text{4}'
    assert q.to_latex(fraction=False) == r'\text{4}'
    assert q.to_html() == '<span class="unitx"><span style="white-space: pre-wrap">4</span></span>'
    assert q.to_siunitx() == r'\num{4}'


@pytest.mark.parametrize('dtype', ['complex64', 'complex128', 'clongdouble'])
def test_numpy_complex_scalars_keep_necessary_grouping(dtype):
    np = pytest.importorskip('numpy')
    value = getattr(np, dtype)(1 + 2j)
    q = Quantity(value, 'm')
    assert format(q, '.1f') == '(1.0+2.0j) m'
    assert q.to_latex('.1f') == r'\text{(1.0+2.0j)}\,\text{m}'
    assert q.to_siunitx('.1f') == r'\qty[parse-numbers=false]{\text{(1.0+2.0j)}}{m}'
    assert q.value is value


@pytest.mark.parametrize(
    ('source', 'expected'),
    [
        ('m/s^2', r'\text{m}/\text{s}^{2}'),
        ('1/s', r'1/\text{s}'),
        ('m/(s A)', r'\text{m}/(\text{s}\,\text{A})'),
        ('m^(1/2)/s^(1/3)', r'\text{m}^{1/2}/\text{s}^{1/3}'),
    ],
)
def test_latex_defaults_to_slash_unit_fractions(source, expected):
    unit = Unit(source)
    q = Quantity(Fraction(2, 3), unit)
    assert unit.to_latex() == expected
    assert format(unit, 'latex') == expected
    assert unit._repr_latex_() == '$' + expected + '$'
    assert q.to_latex() == r'\frac{2}{3}\,' + expected
    assert format(q, ';latex') == q.to_latex()
    assert q._repr_latex_() == '$' + q.to_latex() + '$'
    # Unit layout never controls the exact Fraction magnitude representation.
    assert q.to_latex(fraction=False).startswith(r'\frac{2}{3}\,')


def test_removed_latex_inline_format_is_rejected():
    unit = Unit('m/s')
    with pytest.raises(ValueError):
        format(unit, 'latex_inline')
    with pytest.raises(ValueError, match='Unknown Quantity display option'):
        format(Quantity(2, unit), '.2f;latex_inline')
