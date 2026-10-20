"""siunitx v3 output preserves the stored coordinates and exact expressions."""

from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from unitx.factor import PI, Factor
from unitx.quantity import Quantity
from unitx.unit import ScaledUnit, Unit


@pytest.mark.parametrize(
    ('source', 'expected'),
    [
        ('', r'\unit{1}'),
        ('kg m/s^2', r'\unit{kg.m.s^{-2}}'),
        ('m^(1/3)', r'\unit{m^{1/3}}'),
        ('1/(s A)', r'\unit{s^{-1}.A^{-1}}'),
        ('km/m', r'\unit{km.m^{-1}}'),
        ('um', r'\unit[parse-units=false]{{\text{µm}}}'),
    ],
)
def test_units(source, expected):
    assert Unit(source).to_siunitx() == expected


@pytest.mark.parametrize(
    ('value', 'spec', 'expected'),
    [
        (12.3456, '.3f', r'\qty{12.346}{m.s^{-1}}'),
        (Decimal('1.234e-30'), '.3e', r'\qty{1.234e-30}{m.s^{-1}}'),
        (Fraction(2, 3), '', r'\qty[parse-numbers=false]{\frac{2}{3}}{m.s^{-1}}'),
        (1 + 2j, '.1f', r'\qty[parse-numbers=false]{\text{(1.0+2.0j)}}{m.s^{-1}}'),
        (float('inf'), '', r'\qty[parse-numbers=false]{\infty}{m.s^{-1}}'),
        (float('nan'), '', r'\qty[parse-numbers=false]{\mathrm{NaN}}{m.s^{-1}}'),
        (2, ';>5', r'\qty[parse-numbers=false]{\text{;;;;2}}{m.s^{-1}}'),
        (2, '04d', r'\qty{0002}{m.s^{-1}}'),
    ],
)
def test_quantity_and_format(value, spec, expected):
    q = Quantity(value, 'm/s')
    assert q.to_siunitx(spec) == expected
    assert format(q, spec + ';siunitx') == expected


def test_dimensionless_and_scale():
    assert Quantity(2, '').to_siunitx() == r'\num{2}'
    assert Quantity(Fraction(2, 3), '').to_siunitx() == (r'\num[parse-numbers=false]{\frac{2}{3}}')
    assert Quantity(2, 'km/m').to_siunitx() == r'\qty{2}{km.m^{-1}}'
    assert ScaledUnit('m', Factor(1)).to_siunitx() == r'\unit{m}'
    unit = ScaledUnit('m', Factor(Fraction(2, 3)))
    assert unit.to_siunitx() == r'\unit[parse-units=false]{{\text{2}/\text{3}\,\text{m}}}'
    assert Quantity(4, unit).to_siunitx() == (
        r'\qty[parse-units=false]{4}{\left({\text{2}/\text{3}\,\text{m}}\right)}'
    )
    assert Quantity(Fraction(2, 3), unit).to_siunitx() == (
        r'\qty[parse-numbers=false,parse-units=false]{\frac{2}{3}}'
        r'{\left({\text{2}/\text{3}\,\text{m}}\right)}'
    )
    assert Quantity(4, ScaledUnit('', PI)).to_siunitx() == (
        r'\qty[parse-units=false]{4}{\left({\text{π}}\right)}'
    )
    assert unit.scale == Factor(Fraction(2, 3))


def test_exact_decimal_and_errors():
    q = Quantity(Decimal('12345678901234567890.123456789'), 'm')
    with localcontext() as ctx:
        ctx.prec = 2
        assert q.to_siunitx('.9f') == r'\qty{12345678901234567890.123456789}{m}'
    with pytest.raises(ValueError):
        format(q, '.2f;siunitx;html')
    with pytest.raises(ValueError):
        q.to_siunitx('invalid')


def test_custom_unit_and_number_escaping():
    from unitx.unitregistry import (
        UNIT_DEFINITIONS,
        UNIT_NAME_TO_SYMBOL,
        UNIT_SYMBOL_ALIAS_TO_SYMBOL,
    )

    tables = UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL
    snapshots = [table.copy() for table in tables]
    try:
        unit = Unit('m').define('sifmt_$', 'siunitx display token', prefixable=False)
        assert unit.to_siunitx() == r'\unit[parse-units=false]{{\text{sifmt\_\$}}}'

        class Label(int):
            def __format__(self, spec):
                return r'\input{secret}$%'

        q = Quantity(Label(2), unit)
        assert q.to_siunitx() == (
            r'\qty[parse-numbers=false,parse-units=false]'
            r'{\text{\textbackslash{}input\{secret\}\$\%}}{{\text{sifmt\_\$}}}'
        )
    finally:
        for table, snapshot in zip(tables, snapshots):
            table.clear()
            table.update(snapshot)


def test_array_boundary():
    np = pytest.importorskip('numpy')
    data = np.array([1.0, 2.0])
    data.flags.writeable = False
    q = Quantity(data, 'm')
    assert q.to_siunitx() == r'\qty[parse-numbers=false]{\text{[1. 2.]}}{m}'
    with pytest.raises(TypeError):
        q.to_siunitx('.2f')
    assert q.value is data
    np.testing.assert_array_equal(data, [1.0, 2.0])


def test_generated_ascii_output_compiles_when_tex_available(tmp_path):
    import shutil
    import subprocess

    engine = shutil.which('pdflatex')
    kpsewhich = shutil.which('kpsewhich')
    if engine is None or kpsewhich is None:
        pytest.skip('TeX is optional')
    package = subprocess.run([kpsewhich, 'siunitx.sty'], capture_output=True, text=True, timeout=10)
    if package.returncode != 0:
        pytest.skip('siunitx is optional')
    objects = [
        Unit(),
        Unit('kg m/s^2'),
        Unit('m^(1/3)'),
        Quantity(1.234e-30, 'm/s'),
        Quantity(Fraction(2, 3), 'm'),
        Quantity(1 + 2j, 'm'),
        Quantity(float('inf'), 'm'),
        Quantity(2, ''),
        Quantity(Fraction(2, 3), ScaledUnit('m', Factor(Fraction(2, 3)))),
        ScaledUnit('m/(s A)', Factor(Fraction(2, 3))),
        Quantity(4, ScaledUnit('m/(s A)', Factor(Fraction(2, 3)))),
    ]
    source = (
        r'\documentclass{article}\usepackage{amsmath,siunitx}\begin{document}'
        + '\n'.join(obj.to_siunitx() + r'\par' for obj in objects)
        + '$'
        + Quantity(Decimal('1e300'), 'm/s').to_latex('.2e', fraction=False)
        + r'$\par'
        + '$'
        + Unit('m^(1/3)/s').to_latex(fraction=False)
        + r'$\par'
        + '$'
        + Unit('kg m/(s^2 A)').to_latex()
        + r'$\par'
        + '$'
        + Quantity(Fraction(2, 3), 'm/(s A)').to_latex()
        + r'$\par'
        + r'\end{document}'
    )
    (tmp_path / 'display.tex').write_text(source)
    result = subprocess.run(
        [engine, '-interaction=nonstopmode', '-halt-on-error', '-no-shell-escape', 'display.tex'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Missing character:' not in result.stdout
    assert (tmp_path / 'display.pdf').is_file()
