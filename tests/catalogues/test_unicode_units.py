"""Unicode unit spellings retain canonical identity and coordinate semantics."""

from __future__ import annotations

import copy
import pickle
from fractions import Fraction

import pytest

from unitx import AffineQuantity, Unit
from unitx.exceptions import UnitSymbolError, UnitSyntaxError
from unitx.imperial import imperial
from unitx.serialization import from_dict, to_dict
from unitx.si import si


@pytest.mark.parametrize(
    ('alias', 'canonical'),
    [
        ('℃', 'degC'),
        ('K', 'K'),
        ('Ω', 'Ω'),
        ('ℓ', 'L'),
    ],
)
def test_unit_unicode_identity(alias: str, canonical: str) -> None:
    unit, expected = Unit(alias), Unit(canonical)
    assert unit == expected
    assert hash(unit) == hash(expected)
    assert {unit: 'same'}[expected] == 'same'
    assert unit.expression() == expected.expression()
    assert copy.copy(unit) == pickle.loads(pickle.dumps(unit)) == expected
    assert Unit(f'{alias}/s') == expected / si.s


@pytest.mark.parametrize(('alias', 'canonical'), [('kΩ', 'kΩ'), ('mℓ', 'mL')])
def test_unit_unicode_prefix(alias: str, canonical: str) -> None:
    assert Unit(alias) == Unit(canonical)


def test_affine_quantity_unicode_coordinate() -> None:
    assert AffineQuantity(0, '℃').value_in('K') == Fraction(5463, 20)
    assert AffineQuantity(32, imperial.unit('℉')) == AffineQuantity(0, '℃')
    assert AffineQuantity(273, 'K') == AffineQuantity(273, 'K')


def test_imperial_unicode_coordinate() -> None:
    unit = imperial.unit('℉')
    assert unit == imperial.unit('degF')
    assert hash(unit) == hash(imperial.unit('°F'))
    assert pickle.loads(pickle.dumps(unit)) == unit
    assert imperial.unit('℉/s') == unit / si.s


@pytest.mark.parametrize(
    ('glyph', 'expression'),
    [
        ('㍳', 'au'),
        ('㍴', 'bar'),
        ('㍶', 'pc'),
        ('㎈', 'cal'),
        ('㎐', 'Hz'),
        ('㎩', 'Pa'),
        ('㎭', 'rad'),
        ('㏃', 'Bq'),
        ('㏅', 'cd'),
        ('㏉', 'Gy'),
        ('㏊', 'ha'),
        ('㏏', 'kn'),
        ('㏐', 'lm'),
        ('㏓', 'lx'),
        ('㏖', 'mol'),
        ('㏛', 'sr'),
        ('㏜', 'Sv'),
        ('㏝', 'Wb'),
        ('㎏', 'kg'),
        ('㎍', 'ug'),
        ('㎞', 'km'),
        ('㎛', 'um'),
        ('㎡', 'm2'),
        ('㎣', 'mm3'),
        ('㎦', 'km3'),
        ('㍸', 'dm2'),
        ('㎧', 'm/s'),
        ('㎨', 'm/s2'),
        ('㎮', 'rad/s'),
        ('㎯', 'rad/s2'),
        ('㍱', 'hPa'),
        ('㎫', 'MPa'),
        ('㏀', 'kΩ'),
        ('㏁', 'MΩ'),
        ('㎀', 'pA'),
        ('㎄', 'kA'),
        ('㎌', 'uF'),
        ('㎉', 'kcal'),
        ('㎔', 'THz'),
        ('㎕', 'uL'),
        ('㎘', 'kL'),
        ('㎙', 'fm'),
        ('㎰', 'ps'),
        ('㎲', 'us'),
        ('㎶', 'uV'),
        ('㎹', 'MV'),
        ('㎺', 'pW'),
        ('㎼', 'uW'),
        ('㎿', 'MW'),
        ('㏆', 'C/kg'),
        ('㏈', 'dB'),
        ('㏙', 'ppm'),
        ('㏞', 'V/m'),
        ('㏟', 'A/m'),
    ],
)
def test_unit_from_unicode_identity(glyph: str, expression: str) -> None:
    unit, expected = Unit.from_unicode(glyph), Unit(expression)
    assert unit == expected
    assert hash(unit) == hash(expected)
    assert unit.expression() == expected.expression()
    assert Unit.from_unicode(glyph + '/s') == expected / si.s
    assert copy.copy(unit) == copy.deepcopy(unit) == pickle.loads(pickle.dumps(unit)) == expected
    assert from_dict(to_dict(unit)) == expected
    with pytest.raises(UnitSymbolError):
        Unit(glyph)


@pytest.mark.parametrize(
    'text', ['k㏖', 'µ㏖', 'u㏖', 'm㍴', 'k㎏', '㎏m', '㎏㎡', 'kilomole㏖', '㎆', '㏋', '㏗', '㌃']
)
def test_unit_from_unicode_token_invalid(text: str) -> None:
    with pytest.raises(UnitSymbolError, match='standalone Unicode unit character'):
        Unit.from_unicode(text)
    with pytest.raises(UnitSymbolError):
        Unit(text)


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('㎏/㎡', 'kg/m2'),
        ('m/㎧', 's'),
        ('㎨²', 'm2/s4'),
        ('㎨^-2', 's4/m2'),
        ('㎧^(1/2)', '(m/s)^(1/2)'),
        ('㎧ ㏖', 'm/s*mol'),
    ],
)
def test_unit_from_unicode_grouping(text: str, expected: str) -> None:
    assert Unit.from_unicode(text) == Unit(expected)


@pytest.mark.parametrize('text', ['', 'µmol', 'Ω', 'm²', '℃', '[metre] / s', 'm^(1/2)'])
def test_unit_from_unicode_ordinary_expression(text: str) -> None:
    assert Unit.from_unicode(text) == Unit(text)


@pytest.mark.parametrize('text', ['㎏^', '㎧**(1/0)', '[㎏]', '㎨ **²'])
def test_unit_from_unicode_syntax_invalid(text: str) -> None:
    with pytest.raises((UnitSyntaxError, UnitSymbolError)):
        Unit.from_unicode(text)


@pytest.mark.parametrize('value', [None, 1, si.kg])
def test_unit_from_unicode_type_invalid(value: object) -> None:
    with pytest.raises(TypeError, match='Unit expression must be str'):
        Unit.from_unicode(value)  # pyright: ignore[reportArgumentType]
