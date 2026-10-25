"""Unicode unit spellings retain canonical identity and coordinate semantics."""

from __future__ import annotations

import copy
import pickle
from fractions import Fraction

import pytest

from unitx import AffineQuantity, Unit
from unitx.imperial import imperial
from unitx.si import si


@pytest.mark.parametrize(
    ('alias', 'canonical'),
    [
        ('℃', 'degC'),
        ('K', 'K'),
        ('Ω', 'Ω'),
        ('ℓ', 'L'),
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


@pytest.mark.parametrize(('alias', 'canonical'), [('m㍴', 'mbar'), ('kΩ', 'kΩ'), ('mℓ', 'mL')])
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
