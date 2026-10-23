"""Bound linear algebra remains independent of coordinate implementations."""

import builtins
import gc
import pickle
from decimal import Decimal
from fractions import Fraction

import pytest

from unitx import AffineQuantity, AffineUnit, LogQuantity, LogUnit, Quantity, Registry
from unitx.si import si


def test_quantity_linear_operations_without_coordinates(monkeypatch):
    original_import = builtins.__import__
    coordinate_modules = {
        '_coordinates',
        'resolve',
        'logarithms',
        'points',
        'definition',
        'affine',
        'logarithmic',
        'affineunit',
        'affinequantity',
        'logunit',
        'logquantity',
    }

    def independent_import(name, *args, **kwargs):
        if name.rsplit('.', 1)[-1] in coordinate_modules:
            raise AssertionError(f'Linear algebra imported coordinate implementation {name}')
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(builtins, '__import__', independent_import)
        distance = Quantity(Decimal('1.25'), si.m) + 25 * si.cm
        speed = distance / (2 * si.s)
        assert speed.value_in('cm/s') == Decimal(75)
        assert speed == Quantity(Decimal('0.75'), si.m / si.s)
        assert hash(speed) == hash(Quantity(Fraction(3, 4), si.m / si.s))
        assert 1 * si.dB != 1 * si.one
        assert (si.dB / si.Np).dimension is si.one.dimension


@pytest.mark.parametrize('family', ['affine', 'logarithmic'])
def test_coordinate_linear_projection_retains_private_mapping(family):
    def project(reference):
        coordinate = (
            AffineUnit(reference * si.K)
            if family == 'affine'
            else LogUnit(reference * si.W, delta_unit=si.dB, kind='power')
        ).define('privateReading')
        return Registry([coordinate]).parse('privateReading')

    first, second = project(10), project(20)
    gc.collect()
    reading_class = AffineQuantity if family == 'affine' else LogQuantity
    first_reading, second_reading = reading_class(0, first), reading_class(0, second)
    physical = si.K if family == 'affine' else si.W
    assert first != second
    assert first_reading.to_quantity(physical) == 10 * physical
    assert second_reading.to_quantity(physical) == 20 * physical
    assert pickle.loads(pickle.dumps(first_reading)) == first_reading


def test_affine_unit_invalid_symbolic_origin_normalization():
    from unitx import pi

    with pytest.raises(TypeError, match='requires a symbolic sum for its origin'):
        AffineUnit(1 * si.m, at=1, delta_unit=pi * si.m)
    coordinate = AffineUnit(1 * si.m, delta_unit=pi * si.m)
    assert coordinate.origin == 1 * si.m
    assert coordinate.delta_unit == pi * si.m


def test_affine_unit_invalid_symbolic_reference_normalization():
    from unitx import pi

    coordinate = AffineUnit(1 * si.m, delta_unit=pi * si.m)
    with pytest.raises(TypeError, match='requires? a symbolic sum for exact physical evaluation'):
        AffineUnit(1 @ coordinate)
    assert AffineUnit(0 @ coordinate).origin == coordinate.origin
