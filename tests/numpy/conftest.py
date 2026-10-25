"""Samples shared by array preparation and indexing tests."""

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity, Unit
from unitx._numerics import Array
from unitx.factor import Factor, pi

ArrayQuantity = Quantity[Array] | AffineQuantity[Array] | LogQuantity[Array]


@pytest.fixture(params=['linear', 'affine', 'logarithmic'])
def array_sample(request: pytest.FixtureRequest) -> ArrayQuantity:
    np = pytest.importorskip('numpy')
    values = np.arange(24, dtype=np.float64).reshape(2, 3, 4)[:, :, ::2]
    if request.param == 'linear':
        return Quantity(values, Factor(2) * pi * Unit('cm'))
    if request.param == 'affine':
        return AffineQuantity(values, 'degC')
    return LogQuantity(values, 'dBm')
