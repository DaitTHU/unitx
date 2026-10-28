"""External allocation and bare-array writes must not silently discard units."""

from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity
from unitx.exceptions import ImplicitConversionError
from unitx.si import si

np = pytest.importorskip('numpy')


@pytest.mark.parametrize('value', [2.0, np.array(2.0), np.array([2.0, 3.0])])
@pytest.mark.parametrize(
    'cls,unit', [(Quantity, si.m), (AffineQuantity, si.degC), (LogQuantity, si.dBm)]
)
def test_quantity_external_coercion_invalid(value: Any, cls: Any, unit: Any) -> None:
    q = cls(value, unit)
    with pytest.raises(ImplicitConversionError):
        np.full(2, q)
    target = np.full(2, -1.0)
    with pytest.raises(ImplicitConversionError):
        target[:] = q
    np.testing.assert_array_equal(target, [-1, -1])


@pytest.mark.parametrize('value', [2.0, np.array(2.0)])
@pytest.mark.parametrize(
    'cls,unit', [(Quantity, si.m), (AffineQuantity, si.degC), (LogQuantity, si.dBm)]
)
def test_quantity_external_scalar_assignment_invalid(value: Any, cls: Any, unit: Any) -> None:
    target = np.full(2, -1.0)
    # NumPy may translate the rejected scalar conversion into its own ValueError.
    with pytest.raises((ValueError, TypeError)):
        target[0] = cls(value, unit)
    np.testing.assert_array_equal(target, [-1, -1])
