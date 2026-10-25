"""Scalar quantities reject iterable protocols with ordinary Python errors."""

from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity


@pytest.mark.parametrize('q', [Quantity(1, 'm'), AffineQuantity(1, 'degC'), LogQuantity(1, 'dBm')])
def test_quantity_scalar_invalid_iteration(q: Any) -> None:
    with pytest.raises(TypeError, match='not iterable'):
        iter(q)


@pytest.mark.parametrize('q', [Quantity(1, 'm'), AffineQuantity(1, 'degC'), LogQuantity(1, 'dBm')])
def test_quantity_scalar_invalid_length(q: Any) -> None:
    with pytest.raises(TypeError, match='no len'):
        len(q)
