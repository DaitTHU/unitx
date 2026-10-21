"""Shared attribute protection must preserve the compact instance layout."""

import pytest

from unitx.dimension import Dimension
from unitx.factor import PI
from unitx.monomial import Monomial
from unitx.singleunit import SingleUnit


@pytest.mark.parametrize(
    'value',
    [
        Monomial({'x': 1}),
        Dimension(L=1),
        PI,
        SingleUnit('km'),
    ],
)
def test_immutable_slots(value: object) -> None:
    assert not hasattr(value, '__dict__')
