import pytest

from unitx._numerics import as_fraction
from unitx.exceptions import InexactFloatWarning

_DOCTEST_FILTER: dict[str, tuple[pytest.MarkDecorator, ...]] = {
    f'{as_fraction.__module__}.{as_fraction.__qualname__}': (
        pytest.mark.filterwarnings(
            f'ignore::{InexactFloatWarning.__module__}.{InexactFloatWarning.__qualname__}'
        ),
    ),
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply configured warning filters to selected doctests."""
    for item in items:
        for mark in _DOCTEST_FILTER.get(item.name, ()):
            item.add_marker(mark)
