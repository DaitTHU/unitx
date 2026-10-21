from importlib.util import find_spec

import pytest

from unitx._numerics import as_fraction
from unitx.exceptions import InexactFloatWarning
from unitx.factor import Factor

# Static-only fixtures run through Pyright in tests/test_typing.py.
collect_ignore = ['tests/example_typing', 'tests/numpy_typing']
if find_spec('numpy') is None:
    collect_ignore.append('src/unitx/_numpy.py')

_DOCTEST_FILTER: dict[str, tuple[pytest.MarkDecorator, ...]] = {
    f'{obj.__module__}.{obj.__qualname__}': tuple(
        pytest.mark.filterwarnings(f'ignore::{warning.__module__}.{warning.__qualname__}')
        for warning in warnings
    )
    for obj, warnings in ((as_fraction, (InexactFloatWarning,)), (Factor, (InexactFloatWarning,)))
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply configured warning filters to selected doctests."""
    for item in items:
        if isinstance(item, pytest.DoctestItem):
            for mark in _DOCTEST_FILTER.get(item.name, ()):
                item.add_marker(mark)
