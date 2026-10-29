"""Lazy delegation of explicitly declared methods to the NumPy adapters."""

from typing import Any

from .numpy_typing import _UNSET


def call_numpy_method(quantity: Any, name: str, /, **options: Any) -> Any:
    """Forward already bound method arguments to the existing NumPy dispatch."""
    quantity._array_value()
    from . import numpy

    options = {key: value for key, value in options.items() if value is not _UNSET}
    return numpy.dispatch_function(getattr(numpy.np, name), (), (quantity,), options)
