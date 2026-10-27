"""Lazy delegation of explicitly declared methods to the NumPy adapters."""

from collections.abc import Callable
from functools import wraps
from inspect import signature
from typing import Any, cast

from .numpy_typing import _UNSET


def numpy_method[**P, R](method: Callable[P, R]) -> Callable[P, R]:
    """Use the declaration's signature/defaults and the existing NumPy dispatch.

    Keep this private: each public method still declares its own typed contract.
    NumPy is imported only when a method is called on a NumPy magnitude.
    """
    parameters = signature(method)

    @wraps(method)
    def call(*args: P.args, **kwargs: P.kwargs) -> R:
        bound = parameters.bind(*args, **kwargs)
        bound.apply_defaults()
        quantity: Any = bound.arguments.pop('self')
        quantity._array_value()
        from . import numpy

        options = {name: value for name, value in bound.arguments.items() if value is not _UNSET}
        function = getattr(numpy.np, method.__name__)
        return cast(R, numpy.dispatch_function(function, (), (quantity,), options))

    return call
