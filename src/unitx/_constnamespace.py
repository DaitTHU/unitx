"""Class-based namespaces for constants accessed directly as class attributes.

Namespaces cannot be instantiated, and their attributes cannot be rebound or
deleted. Objects referenced by those attributes are not recursively frozen.
"""

from typing import Any, NoReturn

__all__ = ['ConstNamespace']


class _ConstNamespaceMeta(type):
    def __call__(metacls, *args: Any, **kwds: Any) -> NoReturn:
        """
        This namespace cannot be instantiated; access constants via class attributes.
        """
        raise TypeError(metacls.__call__.__doc__)

    def __setattr__(cls, name, value) -> NoReturn:
        """
        Constant namespace attributes cannot be assigned.
        """
        raise AttributeError(cls.__class__.__setattr__.__doc__)

    def __delattr__(cls, name) -> NoReturn:
        """
        Constant namespace attributes cannot be deleted.
        """
        raise AttributeError(cls.__class__.__delattr__.__doc__)


class ConstNamespace(metaclass=_ConstNamespaceMeta):
    """Base for constant namespaces with protected class attribute bindings."""
