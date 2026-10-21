"""Shared protection against instance attribute assignment and deletion."""


class _Immutable:
    """Block ordinary attribute writes; contained values may still be mutable."""

    __slots__ = ()

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f'{type(self).__name__} objects are immutable.')

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f'{type(self).__name__} objects are immutable.')
