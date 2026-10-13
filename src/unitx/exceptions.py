"""Custom exceptions and warnings for the unitx library."""

__all__ = ['InexactFloatWarning', 'UnitxError', 'UnitxWarning']


class UnitxError(Exception):
    """Base class for all exceptions raised by unitx."""


class UnitxWarning(UserWarning):
    """Base class for all warnings raised by unitx."""


class InexactFloatWarning(UnitxWarning):
    """Warning for inexact float to Fraction conversion."""
