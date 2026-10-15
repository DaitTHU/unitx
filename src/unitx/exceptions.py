"""Custom exceptions and warnings for the unitx library."""

import os

__all__ = [
    'IncompleteFactorWarning',
    'InexactFloatWarning',
    'UnitSymbolError',
    'UnitSymbolWarning',
    'UnitSyntaxError',
    'UnitxError',
    'UnitxWarning',
]


class UnitxError(Exception):
    """Base class for all exceptions raised by unitx."""


class UnitSymbolError(UnitxError):
    """Exception raised for invalid unit symbols or names."""


class UnitSyntaxError(UnitxError):
    """Exception raised for invalid unit syntax."""


# Python 3.12 warnings.warn() new API
_WARNING_SKIP_PREFIXES = (os.path.dirname(__file__) + os.sep,)


class UnitxWarning(UserWarning):
    """Base class for all warnings raised by unitx."""


class InexactFloatWarning(UnitxWarning):
    """Warning for inexact float to Fraction conversion."""


class IncompleteFactorWarning(UnitxWarning):
    """Warning for incomplete factorization of integers."""


class UnitSymbolWarning(UnitxWarning):
    """Warning for improper unit symbols or names."""
