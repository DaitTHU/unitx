"""Custom exceptions and warnings for the unitx library."""

import os
from typing import Final

__all__ = [
    'DimensionError',
    'IncompleteFactorWarning',
    'InexactFloatWarning',
    'SerializationError',
    'UnitRegistryError',
    'UnitSymbolError',
    'UnitSymbolWarning',
    'UnitSyntaxError',
    'UnitxError',
    'UnitxWarning',
    'VersionError',
]


class UnitxError(Exception):
    """Base class for all exceptions raised by unitx."""


class UnitSymbolError(UnitxError):
    """Exception raised for invalid unit symbols or names."""


class UnitSyntaxError(UnitxError):
    """Exception raised for invalid unit syntax."""


class DimensionError(UnitxError):
    """Exception raised for dimension mismatches in quantity operations."""


class UnitRegistryError(UnitxError):
    """Exception raised for errors in the unit registry."""


class SerializationError(UnitxError, ValueError):
    """A serialized record has an invalid format or cannot be restored."""


class VersionError(UnitxError, ValueError):
    """An explicitly specified version is unsupported by the operation."""


# Python 3.12 warnings.warn() new API
_WARNING_SKIP_PREFIXES: Final = (os.path.dirname(__file__) + os.sep,)


class UnitxWarning(UserWarning):
    """Base class for all warnings raised by unitx."""


class InexactFloatWarning(UnitxWarning):
    """Warning for inexact float to Fraction conversion."""


class IncompleteFactorWarning(UnitxWarning):
    """Warning for incomplete factorization of integers."""


class UnitSymbolWarning(UnitxWarning):
    """Warning for improper unit symbols or names."""
