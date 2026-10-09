class UnitxError(Exception):
    '''Base class for all exceptions raised by unitx.'''


class UnitSyntaxError(UnitxError, ValueError):
    '''Exception raised when a unit expression has invalid syntax.'''


class UnitSymbolError(UnitxError, ValueError):
    '''Exception raised when a unit symbol is invalid.'''


class DimensionError(UnitxError, ValueError):
    '''Exception raised when quantities with incompatible dimensions are combined.'''


class UnitxWarning(UserWarning):
    '''Base class for all warnings raised by unitx.'''


class InexactFloatWarning(UnitxWarning):
    '''Warning issued when a float is converted to an approximate Rational.'''


class SymbolExistsWarning(UnitxWarning):
    '''Warning issued when a created symbol already exists.'''


class IncompleteFactorWarning(UnitxWarning):
    '''Warning issued when the factorization of a number is not completed due to time constraints.'''