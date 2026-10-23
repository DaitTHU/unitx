"""Root unit definitions used to initialize spelling environments."""

from .dimension import _LOG_DIMENSION, Dimension
from .factor import ln10, one, ten
from .unitdefinition import UnitDefinition, _ConversionData

_UNIT_ENTRIES: tuple[UnitDefinition, ...] = (
    UnitDefinition._identity(),
    UnitDefinition('s', 'second', dimension=Dimension(T=1)),
    UnitDefinition('m', 'metre', 'meter', dimension=Dimension(L=1)),
    UnitDefinition('g', 'gram', factor=ten**-3, dimension=Dimension(M=1)),
    UnitDefinition('A', 'ampere', dimension=Dimension(I=1)),
    UnitDefinition('K', 'kelvin', dimension=Dimension(Theta=1), symbol_aliases='K'),
    UnitDefinition('mol', 'mole', dimension=Dimension(N=1), symbol_aliases='㏖'),
    UnitDefinition('cd', 'candela', dimension=Dimension(J=1), symbol_aliases='㏅'),
    UnitDefinition._derived('Np', 'neper', conversion=_ConversionData(one, _LOG_DIMENSION)),
    UnitDefinition._derived('B', 'bel', conversion=_ConversionData(ln10 / 2, _LOG_DIMENSION)),
)
