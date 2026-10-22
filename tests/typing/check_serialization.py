from decimal import Decimal
from fractions import Fraction
from typing import assert_type

from unitx import AffineQuantity, AffineUnit, Quantity, Unit
from unitx._numerics import Fractionable
from unitx.exceptions import SerializationError, UnitxError, VersionError
from unitx.serialization import from_dict, to_dict

assert_type(to_dict(Quantity(Fraction(1, 3), 'm')), dict[str, object])
assert_type(to_dict(AffineQuantity(Decimal('20'), 'degC')), dict[str, object])
assert_type(to_dict(Unit('m')), dict[str, object])

result = from_dict(to_dict(Quantity(1, 'm')))
assert_type(result, Unit | AffineUnit | Quantity[Fractionable] | AffineQuantity[Fractionable])
if isinstance(result, Quantity):
    assert_type(result.value, Fractionable)
    assert_type(result.unit, Unit)
if isinstance(result, AffineQuantity):
    assert_type(result.value, Fractionable)
    assert_type(result.unit, AffineUnit)

try:
    from_dict({'version': 2})
except VersionError as version_error:
    assert_type(version_error, VersionError)
    domain_error: UnitxError = version_error
    value_error: ValueError = version_error
except SerializationError as serialization_error:
    assert_type(serialization_error, SerializationError)
    domain_error = serialization_error
    value_error = serialization_error
