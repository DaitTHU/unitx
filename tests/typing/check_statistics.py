from decimal import Decimal
from fractions import Fraction
from typing import assert_type

from unitx import AffineQuantity, LogQuantity, Quantity, statistics
from unitx.si import si

type Scalar = int | float | Fraction | Decimal

assert_type(statistics.mean([1 * si.m, 200 * si.cm]), Quantity[Scalar])
assert_type(statistics.mean([20 @ si.degC, 30 @ si.degC]), AffineQuantity[Scalar])
assert_type(statistics.mean([1 @ si.dBm, 2 @ si.dBm]), LogQuantity[Scalar])
assert_type(statistics.fmean([1 * si.m]), Quantity[float])
assert_type(statistics.fmean([20 @ si.degC]), AffineQuantity[float])
assert_type(statistics.fmean([1 @ si.dBm]), LogQuantity[float])
assert_type(statistics.variance([1 * si.m, 2 * si.m]), Quantity[Scalar])
assert_type(statistics.stdev([1 * si.m, 2 * si.m]), Quantity[float | Decimal])
assert_type(statistics.covariance([1 * si.m, 2 * si.m], [1 * si.s, 2 * si.s]), Quantity[float])
assert_type(statistics.correlation([1, 2], [2, 3]), Quantity[float])
