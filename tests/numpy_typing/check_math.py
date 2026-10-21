from typing import assert_type

import numpy as np

from unitx import Quantity, math
from unitx._numerics import Array

q = Quantity(np.array([1.0, 2.0]), 'm2')
assert_type(math.sqrt(q), Quantity[Array])
assert_type(math.sin(np.array([1.0])), Quantity[Array])
assert_type(math.asin(Quantity(np.float64(0.5), '')), Quantity[Array])
assert_type(math.hypot(q, q), Quantity[Array])
assert_type(math.atan2(q, q), Quantity[Array])
assert_type(math.atan2(Quantity(1, 'm2'), q), Quantity[Array])
