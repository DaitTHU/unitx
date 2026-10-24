"""NumPy >= 2.5 stubs preserve the wrapper for unannotated asarray input."""

from typing import assert_type

import numpy as np

from unitx import Unit
from unitx.si import si

temperature = np.asarray([3000, 4000, 5000]) * si.K
assert_type(temperature.unit, Unit)
assert_type(temperature.value.ndim, int)
temperature.value.reshape((1, 3))
