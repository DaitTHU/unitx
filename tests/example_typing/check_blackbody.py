"""The documented asarray construction path; checked with current NumPy stubs."""

from typing import Any, TypeAlias, assert_type

import numpy as np
import numpy.typing as npt

from unitx import Quantity, Unit
from unitx.si import SI, si

NumpyValue: TypeAlias = npt.NDArray[Any] | np.number[Any]

lam = np.linspace(0.01, 3, 100)[:, None] * si.um
T = np.asarray([3000, 4000, 5000]) * si.K
nu = SI.c / lam
radiance = 2 * SI.h * SI.c**2 / (lam**5 * np.expm1(SI.h * nu / (SI.kB * T)))
radiance = radiance.to('kW/sr/m2/nm')

# Validate the container and useful members without promising that NumPy's
# inferred dtype for an unannotated input list matches its runtime dtype.
assert_type(T.unit, Unit)
assert_type(T.value.ndim, int)
assert_type(nu, Quantity[NumpyValue])
assert_type(radiance, Quantity[NumpyValue])
assert_type(radiance.value, NumpyValue)
assert_type(radiance.value.ndim, int)
radiance.value.reshape((100, 3))
