"""Infer types from the actual runnable workflows, including their boundaries."""

from typing import assert_type

import blackbody_spectrum as spectrum
import measurement_fit as measurement
import numpy as np
import sampled_signal as pulse
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from unitx import Quantity, Unit

# Even when upstream dtype inference is broad, none of these whole quantities
# may become Any or a bare ndarray. Unit and numerical members must be visible.
assert_type(spectrum.radiance.unit, Unit)
assert_type(spectrum.peak.unit, Unit)
assert_type(spectrum.radiance.value.ndim, int)
assert_type(measurement.signal.unit, Unit)
assert_type(measurement.prediction.unit, Unit)
assert_type(measurement.amplitude, Quantity[float])
assert_type(measurement.lifetime, Quantity[float])
assert_type(measurement.lifetime_error, Quantity[float])
assert_type(measurement.covariance, NDArray[np.float64])
assert_type(pulse.energy.unit, Unit)
assert_type(pulse.restored_time.unit, Unit)
assert_type(pulse.restored_power.unit, Unit)
assert_type(pulse.restored_energy.unit, Unit)
assert_type(pulse.restored_power, Quantity[NDArray[np.float64]])
assert_type(pulse.restored_energy, Quantity[NDArray[np.float64] | np.float64])
assert_type(spectrum.fig, Figure)
assert_type(measurement.ax, Axes)
assert_type(pulse.ax, Axes)
