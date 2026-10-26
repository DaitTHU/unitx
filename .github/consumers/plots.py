"""Check plotting through the optional installed Matplotlib adapter."""

from typing import assert_type

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from unitx import Quantity
from unitx.matplotlib import register
from unitx.si import si

assert_type(register(), None)
# Matplotlib 3.9 leaves add_subplot's **kwargs unannotated.
axes = Figure().add_subplot()  # pyright: ignore[reportUnknownMemberType]
assert_type(axes, Axes)
time = Quantity(np.array([0.0, 1, 2]), si.s)
distance = Quantity(np.array([1.0, 2, 3]), si.m)
# Matplotlib leaves **kwargs and Axis.set_units' unit parameter unannotated.
# Suppress only those third-party unknown-member diagnostics, retaining input
# and return checking for the unitx objects and the explicit registration API.
(line,) = axes.plot(time, distance)  # pyright: ignore[reportUnknownMemberType]
np.testing.assert_array_equal(line.get_xdata(orig=False), [0, 1, 2])
np.testing.assert_array_equal(line.get_ydata(orig=False), [1, 2, 3])
axes.yaxis.set_units(si.cm)  # pyright: ignore[reportUnknownMemberType]
np.testing.assert_array_equal(line.get_ydata(orig=False), [100, 200, 300])
assert axes.get_ylabel() == 'cm'
try:
    np.asarray(distance)
except TypeError:
    pass
else:
    error_message = 'Plotting must not enable implicit unit stripping.'
    raise AssertionError(error_message)
