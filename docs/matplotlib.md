# Matplotlib plots

Install `unitx[matplotlib]` and enable the adapter explicitly:

```python
import numpy as np
import matplotlib.pyplot as plt
from unitx.matplotlib import register
from unitx.si import si

register()
time = np.array([0.0, 1.0, 2.0]) * si.s
distance = np.array([0.0, 2.0, 3.0]) * si.m
fig, ax = plt.subplots()
ax.plot(time, distance)
ax.set_xlabel('Time')
ax.set_ylabel('Distance')
```

Matplotlib reads numerical magnitudes through its own
[unit converter interface](https://matplotlib.org/stable/api/units_api.html).
The first unit-bearing data on each axis selects that axis's unit or point
coordinate. Later compatible data are converted to it. An unused default
axis label is set to the unit symbol; explicit labels remain under your control.
The original plotted input stays wrapped, so changing an axis's units also
updates the numerical curve:

```python
ax.yaxis.set_units(si.cm)
# Or select units when creating a curve:
ax.plot(time, distance, xunits=si.ms, yunits=si.cm)
ax.set_xlim(0.0 * si.s, 2.0 * si.s)
```

Axis unit arguments are actual `Unit`, `AffineUnit` or `LogUnit` objects,
such as `si.ms`, `si.degC` or `si.dBm`. Bare numbers passed to Matplotlib are
readings in the chosen axis unit, including numerical limits and Matplotlib's
own default values. A complete Quantity should wrap each data array. Scalar
quantities and empty arrays also work with `plot`; sequences of individually
wrapped quantities are not a promised plotting input format.

Quantities, affine points and logarithmic readings can each be plotted.
Each axis must keep compatible dimensions and the same point/difference
semantics. Point families and logarithmic conventions must also match.
Celsius points can align to Kelvin points; dBW readings can align to dBm.
Plotting a temperature difference on a temperature-point axis is rejected.
Matplotlib reports conversion failures through its `ConversionError`, with
the underlying unit error as the cause.

Plot real data explicitly. Use `np.real(q)` or `np.imag(q)` for complex
quantities before plotting. The adapter rejects implicit complex-to-real
conversion rather than discarding one component.

`register()` changes Matplotlib's global converter registry for the three
unitx quantity classes. Repeated registration is safe. Importing `unitx` does
not import Matplotlib or register anything. Enabling the adapter does not
allow `np.asarray(q)` or `np.array(q)` to drop units.

Supported workflows include `plot`, its scalar limits, default labels and axis
unit changes, plus the finite scatter/fill paths below. Other Matplotlib APIs may convert at the same boundary, but
functions that coerce inputs before consulting axis converters need separate
validation. The adapter does not claim support for every plotting function.

## Scatter, filled intervals and error bars

`scatter(x, y)` and `fill_between(x, y1, y2)` support wrapped real numeric
arrays, including affine and logarithmic readings. Both filled boundaries must
be compatible quantities or the same point family. Explicit `xunits`/`yunits`
choose the initial axis scale. Use bare numeric marker sizes, colors and boolean
masks; these rendering parameters are not additional physical data axes.
Unlike `plot`, collection-based artists do not promise to recompute stored
vertices/offsets when axis units change later. Choose their display units before
creating them, and recreate the artists to change scales.

`errorbar` coerces its inputs before the unit converter can reliably handle
them. Use an explicit numerical boundary for this function. For a point axis,
the error is a **difference**, so convert its width to the coordinate's
`delta_unit`, not to the point coordinate with an origin:

```python
from unitx import AffineQuantity, Quantity

time = Quantity(np.array([0.0, 1.0, 2.0]), si.s)
readings = AffineQuantity(np.array([20.0, 21.0, 19.5]), si.degC)
errors = Quantity(np.array([0.2, 0.3, 0.2]), si.K)
fig, ax = plt.subplots()
ax.errorbar(
    time.value_in(si.s),
    readings.value_in(si.degC),
    yerr=errors.value_in(si.degC.delta_unit),
    fmt='o',
)
ax.set_xlabel(f'Time / {si.s}')
ax.set_ylabel(f'Temperature / {si.degC}')
```

For logarithmic readings, convert widths into logarithmic differences in the
chosen convention. If displaying physical power instead, transform the lower
and upper reading bounds separately; symmetric log errors generally become
asymmetric physical errors. The explicit boundary does not relax `np.asarray(q)`
or promise that arbitrary plotting functions preserve units.

[NumPy support](numpy.md) · [Typing](typing.md)
