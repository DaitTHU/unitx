# Optional SciPy support

Install the finite SciPy integration with `python -m pip install 'unitx[scipy]'`.
It supports SciPy >= 1.14 with NumPy >= 2.1. Core unitx imports remain independent
of both libraries; importing `unitx.scipy` also defers them until a call.
This is a limited support list, not general SciPy compatibility.

## Bessel function

`scipy.special.jv(order, argument)` works directly when either input is a
Quantity. Both inputs must be dimensionless linear values; scaled units
normalize to coherent values independently. The result is a dimensionless
Quantity. Real order, real/complex argument, broadcasting, numerical domains
and supported ufunc options follow SciPy. `out` requires a dimensionless
Quantity buffer and uses unitx's staged write rules. Points are unsupported.
No registration, global monkeypatch or extra import from unitx.scipy is needed.
Other special functions remain unsupported unless separately documented.

```python
import numpy as np
from scipy.special import jv
from unitx.si import si

value = jv(0, 2.0 * si.one)
np.testing.assert_allclose(value.value, 0.22389077914123562)
```

## Simpson integration

Use the explicit adapter `unitx.scipy.simpson(y, *, x=None, dx=1.0, axis=-1)`.
Native `scipy.integrate.simpson` converts its inputs to arrays without an
applicable Quantity dispatch hook and still requires explicit numerical input.

The adapter accepts numeric ndarrays/scalars and linear Quantities with one
unit per array. Complex samples are supported; coordinates are real. Points,
object arrays, custom array backends and implicit list/tuple magnitudes are
unsupported. Bare inputs are dimensionless. With x, the result unit is
`y.unit * x.unit`; without x it is `y.unit * dx.unit`, where dx must be scalar.
Bare spacing means dimensionless spacing. SciPy ignores dx when x is supplied,
and so does the adapter. Axis, coordinate shapes, odd/even sample handling,
nonuniform spacing, orientation and numerical errors follow SciPy.

```python
from unitx.scipy import simpson

x = np.linspace(0.0, 2.0, 5) * si.s
y = np.linspace(0.0, 2.0, 5) ** 2 * si.W
integral = simpson(y, x=x)
np.testing.assert_allclose(integral.value_in(si.J), 8 / 3)
```

Results have native NumPy magnitudes, including NumPy scalars for scalar
integrals, annotated `Quantity[Array]`. Decimal/Fraction scalars enter NumPy's
floating domain; this is not arbitrary-precision quadrature. No output buffer
is accepted. The adapter does not add a quadrature algorithm or change SciPy's
sampling assumptions.

The native jv stubs do not describe unit-aware dispatch, as with NumPy's
[typing limitation](typing.md#numpy-magnitude-types). The simpson adapter has
NumPy-independent public annotations and can be imported and type-checked
without SciPy installed.

Backends: [SciPy jv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jv.html),
[SciPy simpson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simpson.html).

[Documentation index](README.md) · [NumPy support](numpy.md)

## Nonlinear fit with heterogeneous parameter units

The executable [decay-fit recipe](https://github.com/DaitTHU/unitx/blob/main/examples/nonlinear_fit.py)
fits `signal(t) = amplitude * exp(-t/lifetime)` to synthetic voltage data.
Run `python examples/nonlinear_fit.py` with `unitx[scipy]` installed.

Amplitude and lifetime have different units. The solver vector contains their
values relative to explicit scales of 1 V and 1 s; it is not one Quantity array.
The callback reconstructs both parameters, evaluates the model with quantities,
then exits through `(prediction - observations) / sigma` in coherent dimensionless
values. Sigma is a known measurement standard deviation, so the objective is
weighted least squares. Positive numerical bounds are in the selected parameter
scales. The solver's `ftol`, `xtol` and `gtol` act on this normalized problem;
they are not voltage or time tolerances.

The example uses SciPy's
[`least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
with a real numerical residual and checks solver success. It computes the local
parameter covariance by SVD of the weighted Jacobian. This assumes independent
Gaussian errors, locally identifiable parameters and a valid linearization;
active bounds or strongly nonlinear confidence regions require another analysis.
Known absolute sigma means no residual-variance rescaling is applied. With
relative weights instead, that statistical assumption must be changed explicitly.

A numeric covariance C is accompanied by parameter units `(V, s)`, with element
units `u[i] * u[j]`. Standard errors use `sqrt(C[i,i]) * u[i]`. Changing a parameter
unit multiplies both its covariance row and column by the corresponding numeric
conversion factor. A correlation matrix is dimensionless; eigenvalues of the
unscaled heterogeneous numeric covariance are not unit-invariant physical
quantities. This finite recipe needs no general heterogeneous Quantity container.

The same dataset is fitted again in ms/mV, verifying matching physical parameters
and covariance within floating tolerances. Solver kernels run in NumPy's floating
domain; keeping quantities around the callback does not make them exact or
arbitrary precision. No general `scipy.optimize` wrapper is added.
