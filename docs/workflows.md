# Three scientific workflows

These scripts carry quantities from inputs through computation to a plotted
or restored result. They use existing unitx APIs and ordinary NumPy/SciPy
numerical boundaries. Inputs are illustrative synthetic data.

Install into the interpreter you will use to run the scripts:

```sh
python -m pip install 'unitx[scipy,matplotlib]'
```

Save the linked scripts in one directory, or run them from `examples/` in a
checkout matching your installed version. A notebook installation is unnecessary.
Each script opens a plot when run directly; importing it computes and checks
the example without calling `show()`. The scripts do not write project files.

## Formula: blackbody spectra

Save [blackbody_spectrum.py](https://github.com/DaitTHU/unitx/blob/main/examples/blackbody_spectrum.py)
and run `python blackbody_spectrum.py`.

The script evaluates Planck's spectral radiance at 3000, 4000 and 5000 K.
A wavelength column broadcasts against three temperatures. Constants come
from `SI`, and `unitx.math.expm1` retains the Quantity interface for the
dimensionless exponential argument. The numerical plotting boundary chooses
micrometres and kW/(sr m² nm) explicitly.

Expect three smooth spectra: the hotter curves are higher and peak at shorter
wavelengths. The script checks the finite grid peaks against Wien's law within
5 nm, checks finite nonnegative radiance, and repeats the formula using nm
inputs to verify unit-independent numerical results. This uses positive vacuum
wavelengths; converting a spectral density from wavelength to frequency would
also require a Jacobian, as described in [scientific boundaries](scientific-boundaries.md).

## Measurements: fitting and error bars

Save [measurement_fit.py](https://github.com/DaitTHU/unitx/blob/main/examples/measurement_fit.py)
and its companion [nonlinear_fit.py](https://github.com/DaitTHU/unitx/blob/main/examples/nonlinear_fit.py)
together, then run `python measurement_fit.py`.

Synthetic voltage readings decay with time and have independent Gaussian
noise with a known 20 mV standard deviation. The existing fitter uses explicit
V/s parameter scales, dimensionless weighted residuals and an analytic Jacobian.
The plot shows measurements with error bars and the fitted curve in ms/mV.
Each numerical plotting input, including the uncertainty width, is converted
to its chosen axis unit.

Expect an amplitude near 2.5 V and a decay time near 1.7 s. The script checks
these values and repeats the fit with ms/mV inputs, comparing both parameters
and the covariance in the fitter's fixed V/s parameter scales. The reported
standard error comes from a local linear covariance estimate with known
measurement errors; it is not a general confidence interval for nonlinear
models. See [the fitting derivation](scipy.md#nonlinear-fit-with-heterogeneous-parameter-units).

## Sampled data: integration, storage and recovery

Save [sampled_signal.py](https://github.com/DaitTHU/unitx/blob/main/examples/sampled_signal.py)
and run `python sampled_signal.py`.

The script samples a Gaussian power pulse, integrates it using the typed
`unitx.numpy.trapezoid` entry, and checks the result against the analytic pulse
energy. With 2 W peak power and 1.5 ms rms width, the result is approximately
7.520 mJ. Integration in W/ms and mW/s must agree within floating tolerance.

It then saves numeric arrays with JSON unit metadata to an in-memory `.npz`
archive, restores them with `allow_pickle=False`, and integrates again. Checks
cover units, stored values, shape, dtype and recovered energy. The plot uses
the restored data. NumPy's archive records are dynamically typed; the example
annotates and validates the expected float64 arrays at that storage boundary.
Replace the `BytesIO` object with a file path to retain the
archive. Custom definitions must be available during restoration; see
[array storage](array-storage.md) for coordinates and application-level validation.

## What is checked

Pytest executes these actual scripts, their numerical assertions and plot data
checks. Pyright checks the scripts and a consumer that asserts inferred
Quantity interfaces, numerical outputs and Matplotlib objects. The same source
scripts and consumer run outside the checkout against the installed wheel in
minimum/current dependency CI and release validation.

The checks use public numerical exits at actual library boundaries. They do not
add casts or dtype conversions merely to suppress typing errors, and they do
not claim that arbitrary NumPy/SciPy calls retain unitx types. See
[typing](typing.md) for those boundaries and [contributing](https://github.com/DaitTHU/unitx/blob/main/CONTRIBUTING.md)
for the validation commands.

[Ten-minute introduction](quickstart.md) · [Documentation index](README.md)
