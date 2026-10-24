# Scientific examples

For a small extension without optional dependencies, see
[Extending Quantity: complex impedances](impedance.md). Its standalone Python
script uses a Quantity subclass for series `+` and parallel `//` composition
and calculates an RLC circuit with complex current and power.

These notebooks use unitx for dimensional calculations and NumPy for numerical
work. Each contains a model, assumptions, plots, and numerical checks. The
material parameters and measurement draws are illustrative, not experimental data.

| Notebook | Topic | unitx features |
| --- | --- | --- |
| [Blackbody radiation](blackbody.ipynb) | Planck spectra and Wien peak checks | Constants, broadcasting, spectral-density conversion |
| [Damped oscillator](damped_oscillator.ipynb) | Release from rest and energy loss | Dimensionless ufunc arguments, energy units, comparisons |
| [Gaussian wave packet](gaussian_wavepacket.ipynb) | Free electron propagation and uncertainty | Probability-density units, quantum constants, unit-aware integration |
| [Optical attenuation](optical_attenuation.ipynb) | Transmission through absorbing layers | prod/multiply.reduce, input masks, Quantity initial, fixed-unit out |
| [Pendulum uncertainty](pendulum_uncertainty.ipynb) | Correlated input uncertainties and Monte Carlo propagation | Mean/std/var units, mixed measurement units, explicit quantiles |
| [Arrhenius fit](arrhenius_fit.ipynb) | Estimate activation energy from synthetic reaction rates | Absolute temperatures, regression boundary, parameter units and model-based standard errors |
| [Smith chart](smith_chart.ipynb) | Series/parallel RLC impedance and reflection trajectories | Quantity subclass, complex arithmetic, explicit normalization and plotting boundary |

## Run

Use Python 3.12 or newer. From the repository root, install the current checkout
and notebook tools in the environment you will select as the Jupyter kernel:

```sh
python -m pip install -e '.[numpy]' matplotlib jupyterlab
jupyter lab examples/
```

Open a notebook and run all cells from the beginning. Use the current checkout
so that all demonstrated APIs are available.
Keep `smith_chart.ipynb` beside its companion `impedance.py` module.
The examples do not download datasets or write files. Keep physical quantities in calculations;
plot their `.value` and derive labels from their `.unit`. Convert only to choose
a useful display scale or to align units at a numerical boundary. Quantiles use
an explicit numerical boundary and reattach the input unit, without naming a
variable after its unit. Display equations use `$$...$$`.
Generate physical parameter labels from Quantity formatting or `to_latex()`;
do not repeat their numerical values and unit spellings in plot strings.

## Keep Git history small

Commit only notebook source and minimal kernel metadata. Clear all outputs and
execution counts before committing; do not embed figures, widget state, or
attachments. Figures are regenerated when cells run. The test suite checks this
policy and executes the numerical examples when NumPy and Matplotlib are present.

To clear a notebook after running it, use JupyterLab's **Clear Outputs of All
Cells**, then save it. For a command-line cleanup using nbconvert:

```sh
python -m pip install nbconvert
jupyter nbconvert --clear-output --inplace examples/*.ipynb
```

The notebooks use ordinary Python code cells. Their code is also extracted for
the existing Pyright example check, without committing duplicate `.py` exports.
Assertions on NumPy results also check that dispatch retains the Quantity
wrapper. NumPy's stubs cannot describe every wrapped result; the optical
example documents local suppressions for its Quantity `initial` and `out` call.
