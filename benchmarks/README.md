# Performance measurements

Install the checkout in the environment to be measured, then run from the
repository root:

```sh
python -m pip install -e .
python benchmarks/scalars.py --repeat 7 --number 1000
```

The scalar script measures fresh expression parsing, Fraction/Decimal
conversion, cross-unit equality and ordering, factor construction, rational
powers, symbolic cancellation and sparse Dimension projection/display. Inputs are prepared
outside the timed operations; each parsing call constructs a new parser.
It also compares repeated formulas with inline/reused compound units and
isolates unit powers, products and construction. The new formula/product cases
validate their exact results before measurement.
The JSON output records per-call medians and individual samples, separate peak
traced allocations, source SHA-256 fingerprints, an optional `--label`,
Python/platform, factorization backend, Decimal precision and timing parameters. SymPy is used
automatically when installed; compare backends in separate environments.

Array write measurements require the optional NumPy dependency:

```sh
python -m pip install -e '.[numpy]'
python benchmarks/numpy_writes.py --size 1000000 --repeat 7 --number 20
```

That script also reports peak traced allocations for each operation. Timing
and allocation tracing are separate measurements. Run baseline and candidate
with the same interpreter, dependencies, arguments and machine conditions.
These are development measurements, not performance guarantees; CI does not
enforce timing thresholds.

For prepared contractions and an independent circuit solve-to-energy workflow:

```sh
python benchmarks/workflows.py --repeat 7 --number 1000 --label candidate
```

This script validates trace, dot and einsum results and follows a circuit from
linear solve through power contraction to time-integrated energy. Equivalent
scaled inputs exercise exact unit factors. Conversion and a short exponential
expression serve as controls. Inputs and fixed units are prepared before timing;
source fingerprints, samples, separately traced allocations and untimed Unit
metadata call counts are recorded. The workflow complements the weighted fit
without attributing preparation or solver algorithm changes to library speed.

The complete weighted-fit recipe has a separate benchmark requiring SciPy:

```sh
python benchmarks/nonlinear_fit.py --repeat 7 --number 10
```

It measures the physical model, normalized residual callback, optimization and
local covariance together, excluding import and data generation. Analytic and
two-point finite-difference Jacobians are timed separately after validating
their fitted parameters and covariance. Individual timing samples and actual
residual/Jacobian callback counts are reported; cProfile counts come from a
separate untimed fit, rather than relying on SciPy's `nfev` convention. The
[recorded performance snapshot](performance.json) includes commands, interpreter,
platform, backend versions and timing parameters. Array writes intentionally
allocate staging storage to preserve their failure invariance.

## Shared float64 comparisons

Use a separate environment for development-only comparison dependencies:

```sh
python -m venv /tmp/unitx-performance
/tmp/unitx-performance/bin/python -m pip install -e . -r benchmarks/requirements.txt
/tmp/unitx-performance/bin/python benchmarks/compare.py \
  --libraries numpy unitx pint astropy unyt \
  --sizes scalar 1 1000 1000000 --repeat 7 --label candidate > comparison.json
```

The script validates each operation in the expected output unit before timing.
Registries, units and inputs are prepared outside the measurement. The shared
subset covers float64 construction, m-to-cm conversion, same/scaled-unit addition,
multiplication, a dimensionless exponential and array means. NumPy is a manual
numerical reference without dimensional checks. Exact arithmetic, coordinate
semantics and fixed-output failure guarantees require separate comparisons.

Validated public constructors are used throughout. Astropy explicitly avoids
array copying; the report checks actual backing-array sharing for every library.
All timings include the small Python benchmark callback. The JSON records
dependency versions, the unitx factorization backend, calibrated loop counts,
individual timing samples and a separate tracemalloc peak. Traced allocations
are not process RSS or a guarantee that every native allocation is visible.

The `label` identifies the tested source revision; installed distribution metadata
alone cannot identify an exported source tree selected through `PYTHONPATH`.
For a before/after comparison, export the baseline commit to another directory,
then run the same script and interpreter with each source directory on
`PYTHONPATH`. Alternate baseline/candidate/candidate/baseline runs on an otherwise
idle machine. Do not run tests, other benchmarks or profilers concurrently.
Compare samples and physical results before interpreting small timing changes.

Keep the [performance report](performance.md) explicit about the measured source
snapshot, cross-library gaps, scope and limits. New source changes do not update
existing measurements; rerun before claiming current timings. Local before/after
experiments guide implementation decisions; they do not need separate versioned reports.

## Sampling and call profiles

Run a warmed workload with imports excluded from the recorded region:

```sh
python benchmarks/profile_workload.py fit --seconds 10 --profile /tmp/unitx-fit
python benchmarks/profile_workload.py array --seconds 10 --profile /tmp/unitx-array
python benchmarks/profile_workload.py scalar --seconds 10 --cprofile /tmp/unitx-scalar.pstats
python -m pstats /tmp/unitx-scalar.pstats
```

`scalar` retains inline unit powers; `scalar-reused` moves those fixed unit
constructions outside the repeated call. `array` uses 41 elements and
`array-large` uses 1,000,000, separating dispatch overhead from numerical kernels
and memory traffic. `fit` includes the full weighted solver and covariance recipe
with its analytic Jacobian; `fit-finite` selects two-point finite differences.

`--profile` requires Pyinstrument and writes interactive `.html` and
`.speedscope.json` reports. Open the HTML directly or load the JSON in
[speedscope](https://www.speedscope.app/) for flamegraph/left-heavy views.
`--cprofile` uses only the standard library. Neither profile's elapsed times
should be substituted for the uninstrumented benchmark samples. Keep generated
HTML, profiler databases and machine-specific paths outside the repository;
the scripts and compact numerical evidence are sufficient to reproduce them.
