# Performance measurements

Install the checkout in the environment to be measured, then run from the
repository root:

```sh
python -m pip install -e .
python benchmarks/scalars.py --repeat 7 --number 1000
```

The scalar script measures fresh expression parsing, Fraction/Decimal
conversion, cross-unit equality and ordering, factor construction, rational
powers and symbolic cancellation. Input quantities and factors are prepared
outside the timed operations; each parsing call constructs a new parser.
The JSON output records median seconds per call, Python/platform, the selected
factorization backend, Decimal precision and timing parameters. SymPy is used
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


The complete weighted-fit recipe has a separate benchmark requiring SciPy:

```sh
python benchmarks/nonlinear_fit.py --repeat 7 --number 10
```

It measures the physical model, normalized residual callback, optimization and
local covariance together, excluding import and data generation. The recorded
[pre-1.0 baseline](pre1-baseline.json) includes commands, interpreter, platform,
backend versions and timing parameters. It is one machine's observation, not a
release threshold or a cross-library comparison. Array writes intentionally
allocate staging storage; this baseline supplies no reason to trade away their
failure invariance or introduce a cache/adapter framework.
