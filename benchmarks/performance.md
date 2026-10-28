# Performance study for the 0.17.6 candidate

This is a single-machine development study, recorded on 2026-10-28. It measures
specific operations, not an overall library ranking or a performance guarantee.
The baseline is commit `d0acb06` (0.17.5). The candidate uses private mutable
monomial accumulation and one shared immutable dimensionless Unit in the NumPy
adapter. Installed unitx metadata was 0.17.5 while testing the candidate source;
source fingerprints and numerical evidence are in [the JSON record](performance-baseline.json).

## Method and environment

Python 3.12.15, macOS 27.0.1 ARM64, NumPy 2.5.3 and SciPy 1.18.1.
Before/after runs used the same interpreter and built-in factorization backend,
with exported baseline/candidate sources selected through PYTHONPATH. Four
sequential runs followed baseline/candidate/candidate/baseline order. Timings
were collected without concurrent tests or profiling. CPU frequency and other
system processes were not controlled; no statistical confidence intervals are claimed.

Common-operation rows combine fourteen per-call samples (seven per run), with
loop counts calibrated to at least 0.02 seconds per sample. Scalar and fit rows
use the median of two run medians, each based on seven repeats of respectively
2000 and 50 calls. Setup and imports are excluded. Profiles were collected
separately and are not the source of these timings.

## Before and after

Times below are microseconds per call; a ratio above one means the candidate
was faster in this measurement.

| Operation | Baseline µs | Candidate µs | Baseline/candidate |
| --- | ---: | ---: | ---: |
| Fresh unit expression parsing | 28.89 | 20.68 | 1.40× |
| Scalar multiplication | 9.06 | 7.43 | 1.22× |
| 1-element array multiplication | 13.04 | 10.81 | 1.21× |
| 1000-element array multiplication | 12.87 | 11.06 | 1.16× |
| 1000-element dimensionless exponential | 16.91 | 15.47 | 1.09× |
| Complete 41-sample weighted fit | 2143.97 | 1882.67 | 1.14× |
| 1,000,000-element array multiplication | 413.88 | 422.69 | 0.98× |

The largest repeatable changes are in unit expression construction and small
unit products. Complete-fit elapsed time decreased by about 12%. Large-array
multiplication was about 2% slower in this run; no large-array throughput gain
is claimed. Small changes on untouched numerical paths, including conversion,
should not be attributed to this refactor. Individual samples remain available
for checking variation rather than relying only on this table.

## Shared float64 operations across libraries

This separate run used Pint 0.26.2, Astropy 8.0.2, unyt 3.2.0 and SymPy 1.14.0;
unitx selected the SymPy factorization backend in this environment. It is not
the built-in-backend environment of the before/after table. All libraries used
validated public constructors, prepared units and a common 1000-element float64
input. Actual backing-array sharing was verified for all five constructors.
NumPy is a manual numerical reference without units. Values are median µs/call.

| Operation | NumPy | unitx candidate | Pint | Astropy | unyt |
| --- | ---: | ---: | ---: | ---: | ---: |
| construct | 0.03 | 0.72 | 1.13 | 0.50 | 0.29 |
| convert | 0.45 | 6.25 | 4.08 | 1.86 | 2.01 |
| add_same | 0.30 | 6.27 | 3.11 | 1.93 | 1.95 |
| add_scaled | 0.78 | 9.56 | 7.65 | 3.01 | 3.18 |
| multiply | 0.29 | 11.28 | 3.09 | 3.74 | 1.98 |
| exp_ratio | 1.98 | 15.62 | 12.41 | 5.40 | 4.73 |
| mean | 0.91 | 4.51 | 13.19 | 2.50 | 8.00 |

unitx remains substantially slower on several small-array operations, especially
multiplication and dimensionless expressions. Its mean reduction compares more
favorably with Pint and unyt in this example. These results do not establish a
general winner. Exact factors, Fraction/Decimal behavior, affine/log coordinates,
implicit-conversion rejection and output failure guarantees are outside this
shared subset. Cross-library tests of those capabilities must establish matching
semantics first. Full scalar, 1-element and million-element samples are in the JSON.

## What changed and what remains expensive

Unit conversion metadata previously built Factor and Dimension powers and
products at every term. Each step froze another Monomial. They now each
accumulate rational exponents in one private mutable monomial and freeze once.
The resulting factor is validated once for conflicting symbolic bases. Frozen
snapshots have independent storage, so later accumulation cannot change a
hashed value. Unit keeps its formal product and extra scale; it does not
inspect the internals of Factor or Dimension or select exponent-specific paths.

The NumPy adapter also uses one immutable dimensionless Unit for its unitless
operands, results and tolerances instead of constructing it repeatedly. There
is no growing operation cache, changed unit identity or weakened write staging.

The warmed-fit profiles still show NumPy dispatch, exact unit construction,
monomial combination and numerical differentiation repeatedly entered by the
residual callback. The next investigation should isolate dispatch overhead and
unit algebra from the numerical kernel, and measure allocations before adding
another optimization. The present study does not justify changing exact
arithmetic or fixed-output failure semantics.

## Reproduce and inspect

Use the [benchmark instructions](README.md#shared-float64-comparisons) for the
comparison script and the [profile commands](README.md#sampling-and-call-profiles)
for warmed scalar, array and fitting workloads. The speedscope export retains
small frames (`filter_threshold=0`), otherwise repeated short residual calls
can disappear from the timeline. Sampling and deterministic profilers perturb
Python execution; use them to locate call chains, then verify any speed claim
with profiling disabled. Raw profiles contain machine paths and can be large,
so they are generated outside the repository.
The profiler reported high timer overhead on this machine; these profiles are
qualitative call-chain diagnostics rather than precise CPU-time attribution.

Cold-start latency, broader allocation/RSS behavior, other machines and complete
cross-library scientific workflows remain future measurements. No timing gate
has been added to CI.
