# Performance

Current measurements were recorded on 2026-10-29. This document describes
unitx's present costs, comparisons with other libraries and remaining
optimization priorities. It is updated in place. Raw current samples, commands,
allocation peaks and source SHA-256 fingerprints are in
[performance.json](performance.json); reproduction commands are in the
[benchmark guide](README.md).

## Measurement conditions

Python 3.12.15, macOS 27.0.1 ARM64 and NumPy 2.5.3. The cross-library environment
uses Pint 0.26.2, Astropy 8.0.2, unyt 3.2.0 and SymPy 1.14.0; unitx selects the
SymPy factorization backend there. Scalar, fitting and array-write measurements
use the built-in backend; fitting uses SciPy 1.18.1.

Registries, units and input data are prepared before timing. Public constructors
are validated, and array sharing is checked for every library. Each shared
operation is validated in the expected output unit before measurement. Imports
and setup are excluded. Timings include the small Python benchmark callback;
tracemalloc is run separately. Tests and profilers do not run concurrently with
timing.

The shared-operation benchmark takes seven samples per operation, calibrating
loop counts to at least 0.02 seconds per sample. Values are medians. Scalar and
fit summaries combine two run medians, each from seven repeats of respectively
2000 and 50 calls. CPU frequency and other processes are not controlled. These
are observations on one machine, without confidence intervals or CI timing
thresholds. Exported source is identified by hashes; installed distribution
metadata in the raw measurements can differ from that source.

## Shared float64 operations

Inputs are 1000-element float64 arrays. Values are microseconds per call.
NumPy is a manual numerical reference without dimension checking.

| Operation | NumPy | unitx | Pint | Astropy | unyt |
| --- | ---: | ---: | ---: | ---: | ---: |
| construct | 0.026 | 0.522 | 1.129 | 0.483 | 0.288 |
| convert | 0.427 | 5.893 | 4.122 | 1.834 | 2.011 |
| add_same | 0.272 | 4.149 | 3.080 | 1.891 | 1.957 |
| add_scaled | 0.730 | 4.750 | 7.585 | 2.983 | 3.190 |
| multiply | 0.273 | 8.332 | 3.089 | 3.712 | 2.011 |
| exp_ratio | 1.940 | 10.925 | 12.301 | 5.265 | 4.547 |
| mean | 0.903 | 4.213 | 13.533 | 2.459 | 7.977 |
| mean_method | 0.722 | 4.504 | 2.441 | 2.107 | 7.461 |

`convert` changes m to cm; `add_scaled` adds equivalent values stored in m and
cm; `exp_ratio` evaluates `exp(q / scale)` with compatible dimensional inputs.
`mean` calls `np.mean(q)`; `mean_method` calls `q.mean()` with the same inputs.
Construction shares the existing input array and does not copy its elements.

The main gaps are conversion, multiplication and dimensionless expressions.
Unitx multiplication takes about 2.2 times Astropy's time and 4.1 times unyt's
in this case. Conversion takes about 3.2 times Astropy's time. Scaled addition
and the exponential ratio are faster than Pint here; mean is faster than Pint
and unyt. These are operation-specific comparisons, not an overall ranking.
Scalar, 1-element and million-element samples are also in the raw record.

The shared subset does not establish matching Fraction/Decimal exactness,
formal unit identity, affine/log coordinates, implicit-conversion rejection or
fixed-output failure guarantees across libraries. The remaining differences
cannot all be attributed to either semantics or implementation without further
isolated measurements.

## Exact scalar operations and a complete fit

| Scalar operation | µs/call |
| --- | ---: |
| `parse_expression` | 18.715 |
| `convert_fraction` | 4.431 |
| `convert_decimal` | 4.608 |
| `equal_cross_unit` | 2.693 |
| `order_cross_unit` | 2.703 |
| `factor_construct` | 8.019 |
| `factor_rational_power` | 1.522 |
| `factor_cancel` | 1.958 |
| `dimension_physical` | 0.723 |
| `dimension_repr` | 1.075 |
| `dimension_dict` | 0.615 |

These scalar operations retain exact rational/symbolic unit algebra. Fractions
and Decimal use their documented magnitude domains; Decimal precision is 28
for these measurements.

The complete 41-sample weighted exponential-decay fit takes **1.311 ms**.
It includes the physical model, normalized residual callback, SciPy optimization
and local covariance calculation, excluding imports and input preparation. See
[the fitting recipe](../examples/nonlinear_fit.py).

## Array writes and allocations

The following measurements use 1,000,000-element arrays, seven repeats of ten
calls. Traced allocation peaks are measured separately and are not process RSS.

| Operation | ms/call | Peak traced bytes |
| --- | ---: | ---: |
| `add_out` | 1.220 | 8001331 |
| `converted_out` | 1.846 | 16001819 |
| `add_inplace` | 0.798 | 8000930 |
| `scalar_assignment` | 0.062 | 646 |
| `overlapping_assignment` | 0.551 | 8000448 |

Output writes stage results before modifying the destination. This explains
some temporary storage, especially when conversion requires another array.
Allocation reduction must preserve unit/dtype validation, overlapping-storage
behavior and the documented failure invariance.

## Current implementation and optimization priorities

Unit composes powered Factor and Dimension products; each model owns its
empty/single-term products and exponent-one identities. Factor normalizes
arithmetic inputs before applying its product/quotient rules. Dimension projects
physical axes directly from its sparse monomial, and display/dictionary views
iterate in canonical order without constructing seven-element tuples. The
fixed-layout tuple remains at serialization and SI base-unit boundaries. NumPy scaling consumes the evaluated
factor without an additional decomposition. Fixed ufunc groups and the
empty-keyword path reduce dispatch allocations.

Direct array methods use ordinary Python signatures and explicitly forward
arguments to a small shared lazy helper. There is no runtime Signature.bind,
manual parameter binder or generated wrapper. Forwarding adds some repetitive
source lines but leaves binding errors/defaults to Python and unit semantics to
the existing handlers. Arithmetic does not propagate numerical caches between
Factor operands. No public API or growing operation/type cache was added.

Further work should focus on:

1. **Unit metadata in products and ratios.** Nontrivial formal products still
   construct and validate exact monomials, factors and dimensions. Measure
   those allocations separately from the numerical multiplication.
2. **Conversion.** Separate ratio construction, exact-factor evaluation and
   scalar/array scaling to identify the remaining fixed overhead.
3. **NumPy dispatch.** Short arrays repeatedly traverse protocol dispatch,
   operand validation and unit alignment. Other operands' overrides and staged
   output validation remain required behavior.

Warmed profiles locate these call chains; their elapsed times do not substitute
for uninstrumented timings. Keep raw profiler databases outside the repository.
Do not infer numerical-kernel speed from small-array dispatch costs or give up
exactness and failure guarantees solely to match another library's timing.
