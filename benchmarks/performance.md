# Performance

The original tables below were recorded on 2026-10-29 from the **0.17.7 source
snapshot** identified in the raw record. A separate
[focused algebra experiment](#focused-unit-algebra-measurements) compares 0.17.9
with the identity-product changes. It does not update the original cross-library
tables. Each measurement applies to its recorded source fingerprints. Raw samples, commands,
allocation peaks and source SHA-256 fingerprints are in
[performance.json](performance.json); reproduction commands are in the
[benchmark guide](README.md).

A later [routing and Jacobian experiment](#routing-and-jacobian-measurements)
measures the subsequent changes against the existing uncommitted working tree.

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

## Implementation costs and investigation priorities

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

## Focused unit-algebra measurements

The 2026-10-29 experiment compares commit `9e0fc3f` (0.17.9) with the candidate
identified under `unit_algebra_experiment` in [performance.json](performance.json).
It uses Python 3.12.15, NumPy 2.5.3, SciPy 1.18.1 and the built-in factorization
backend on the same machine. Runs alternate baseline/candidate/candidate/baseline,
with tests and profilers stopped. Scalar values below are the median of two run
medians, each with seven samples of 5000 calls; times are microseconds per call.

| Operation | Baseline | Candidate | Elapsed change |
| --- | ---: | ---: | ---: |
| Formula with inline squared units | 21.203 | 19.249 | -9.2% |
| Same formula with reused squared units | 9.032 | 8.326 | -7.8% |
| `si.m**2` | 4.357 | 3.322 | -23.8% |
| Coherent composite-unit product | 7.899 | 6.968 | -11.8% |
| Copy a coherent composite Unit | 5.935 | 5.066 | -14.6% |
| Scaled fractional unit power | 6.584 | 6.642 | +0.9% |
| Composite product with nontrivial factors | 14.851 | 15.199 | +2.3% |
| Fresh expression parsing | 18.880 | 17.853 | -5.4% |

The formula squares a length of 1.5 m, adds 10000 cm² and extracts m², producing
3.25 in both cases. Moving only the fixed `si.m**2` and `si.cm**2` constructions
outside the repeated call cuts elapsed time by about 57% in either source.
This usage improvement is separate from the smaller library improvement.

The implementation reuses empty Monomials after validating the exponent and
filters identity Factor/Dimension terms before accumulating conversion metadata.
Formal unit terms remain intact, including dimensionless named definitions and
logarithmic degree. The filters retain existing term tuples rather than rebuilding
them. No cache, new public entry or preferred unit identity is introduced.

Factor and Dimension retain their exponent-one paths: they can return the whole
object without validating and wrapping the same Monomial again. An additional
Monomial exponent-one branch was considered but omitted. The representative
internal workloads do not send that case through those paths; explicit Unit
exponent-one operations and parser inputs still work through ordinary algebra.

The focused array check measures only unitx, not the other libraries. Multiplication
of 1- and 1000-element arrays takes about 10% less time. Million-element results
remain sensitive to memory traffic and run variability; these samples do not
establish a numerical-kernel speedup. The complete fit is effectively unchanged:
1.305 ms versus 1.297 ms, using seven repeats of 30 fits per run. Some scalar
cases also show small increases, including the nontrivial-factor product above.
The gains therefore apply to identity-heavy algebra rather than every workload.

## Warmed call-path profiles

The 2026-10-29 profiles use the same Python 3.12.15 / NumPy 2.5.3 / SciPy 1.18.1
environment and built-in backend. Each workload is warmed before a sequential
10-second Pyinstrument 5.1.3 capture at a 1 ms interval, followed by a separate
2-second cProfile pass. Source fingerprints and compact observations are saved
under `profile_observations` in [performance.json](performance.json). Generated
interactive HTML, flame graphs and profiler databases remain outside the repository.

Both profilers perturb call-heavy Python code. The following are diagnostic
inclusive shares, not uninstrumented speedup estimates. Ancestors include their
descendants, so rows within a workload cannot be added.

| Workload | Call path | Sampled share |
| --- | --- | ---: |
| Scalar, inline units | `Unit.__pow__` | 42.7% |
| Scalar, reused units | `Unit.__mul__` / its `_from_monomial` child | 40.4% / 22.5% |
| Scalar, reused units | `is_numpy_object` | 9.7% |
| 41-element array expression | `_dispatch_ufunc` | 62.4% |
| 41-element array expression | `dispatch_function`, reached by `.mean()` | 19.1% |
| Complete fit | Residual callback | 75.1% |
| Complete fit | `approx_derivative`, including repeated callbacks | 58.6% |
| Complete fit | `_dispatch_ufunc` within callbacks | 63.3% |

cProfile confirms the main call-chain ranking. Per scalar call with reused units,
it counts one `_from_monomial` and ten `is_numpy_object` calls. The short array
expression performs six supported-operand checks, four linear-part extractions
and four result wraps. Each fit evaluates the residual 18 times and constructs
Unit metadata 92 times.

The million-element profile attributes about 98% to `_dispatch_ufunc`, mostly at
its native ufunc call sites. This is not evidence that Python routing consumes
98% of a large-array calculation. In a separate uninstrumented
`exp(-x/2).mean()` control, bare NumPy and unitx take approximately 1.28 / 17.19 µs
for 41 elements, and 2.38 / 2.65 ms for 1,000,000 elements. Native kernels,
temporary-array lifetimes and memory traffic require separate investigation
from Python dispatch at the larger size.

These profiles suggest the following order of investigation:

1. **Short-array routing and operand preparation.** Measure the long ufunc
   branch chain, repeated recognition/validation and wrapper construction.
   Common fixed ufunc groups and already-bound direct methods may offer shorter
   routes to the same handlers. Preserve protocol overrides, point semantics,
   numeric dtype checks and staged output boundaries.
2. **Scalar product metadata.** Reused fixed units still leave the product's
   formal monomial, Factor and Dimension construction. Isolate those intermediate
   objects and normalized-construction scans before adding cached state.
3. **Scalar backend recognition and lazy lookup.** Built-in scalar inputs
   repeatedly traverse the NumPy-recognition MRO path. Measure a narrow scalar
   negative path and repeated lazy imports while retaining the no-NumPy core
   boundary and the existing subclass/backend rejection rules.
4. **Fit algorithm, separately from the library.** An analytic Jacobian is a
   candidate for this two-parameter model because finite differences repeatedly
   reevaluate the residual. Validate derivatives, normalized parameter scales,
   physical results and covariance before claiming an end-to-end improvement.

Parsing, registration and factorization are not dominant in these warmed
workloads. Their setup/import costs were excluded, so these profiles do not
establish startup performance or replace workloads that exercise those paths.

## Routing and Jacobian measurements

This 2026-10-29 experiment starts from the uncommitted tree containing the
identity-product improvements above, rather than the original 0.17.9 commit.
The baseline and candidate source hashes, raw samples, commands, metadata
variants and separate callback counts are under `routing_experiment` in
[performance.json](performance.json). It uses Python 3.12.15 on macOS 27.2
ARM64, NumPy 2.5.3, SciPy 1.18.1 and the built-in factorization backend.
The existing interpreter's distribution metadata still says 0.14.0; the
recorded source fingerprints identify the actual 0.17.9 working trees.

Scalar and array runs alternate baseline/candidate/candidate/baseline, with
seven samples per run. Scalars use 5000 calls per sample; arrays calibrate to
at least 0.02 seconds. The table gives medians of the two run medians in
microseconds per call. Profiling and allocation tracing are separate from timing.

| Operation | Baseline | Candidate | Elapsed change |
| --- | ---: | ---: | ---: |
| Formula with inline squared units | 19.294 | 17.139 | -11.2% |
| Same formula with reused squared units | 8.407 | 7.090 | -15.7% |
| `si.m**2` | 3.294 | 2.991 | -9.2% |
| Coherent composite-unit product | 6.972 | 6.421 | -7.9% |
| Composite product with nontrivial factors | 15.212 | 14.444 | -5.0% |
| Copy a coherent composite Unit | 5.052 | 4.565 | -9.6% |
| Scaled fractional unit power | 6.637 | 6.343 | -4.4% |
| 41-element same-unit addition | 3.965 | 3.562 | -10.2% |
| 41-element multiplication | 7.180 | 6.159 | -14.2% |
| 41-element exponential ratio | 9.081 | 6.936 | -23.6% |
| 1000-element multiplication | 7.603 | 6.368 | -16.2% |
| 1000-element exponential ratio | 10.842 | 8.675 | -20.0% |

Common multiplicative, unit-preserving and dimensionless ufunc groups now occur
early in the dispatcher. One validation pass also detects points/descriptors;
ordinary inputs no longer rebuild a normalized tuple. Linear operand extraction
no longer goes through generic point-aware extraction, and linear results call
the validated Quantity constructor directly. In the short
`exp(-q/denominator).mean()` profile, generic `_operand_parts` calls fall from
four to zero and `_wrap` calls from four to one. The remaining wrap belongs
to the mean handler. Supported-operand checks remain six: input dtype and
backend validation are still required. Output staging and other operands'
protocol overrides retain their existing behavior.

Unit metadata collects nonidentity factor/dimension terms in one pass. The
formal monomial is retained in full. The product helpers receive these filtered
sequences and keep their empty/single-term paths. Compared with the original
two-generator/two-filter path, the coherent product's separately traced peak
falls from 1496 to 1199 bytes and Unit copy from 1208 to 863 bytes. This is not
an allocation reduction for every case: the nontrivial-factor product's peak
increases from 2128 to 2391 bytes because the two lists coexist.

A separate experiment compares four ways to prepare those terms on otherwise
identical candidate source. Each variant runs twice, in forward then reverse
order, with seven samples of 5000 calls. Values below are microseconds per call.

| Preparation | `si.m**2` | Coherent product | Unit copy | Empty Unit |
| --- | ---: | ---: | ---: | ---: |
| One loop with filtered list appends | 3.025 | 6.511 | 4.648 | 1.311 |
| Two filtered list comprehensions | 3.118 | 6.615 | 4.726 | 1.375 |
| Filtered generators, then `list()` in helpers | 3.334 | 6.898 | 4.976 | 1.617 |
| Filtered generators, streamed with `iter()`/`next()` | 3.253 | 6.778 | 4.886 | 1.524 |

Generator syntax alone does not remove work: materialization still allocates
lists, while streaming needs lookahead to preserve the empty/single-term paths.
The measured one-loop form is retained. These timings do not establish a
universal preference for loops over comprehensions or generators.

Exact Python int, float, complex, Fraction, Decimal and bool types return from
`is_numpy_object` before its MRO scan. Subclasses still take the original scan;
there is no growing type cache. The reused scalar formula still calls this
helper ten times, but those calls avoid the scan. Its operand probe also uses
a direct loop instead of allocating an `any()` generator. These scalar changes
are included in the combined table above, not attributed solely to Unit metadata.

Million-element multiplication and exponential-ratio measurements change by
about -0.1% and -0.5%; same-unit addition increases by 3.5%. These runs show no
consistent large-array improvement and do not establish a numerical-kernel
speedup. CPU frequency and other processes remain uncontrolled.

The fit algorithm is measured separately on the same candidate library. Both
modes include the quantity model, normalized residuals, solver and local SVD
covariance. Two runs each use seven samples of 50 fits. Finite differences take
**1.137 ms**, versus **0.933 ms** for the analytic Jacobian, about **18% less
elapsed time**. Separate warmed call counts show 18 residual evaluations for
two-point differences, versus six residual and six Jacobian evaluations for
the analytic mode. It does not remove the derivative calculation itself.

Derivative checks cover nonoptimal parameters, zero amplitude and equivalent
s/V versus ms/mV input units. Fits with and without noise agree with finite
differences on parameters and covariance. Derivatives use the normalized solver
parameter scales, including residual weighting. The example defaults to the
analytic Jacobian; `analytic_jacobian=False` and the `fit-finite` profile workload
retain the comparison route. This algorithmic gain is separate from the library
routing gains and is specific to this two-parameter model.

Validation uses Python 3.12.15 and the built-in factorization backend. Full pytest
passes with current NumPy 2.5.3 / SciPy 1.18.1 (9201 passed, two skipped), minimum
NumPy 2.1.3 / SciPy 1.14.1 / Matplotlib 3.9.4 (9196 passed, seven skipped), and a
clean `.[test]` environment with both NumPy and SymPy absent (3941 passed,
132 skipped). Markdown doctests pass in all three environments. Pyright
consumers are included in pytest; Ruff, formatting, test annotation checks,
`git diff --check` and strict MkDocs build also pass. Other Python versions and
the SymPy backend were not tested in this investigation.

## Shared operand boundary and contraction preparation

The 2026-10-30 experiment compares the existing uncommitted routing candidate
with a shared NumPy operand module and removal of the initial identity unit
product in contraction preparation. Source fingerprints, commands and samples
are under `operand_experiment` in [performance.json](performance.json).
This is a later experiment; the earlier tables retain their original snapshots.

Operand validation, normalization, scaling and alignment now live in
`_numerics/numpy_operands.py`, which imports no operation dispatcher or write
handlers. Construction and linear conversion can use that boundary without
loading the full dispatcher. Creation adapters no longer depend on the central
module; histogram, miscellaneous and write adapters still depend on it for
other handlers. This is a bounded dependency improvement, not a complete
reorganization of NumPy support. Public validation and staged outputs remain.

Contraction preparation uses the first operand's complete immutable Unit, then
multiplies later units. Trace and single-input einsum avoid one metadata
construction entirely; dot and multi-input einsum also avoid one construction.
Extra scales, independent named definitions and logarithmic dimensions remain
part of the retained Unit. No global cache or public API is added.

Python 3.12.15, NumPy 2.5.3 and the built-in factorization backend are used on
macOS ARM64. Baseline/candidate/candidate/baseline runs use separate source
exports with the same interpreter and script. Each workflow run takes seven
samples of 1500 calls; the table reports the median of two run medians in
microseconds. Inputs and fixed units are prepared before timing, physical
results are validated, and profiles and allocation tracing run separately.
Tests and builds were stopped during measurement.

| Operation | Baseline | Candidate | Elapsed change |
| --- | ---: | ---: | ---: |
| Trace | 8.855 | 5.708 | -35.5% |
| Single-input einsum view | 7.661 | 4.815 | -37.2% |
| Dot | 10.717 | 8.489 | -20.8% |
| Dot with bare first operand | 7.215 | 5.598 | -22.4% |
| Three-input power contraction | 17.930 | 15.724 | -12.3% |
| Circuit solve through integrated energy | 58.829 | 52.707 | -10.4% |
| Same circuit with scaled inputs | 83.599 | 75.161 | -10.1% |
| Conversion control | 5.811 | 5.751 | -1.0% |
| Short exponential-expression control | 15.454 | 15.169 | -1.8% |

The circuit follows the existing tested workflow: solve currents, broadcast
them over a time grid, contract currents and resistance into power, then
integrate energy. Both unit conventions produce 20 J. Untimed metadata counts
fall from five to four per workflow, from two to one for dot, and from one to
zero for trace and single-input einsum. These are combined implementation
measurements, not an isolated claim that moving functions between files saves
the reported time.

The complete decay fit remains effectively unchanged: finite differences take
1.155 / 1.154 ms and analytic Jacobians 0.945 / 0.945 ms, respectively. Short
array controls change between roughly -2.6% and +1.5% for mean, multiplication,
exponential ratio and same-unit addition. Million-element controls fluctuate:
addition changes from 410/436 to 380/373 microseconds across the two runs;
other operations do not show the same percentage. CPU frequency and other
processes are uncontrolled. These results establish neither a general speedup
nor a numerical-kernel improvement.

Validation uses Python 3.12.15 with current NumPy 2.5.3 / SciPy 1.18.1
(9206 passed, two skipped), minimum NumPy 2.1.3 / SciPy 1.14.1 / Matplotlib 3.9.4
(9201 passed, seven skipped), and a clean environment without NumPy or SymPy
(3941 passed, 132 skipped). Full Markdown doctests pass in all three.
Source Pyright and Ruff checks pass. Wheel and sdist contents include the new
module, stubs and py.typed and exclude local design notes. Installed scalar
runtime/types without NumPy and array/SciPy/Matplotlib consumers with current
and minimum dependencies are checked outside the source tree. Other Python
versions and SymPy were not tested in this investigation. No release is implied.

The recorded fingerprints and built artifacts identify the exported candidate
before the final Dimension construction/serialization cleanup and documentation
edits. They describe that measured snapshot, rather than a byte-identical copy
of the release source.
