# FFT and frequency axes

NumPy's discrete Fourier transforms preserve a linear Quantity's complete
unit, including exact extra scales. They do not insert a sampling interval.
Normalization by the sample count (`norm=None`, `'backward'`, `'forward'` or
`'ortho'`) is dimensionless, so it does not change the unit. Affine and
logarithmic points are unsupported; convert points to meaningful differences
or physical quantities before transforming.

## Transforms

| NumPy functions | Options |
| --- | --- |
| `fft`, `ifft`, `rfft`, `irfft`, `hfft`, `ihfft` in `np.fft` | n, axis, norm, out |
| `fft2`, `ifft2`, `rfft2`, `irfft2`, `fftn`, `ifftn`, `rfftn`, `irfftn` in `np.fft` | s, axes, norm, out |
| `np.fft.fftshift`, `np.fft.ifftshift` | axes; preserve unit or point coordinate |

NumPy controls lengths, axes, padding/cropping, Hermitian conventions, dtype
promotion and numerical behavior. Options follow the installed NumPy version,
including its warnings for deprecated combinations. Odd-length real-transform
round trips require the original length on the inverse call. The first
transform input can be positional or supplied as `a=`.

```python
>>> import numpy as np
>>> from unitx import Quantity
>>> from unitx.si import si
>>> x = Quantity(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), si.m)
>>> spectrum = np.fft.rfft(x)
>>> assert spectrum.unit == si.m
>>> recovered = np.fft.irfft(spectrum, n=x.size)
>>> np.testing.assert_allclose(recovered.value_in(si.m), x.value)

```

`fftshift` and `ifftshift` only reorder samples and also support points.
They preserve the input unit or coordinate, follow NumPy's axis rules, and
allocate according to NumPy's storage behavior. They do not assign a time or
frequency origin. For odd lengths the two shifts differ by one sample, so use
the inverse function to undo a shift.

## Output buffers

Transform `out` accepts a Quantity backed by a writable plain numeric ndarray
with exactly the result shape and a compatible linear unit. The result is
computed without a native output buffer, converted to the target unit, and
cast with `same_kind` before writing. The target dtype does not select the
FFT kernel dtype. This is the existing [staged output policy](writes.md#output-buffers),
including support for overlapping input/output storage and failure invariance
for shape, unit, casting, read-only, computation and conversion errors.

```python
>>> output = Quantity(np.empty(3, dtype=complex), si.cm)
>>> assert np.fft.rfft(x, out=output) is output
>>> np.testing.assert_allclose(output.value_in(si.m), spectrum.value)

```

`out` is accepted positionally or by keyword, but only once. `out=None`
allocates a result. Bare ndarray outputs cannot silently discard units.
`fftshift` and `ifftshift` have no `out` argument.

## Frequency axes and integration scales

`np.fft.fftfreq(n, d=spacing)` and `np.fft.rfftfreq(n, d=spacing)` work through
ordinary Quantity arithmetic, without a unitx wrapper or monkey-patching
NumPy. Use a nonzero real scalar linear Quantity for spacing. The result unit
is exactly `spacing.unit**-1`; a time interval produces frequencies in cycles
per time, and a spatial interval produces cycles per length. These are not
angular frequencies: multiply by `2*pi` explicitly when needed. Bare spacing
retains NumPy's ordinary ndarray result. NumPy controls n and device handling.

```python
>>> spacing = 2 * si.ms
>>> frequency = np.fft.rfftfreq(x.size, d=spacing)
>>> assert frequency.unit == si.ms**-1
>>> np.testing.assert_allclose(frequency.value_in(si.Hz), [0, 100, 200])
>>> centered = np.fft.fftshift(np.fft.fftfreq(x.size, d=spacing))
>>> np.testing.assert_allclose(centered.value_in(si.Hz), [-200, -100, 0, 100, 200])
>>> integral_spectrum = spectrum * spacing
>>> assert integral_spectrum.unit == si.m * si.ms

```

For the default forward normalization, `spacing * np.fft.fft(x)` includes
the integration scale in a sampled continuous-transform approximation.
Frequency generation and the discrete transform remain separate operations;
neither infers the other's sampling interval, window, origin or convention.
NumPy's static annotations may still need local casts at dispatch boundaries
and can reject Quantity spacing or output arguments despite runtime support;
see the [typing guide](../typing.md#numpy-component-inference).

References: [NumPy FFT](https://numpy.org/doc/stable/reference/routines.fft.html),
[transform options](https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html),
[frequency convention](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html).

[NumPy support matrix](../numpy.md) · [Documentation index](../README.md)
