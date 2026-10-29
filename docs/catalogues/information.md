# Information units

```pycon
>>> from unitx.information import information as info
>>> from unitx.si import si
>>> (1 * info.MiB).value_in(info.B)
1048576
>>> (1 * info.MB).value_in(info.B)
1000000
>>> info.unit('kibibyte') == info.KiB
True
>>> (8 * info.MiB / si.s).value_in(info.Mibit / si.s)
64

```

`information` is a read-only `UnitCatalog` with an independent registry. It shares
SI seconds for rate expressions. B means byte here; B still means bel in bare
`Unit('B')` and `si.unit('B')`. Pass bound units to conversions, for example
`q.to(info.MiB)`, or parse explicitly with `info.unit('MiB/s')`. Conversion target
strings always use the default registry, even when the source is an information
quantity. This catalogue does not add a b alias that could be confused with barn.

## Scales and interpretation

| Attribute / symbol | Definition |
| --- | --- |
| `nat` | Natural unit of information; coherent dimensionless scale 1 |
| `bit`, `Sh` | `ln(2) nat`; bit and shannon share one formal definition |
| `byte`, `B` | Exactly `8 bit`; octet is an accepted name |
| `Hart` | Hartley, `ln(10) nat` |

These scales use the natural-log convention `H = -sum(p * ln(p)) nat`.
Thus a fair binary variable has entropy `1 bit`; expressing its entropy in
nats gives `ln(2)`. Storage sizes in bits/bytes describe capacity, rather than
the actual entropy of arbitrary file contents. No source distribution or
compression ratio is inferred. These are linear information quantities;
they do not use the logarithmic ratio degree of bel/neper or logarithmic points.

Information remains dimensionless in the existing physical model. `1 nat`
therefore equals `1 * si.one`; `1 bit` has coherent value `ln(2)`, rather than 1.
Dimension checking does not distinguish information from other dimensionless
quantities. This convention introduces no new dimension axis or conversion
context. Record it when exchanging numerical data with libraries that choose
a separate information dimension or normalize bit to one.

The factors `ln(2)` and `ln(10)` are symbolic, so algebraic cancellation is exact.
For example, byte-to-bit and MiB-to-byte conversions have exact integer or
Fraction results. Numerical evaluation of an uncancelled logarithm uses the
ordinary float approximation; Decimal input does not provide arbitrary-precision
logarithmic constants. Both factors are available from `unitx.factor`.

## Prefixes

Bit and byte accept decimal SI prefixes and binary prefixes. Nat and hartley
accept only decimal SI prefixes. Decimal multiples always use powers of ten:
`kB = 1000 B`, never 1024 B. Lowercase k is required; KB is invalid.

| Binary prefix | Factor |
| --- | --- |
| Ki (kibi) | `2**10` |
| Mi (mebi) | `2**20` |
| Gi (gibi) | `2**30` |
| Ti (tebi) | `2**40` |
| Pi (pebi) | `2**50` |
| Ei (exbi) | `2**60` |
| Zi (zebi) | `2**70` |
| Yi (yobi) | `2**80` |
| Ri (robi) | `2**90` |
| Qi (quebi) | `2**100` |

Namespace shortcuts include kbit through Qbit, kB through QB, Kibit through
Qibit, and KiB through QiB. Other permitted spellings use `info.unit()`, such as
`info.unit('mnat')`. Prefixes do not stack. `Kinat`, `KiHart` and `KiKiB` are
invalid. `compact()` respects these permissions and can select binary prefixes.

Definitions follow the bit/byte convention in
[NIST's binary-prefix reference](https://pml.nist.gov/cuu/Units/binary.html).
The full binary range is listed by the
[BIPM SI Brochure](https://www.bipm.org/en/publications/si-brochure), referring to
IEC 80000-13. Natural-log normalization is the catalogue's explicit numerical
convention; no independent information-kind checking is promised.

## Persistence

Use the normal scalar JSON format with an explicitly prepared receiving registry:

```pycon
>>> from unitx import Registry
>>> from unitx.serialization import to_dict, from_dict
>>> receiving = Registry((info.B,))
>>> restored = from_dict(to_dict(3 * info.MiB), registry=receiving)
>>> restored == 3 * info.MiB
True

```

The definition IDs belong to this catalogue; a default-registry B is a different
unit and cannot decode a byte record. See [scalar exchange](../serialization.md)
for the environment requirements. The JSON schema remains version 2.

[Catalogue index](README.md) · [Custom prefix permissions](../custom-units.md#prefix-permissions)
