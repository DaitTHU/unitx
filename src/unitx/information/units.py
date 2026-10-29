"""Information units in an independent spelling environment.

Import the catalogue with `from unitx.information import information`.
"""

from typing import Final

from ..factor import ln2, ln10
from ..si import si
from ..unitcatalog import UnitCatalog, _prefixed_units
from ..unitdefinition import PrefixFamily

__all__ = ['information']

_BINARY_MULTIPLES = ('Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi', 'Yi', 'Ri', 'Qi')


class information(UnitCatalog):
    """Dimensionless information scales using natural-log normalization.

    `1 nat = 1`, `1 bit = ln(2) nat`, `1 Hart = ln(10) nat`, and
    `1 byte = 8 bit`. Bit counts describe storage capacity; actual entropy
    also depends on the source distribution. This catalogue does not infer
    entropy from a file's size or provide logarithmic point coordinates.

    Decimal prefixes always mean powers of ten; IEC binary prefixes Ki
    through Qi mean powers of two and apply only to bit and byte here.
    B denotes byte in this registry; the default SI registry retains bel.
    https://pml.nist.gov/cuu/Units/binary.html
    https://www.iec.ch/prefixes-binary-multiples
    """

    one: Final = si.one
    s: Final = si.s

    nat: Final = one.define('nat', 'nat', 'natural unit of information')
    """Natural information unit; `1 nat = 1` in natural-log normalization."""
    bit: Final = (ln2 * nat).define(
        'bit', 'bit', 'shannon', symbol_aliases='Sh', prefixable=PrefixFamily.ALL
    )
    """Binary information unit; `1 bit = ln(2) nat`, also called shannon."""
    Sh: Final = bit
    """Shannon, sharing the bit definition and its exact logarithmic scale."""
    byte: Final = (8 * bit).define('B', 'byte', 'octet', prefixable=PrefixFamily.ALL)
    """Eight-bit byte; `1 B = 8 bit` exactly."""
    B: Final = byte
    Hart: Final = (ln10 * nat).define('Hart', 'hartley')
    """Hartley; `1 Hart = ln(10) nat = log2(10) bit`."""

    kbit, Mbit, Gbit, Tbit, Pbit, Ebit, Zbit, Ybit, Rbit, Qbit = _prefixed_units(bit, 'kMGTPEZYRQ')
    """Decimal bit multiples, from `10**3 bit` through `10**30 bit`."""
    kB, MB, GB, TB, PB, EB, ZB, YB, RB, QB = _prefixed_units(B, 'kMGTPEZYRQ')
    """Decimal byte multiples; `1 kB = 1000 B`."""
    Kibit, Mibit, Gibit, Tibit, Pibit, Eibit, Zibit, Yibit, Ribit, Qibit = _prefixed_units(
        bit, _BINARY_MULTIPLES
    )
    """Binary bit multiples, from `2**10 bit` through `2**100 bit`."""
    KiB, MiB, GiB, TiB, PiB, EiB, ZiB, YiB, RiB, QiB = _prefixed_units(B, _BINARY_MULTIPLES)
    """Binary byte multiples; `1 KiB = 1024 B`."""
