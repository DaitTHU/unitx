"""Exact mechanical CGS units and four electromagnetic conventions.

Import with `from unitx.cgs import cgs, gaussian, esu, emu, hl`.
`cgs` contains mechanical units; each electromagnetic catalogue inherits them
and adds units, `em` formula coefficients and explicit
`convert(quantity, target)` correspondence with SI.

Mechanical scales follow NIST SP 811, table 10, or the defining CGS algebra.
Electromagnetic conventions follow Kruger (2019), section II; this library calls
its Lorentz coefficient `alpha` rather than `k3`. SI correspondence uses the
CODATA 2022 central value of mu0 and the BIPM defining speed of light. It is
approximate, whereas mechanical unit definitions remain exact.
https://www.nist.gov/pml/special-publication-811/nist-guide-si-chapter-5-units-outside-si
https://arxiv.org/abs/1909.11264
https://physics.nist.gov/cuu/pdf/all.pdf
"""

from .units import cgs, emu, esu, gaussian, hl

__all__ = ['cgs', 'gaussian', 'esu', 'emu', 'hl']
