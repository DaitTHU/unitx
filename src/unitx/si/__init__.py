"""SI units, physical constants and SI-valued physical scales.

Import `si` for units and coordinates, `SI` for physical constants, and
`Particle`, `au`, `nu` or `Planck` for particle data and named scale conventions.
`si.em` groups float-valued coefficients for common electromagnetic formulas.

Sources: BIPM SI Brochure, ninth edition, for SI definitions and prefixes;
NIST SP 811 for common non-SI units; CODATA 2022 for measured constant data.
Individual entries state library conventions and approximate definitions.
Measured values are nominal; uncertainties are not propagated.
https://www.bipm.org/en/publications/si-brochure
https://www.nist.gov/pml/special-publication-811/nist-guide-si-chapter-5-units-outside-si
https://physics.nist.gov/cuu/pdf/all.pdf
"""

from .constants import SI, Particle, Planck, au, nu
from .units import si

__all__ = ['SI', 'Particle', 'Planck', 'au', 'nu', 'si']
