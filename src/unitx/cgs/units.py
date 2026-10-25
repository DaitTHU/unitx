"""Mechanical CGS and Gaussian catalogues with independent parsing."""

from fractions import Fraction
from typing import Final

from .._electromagnetism.coefficients import _EMCoefficients
from .._electromagnetism.conversion import _GaussianConversion
from ..quantity import Quantity
from ..si import si
from ..unit import Unit
from ..unitcatalog import UnitCatalog
from ..unitregistry import Registry

__all__ = ['cgs', 'gaussian']


with Registry(prefixable=True).defining() as _registry:

    class cgs(UnitCatalog, units=(si.one,), registry=_registry):
        """Mechanical CGS units in an independent parser.

        New entries accept SI prefixes on their symbols and singular names.
        Shared SI entries retain their existing spelling and prefix policies.
        Mechanical scales follow NIST SP 811, table 10; barye follows dyn/cm2.
        https://www.nist.gov/pml/special-publication-811/nist-guide-si-chapter-5-units-outside-si
        """

        cm: Final = si.cm
        """Centimetre, `1 cm = 0.01 m`; shared SI definition."""
        g: Final = si.g
        """Gram, `1 g = 0.001 kg`; shared SI definition."""
        s: Final = si.s
        """Second, the shared SI time unit."""
        Gal: Final = si.Gal
        """Galileo, `1 Gal = 1 cm/s**2`; shared SI definition."""

        dyn: Final = (g * cm / s**2).define('dyn', 'dyne')
        """Dyne, `1 dyn = 1 g*cm/s**2 = 1e-5 N`; force."""
        erg: Final = (dyn * cm).define('erg', 'erg')
        """Erg, `1 erg = 1 dyn*cm = 1e-7 J`; energy."""
        Ba: Final = (dyn / cm**2).define('Ba', 'barye')
        """Barye, `1 Ba = 1 dyn/cm**2 = 0.1 Pa`; pressure."""
        P: Final = (dyn * s / cm**2).define('P', 'poise')
        """Poise, `1 P = 1 dyn*s/cm**2 = 0.1 Pa*s`; dynamic viscosity."""
        St: Final = (cm**2 / s).define('St', 'stokes')
        """Stokes, `1 St = 1 cm**2/s = 1e-4 m**2/s`; kinematic viscosity."""


with Registry(
    unit for unit in vars(cgs).values() if isinstance(unit, Unit)
).defining() as _gaussian_registry:

    class gaussian(cgs, _GaussianConversion, registry=_gaussian_registry):
        """Gaussian units, explicit SI correspondence and formula coefficients.

        Inherit the mechanical CGS units unchanged. This independent parser also
        includes those shared definitions; it never adds electromagnetic names
        to the mechanical catalogue or the default environment.

        Units use exact half-integer mechanical dimensions. `convert()` uses
        CODATA 2022 SI data; `em` states the Gaussian formula coefficients.
        Sources: NIST SP 811, table 10; Kruger (2019), section II.
        https://www.nist.gov/pml/special-publication-811/nist-guide-si-chapter-5-units-outside-si
        https://arxiv.org/abs/1909.11264
        https://physics.nist.gov/cuu/pdf/all.pdf
        """

        statC: Final = (cgs.g ** Fraction(1, 2) * cgs.cm ** Fraction(3, 2) / cgs.s).define(
            'statC', 'statcoulomb'
        )
        """Statcoulomb, `g**(1/2)*cm**(3/2)/s`; Gaussian electric charge."""
        statA: Final = (statC / cgs.s).define('statA', 'statampere')
        """Statampere, `1 statA = 1 statC/s`; Gaussian electric current."""
        statV: Final = (cgs.erg / statC).define('statV', 'statvolt')
        """Statvolt, `1 statV = 1 erg/statC`; Gaussian electric potential."""
        statF: Final = (statC / statV).define('statF', 'statfarad')
        """Statfarad, `1 statF = 1 statC/statV = 1 cm`; Gaussian capacitance."""
        G: Final = (cgs.dyn / statC).define('G', 'gauss')
        """Gauss, `1 G = 1 dyn/statC`; Gaussian magnetic flux density B."""
        Mx: Final = (G * cgs.cm**2).define('Mx', 'maxwell')
        """Maxwell, `1 Mx = 1 G*cm**2`; Gaussian magnetic flux."""
        Oe: Final = G.define('Oe', 'oersted')
        """Oersted, same mechanical scale as G, distinct identity; magnetic H."""

        em: Final[_EMCoefficients] = _EMCoefficients(
            k1=Quantity(1.0, si.one),
            alpha=Quantity(1.0 / si.em.c.value, si.em.c.unit**-1),
            c=si.em.c,
        )
        """Gaussian coefficients: `k1=1`, `alpha=1/c`, derived `k2=1/c`.
        Float Quantities using the BIPM vacuum light speed. Convention reference:
        https://arxiv.org/abs/1909.11264 (section II and table II).
        """
