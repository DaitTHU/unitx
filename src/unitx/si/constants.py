from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from typing import Final

from .._constnamespace import ConstNamespace
from ..quantity import quantity
from .bases import _SI
from .units import si

__all__ = ['SI', 'Particle', 'Planck', 'au', 'nu']

_HALF = Decimal('0.5')
_PI = Decimal('3.1415926535897932384626433832795028841971693993751')
"""the ratio of the circumference of a circle to its diameter"""
_WIEN_W = Decimal('4.9651142317442763036987591313228939440555849867973')
"""Approximate positive root of x = 5*(1 - exp(-x))."""
_WIEN_F = Decimal('2.8214393721220788934031913302944851953458817440731')
"""Approximate positive root of x = 3*(1 - exp(-x))."""


# Catalogue values are fixed at import, independent of caller precision, traps,
# flags and exponent limits. Later user arithmetic keeps the user's context.
with localcontext(
    Context(
        prec=28,
        rounding=ROUND_HALF_EVEN,
        Emin=-999999,
        Emax=999999,
        capitals=1,
        clamp=0,
        flags=[],
        traps=[InvalidOperation, DivisionByZero, Overflow],
    )
):

    class SI(_SI):
        """Physical constants as SI Quantities, with nominal measured values.

        Defining constants follow the BIPM SI Brochure; measured inputs use CODATA
        2022 central values. Derived values use an isolated 28-digit ROUND_HALF_EVEN context and
        do not propagate uncertainty. `SI.alpha` is the fine-structure constant;
        formula coefficients are grouped separately in `si.em`.
        https://www.bipm.org/en/publications/si-brochure
        https://physics.nist.gov/cuu/pdf/all.pdf

        >>> str(SI.g)
        '9.80665 m/s²'
        """

        hbar: Final = _SI.h / (2 * _PI)
        """Reduced Planck constant, `hbar = h/(2*pi)`."""

        # universal

        G: Final = quantity(6.674_30e-11, 'm3/(kg*s2)')
        """Newtonian constant of gravitation"""
        kappa: Final = (8 * _PI * G / _SI.c**4).to(si.N**-1)
        """Einstein gravitational constant"""

        # electromagnetic constants

        epsilon0: Final = (1 / (_SI.mu0 * _SI.c**2)).to('F/m')
        """vacuum electric permittivity"""
        eps0: Final = epsilon0
        """vacuum electric permittivity"""
        Z0: Final = (_SI.mu0 * _SI.c).to(si.ohm)  # simplify=True
        """characteristic impedance of vacuum"""
        ke: Final = (1 / (4 * _PI * epsilon0)).to('N*m2/C2')
        """Coulomb constant"""
        KJ: Final = (2 * _SI.e / _SI.h).to('Hz/V')
        """Josephson constant"""
        Phi0: Final = (1 / KJ).to(si.Wb)  # simplify=True
        """magnetic flux quantum"""
        G0: Final = (2 * _SI.e**2 / _SI.h).to(si.S)  # simplify=True
        """conductance quantum"""
        RK: Final = (_SI.h / _SI.e**2).to(si.ohm)  # simplify=True
        """von Klitzing constant"""

        # atomic and nuclear

        me: Final = quantity(9.109_383_7139e-31, si.kg)
        """electron mass"""
        mp: Final = quantity(1.672_621_925_95e-27, si.kg)
        """proton mass"""
        mn: Final = quantity(1.674_927_500_56e-27, si.kg)
        """neutron mass"""

        alpha: Final = (_SI.e**2 / (2 * eps0 * _SI.h * _SI.c)).to(si.one)  # simplify=True
        """fine-structure constant, α ≈ 1/137"""
        a0: Final = (hbar / (alpha * me * _SI.c)).to(si.m)  # simplify=True
        """Bohr radius"""
        lambdaC: Final = (_SI.h / (me * _SI.c)).to(si.m)  # simplify=True
        """Compton wavelength"""
        Rinf: Final = alpha**2 / (2 * lambdaC)
        """Rydberg constant"""
        Eh: Final = (me * (alpha * _SI.c) ** 2).to(si.J)  # simplify=True
        """Hartree energy"""
        re: Final = alpha**2 * a0
        """classical electron radius"""
        sigmae: Final = 8 * _PI / 3 * re**2
        """Thomson cross section"""
        muB: Final = (_SI.e * hbar / (2 * me)).to('J/T')
        """Bohr magneton"""
        muN: Final = (_SI.e * hbar / (2 * mp)).to(muB.unit)
        """nuclear magneton"""

        # physico-chemical

        Mu: Final = (_SI.NA * si.Da).to('kg/mol')
        """molar mass constant"""
        R: Final = _SI.kB * _SI.NA
        """molar gas constant"""
        F: Final = _SI.NA * _SI.e
        """Faraday constant"""
        sigma: Final = (_PI**2 * _SI.kB**4 / (60 * hbar**3 * _SI.c**2)).to('W/(m2*K4)')
        """Stefan-Boltzmann constant"""
        c1L: Final = (2 * _SI.h * _SI.c**2).to('W*m2/sr')
        """first radiation constant for spectral radiance"""
        c1: Final = c1L * _PI * si.sr
        """first radiation constant"""
        c2: Final = _SI.h * _SI.c / _SI.kB
        """second radiation constant"""
        b: Final = c2 / _WIEN_W
        """Wien wavelength displacement law constant"""
        b_: Final = (_WIEN_F * _SI.c / c2).to('Hz/K')
        """Wien frequency displacement law constant"""

        Vm: Final = (R * _SI.T0 / si.bar).to('m3/mol')
        """molar volume of ideal gas (273.15 K, 100 kPa)"""
        n0: Final = _SI.NA / Vm
        """Loschmidt constant (273.15 K, 100 kPa)"""

    class Particle(ConstNamespace):
        """Particle masses and magnetic moments as SI Quantities.

        Nominal inputs follow the CODATA 2022 recommended particle data; g-factors
        are derived using the indicated magneton and sign convention. No uncertainty
        propagation is performed. Shared entries reuse the corresponding `SI` data.
        https://physics.nist.gov/cuu/pdf/all.pdf
        """

        me: Final = SI.me
        """electron (e) mass"""
        mmu: Final = quantity(1.883_531_627e-28, si.kg)
        """muon (μ) mass"""
        mtau: Final = quantity(3.167_54e-27, si.kg)
        """tau (τ) mass"""
        mp: Final = SI.mp
        """proton (p = ¹H) mass"""
        mn: Final = SI.mn
        """neutron (n) mass"""
        mD: Final = quantity(3.343_583_7768e-27, si.kg)
        """deuteron (D = ²H) mass"""
        mT: Final = quantity(5.007_356_7512e-27, si.kg)
        """triton (T = ³H) mass"""
        mh: Final = quantity(5.006_412_7862e-27, si.kg)
        """helion (h = ³He) mass"""
        malpha: Final = quantity(6.644_657_3450e-27, si.kg)
        """alpha (α = ⁴He) particle mass"""

        mue: Final = quantity(-9.284_764_6917e-24, SI.muB.unit)
        """electron magnetic moment"""
        ge: Final = -2 * mue / SI.muB
        """electron g-factor"""
        mumu: Final = quantity(-4.490_448_30e-26, mue.unit)
        """muon magnetic moment"""
        gmu: Final = -2 * mumu / (SI.e * SI.hbar / (2 * mmu))
        """muon g-factor"""
        mup: Final = quantity(1.410_606_795_45e-26, mue.unit)
        """proton magnetic moment"""
        gp: Final = 2 * mup / SI.muN
        """proton g-factor"""
        mun: Final = quantity(-9.662_3653e-27, mue.unit)
        """neutron magnetic moment"""
        gn: Final = 2 * mun / SI.muN
        """neutron g-factor"""
        muD: Final = quantity(4.330_735_087e-27, mue.unit)
        """deuteron magnetic moment"""
        gD: Final = muD / SI.muN
        """deuteron g-factor"""
        muT: Final = quantity(1.504_609_5178e-26, mue.unit)
        """triton magnetic moment"""
        gT: Final = 2 * muT / SI.muN
        """triton g-factor"""
        muh: Final = quantity(-1.074_617_551_98e-26, mue.unit)
        """helion magnetic moment"""
        gh: Final = 2 * muh / SI.muN
        """helion g-factor"""

    class au(ConstNamespace):
        """Hartree atomic-unit scales expressed as SI Quantities.

        Set ħ = e = mₑ = 4πε₀ = 1; the speed of light is then 1/α.
        Attributes are physical scales, not registered Unit objects. Divide a
        quantity by the corresponding scale and convert to '' to obtain its
        numerical value in atomic units. This does not change conversion rules.

        The magnetic dipole scale is e ħ / mₑ = 2 μB, not the Bohr magneton.
        Precision follows the underlying SI constants; uncertainties are not propagated.

        Physical inputs are the CODATA 2022 constants exposed by `SI`.
        https://physics.nist.gov/cuu/pdf/all.pdf
        """

        # defined base

        charge: Final = SI.e
        action: Final = SI.hbar
        mass: Final = SI.me
        permittivity: Final = 4 * _PI * SI.epsilon0

        # derived dynamics

        length: Final = SI.a0  # Bohr radius
        energy: Final = SI.Eh  # Hartree energy
        time: Final = action / energy
        force: Final = energy / length
        velocity: Final = length / time
        momentum: Final = mass * velocity
        temperature: Final = energy / SI.kB

        # derived electromagnetic

        current: Final = (charge / time).to(si.A)
        charge_density: Final = charge / length**3
        electric_potential: Final = (energy / charge).to(si.V)
        electric_field: Final = electric_potential / length
        electric_dipole_moment: Final = charge * length
        electric_quadrupole_moment: Final = charge * length**2
        electric_polarizability: Final = electric_dipole_moment / electric_field
        magnetic_flux_density: Final = (electric_field / velocity).to(si.T)
        magnetic_dipole_moment: Final = (electric_dipole_moment * velocity).to(SI.muB.unit)
        magnetizability: Final = magnetic_dipole_moment / magnetic_flux_density

    class nu(ConstNamespace):
        """Electron-mass natural-unit scales expressed as SI Quantities.

        Set c = ħ = mₑ = ε₀ = 1, with charge scale sqrt(ε₀ ħ c).
        The numerical elementary charge is then sqrt(4πα). The length scale is
        the reduced electron Compton wavelength, not the ordinary wavelength.

        Attributes are physical scales, not registered Unit objects or an active
        conversion context. Precision follows the underlying SI constants.

        Physical inputs are the CODATA 2022 constants exposed by `SI`.
        https://physics.nist.gov/cuu/pdf/all.pdf
        """

        # defined base

        velocity: Final = SI.c
        action: Final = SI.hbar
        mass: Final = SI.me
        permittivity: Final = SI.epsilon0

        # derived dynamics

        energy: Final = (mass * velocity**2).to(si.J)
        momentum: Final = mass * velocity
        length: Final = (action / momentum).to(si.m)
        time: Final = length / velocity
        force: Final = (energy / length).to(si.N)

        # derived electromagnetic

        charge: Final = ((permittivity * action * velocity) ** _HALF).to(si.C)
        current: Final = (charge / time).to(si.A)
        charge_density: Final = charge / length**3
        electric_potential: Final = (energy / charge).to(si.V)
        electric_field: Final = electric_potential / length
        electric_dipole_moment: Final = charge * length
        electric_quadrupole_moment: Final = charge * length**2
        electric_polarizability: Final = electric_dipole_moment / electric_field
        magnetic_flux_density: Final = (electric_field / velocity).to(si.T)
        magnetic_dipole_moment: Final = (electric_dipole_moment * velocity).to(SI.muB.unit)
        magnetizability: Final = magnetic_dipole_moment / magnetic_flux_density

    class Planck(ConstNamespace):
        """Planck mechanical and temperature scales expressed as SI Quantities.

        Set c = ħ = G = kB = 1, using ħ rather than h. These scales do not
        select an electromagnetic convention or define a Planck charge.

        Attributes are physical scales, not registered Unit objects or an active
        conversion context. Precision follows the underlying SI constants;
        derived values do not carry an uncertainty model.

        Physical inputs are the CODATA 2022 constants exposed by `SI`.
        https://physics.nist.gov/cuu/pdf/all.pdf
        """

        # defined base

        velocity: Final = SI.c
        action: Final = SI.hbar

        # derived base

        mass: Final = ((action * velocity / SI.G) ** _HALF).to(si.kg)
        energy: Final = (mass * velocity**2).to(si.J)
        momentum: Final = mass * velocity
        length: Final = (action / momentum).to(si.m)
        time: Final = (length / velocity).to(si.s)

        temperature: Final = energy / SI.kB
