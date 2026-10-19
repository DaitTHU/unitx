from decimal import Decimal
from typing import Final

from .._constnamespace import ConstNamespace
from ..quantity import quantity
from .bases import _SI
from .units import si

__all__ = ['SI', 'Particle']

_PI = Decimal('3.1415926535897932384626433832795028841971693993751')
"""the ratio of the circumference of a circle to its diameter"""
_WIEN_W = Decimal('4.9651142317442763036987591313228939440555849867973')
"""Approximate positive root of x = 5*(1 - exp(-x))."""
_WIEN_F = Decimal('2.8214393721220788934031913302944851953458817440731')
"""Approximate positive root of x = 3*(1 - exp(-x))."""


class SI(_SI):
    """
    The `SI` class contains common physical constants in SI units,
    like speed of light (_c_), Planck constant (_h_)...

    For example, the standard acceleration of gravity (_g_) is:
    >>> str(SI.g)
    '9.80665 m/s²'

    These constants can be directly used in formulas or code.

    Ref: https://physics.nist.gov/cuu/Constants/
    """

    # conventional reference values and derived constants

    hbar: Final = _SI.h / (2 * _PI)
    """reduced Planck constant, ħ: Final = h / 2π"""
    T0: Final = quantity(273.15, si.K)
    """standard temperature"""

    # universal

    G: Final = quantity(6.674_30e-11, 'm3/(kg*s2)')
    """Newtonian constant of gravitation"""
    kappa: Final = (8 * _PI * G / _SI.c**4).to(si.N**-1)
    """Einstein gravitational constant"""

    # electromagnetic constants

    mu0: Final = quantity(1.256_637_061_27e-6, 'H/m')
    """vacuum magnetic permeability, μ₀ ≈ 4π×10⁻⁷ H/m"""
    epsilon0: Final = (1 / (mu0 * _SI.c**2)).to('F/m')
    """vacuum electric permittivity"""
    eps0: Final = epsilon0
    """vacuum electric permittivity"""
    Z0: Final = (mu0 * _SI.c).to(si.ohm)  # simplify=True
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

    Vm: Final = (R * T0 / si.bar).to('m3/mol')
    """molar volume of ideal gas (273.15 K, 100 kPa)"""
    n0: Final = _SI.NA / Vm
    """Loschmidt constant (273.15 K, 100 kPa)"""


class Particle(ConstNamespace):
    """
    Common particle data, including mass and magnetic moment of
    electron (e), muon (μ), tau (τ), proton (p), neutron (n),
    deuteron (D), triton (T), helion (h) and alpha particle (α).

    Separated from `SI` class for symbol and namespace clarity.
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
