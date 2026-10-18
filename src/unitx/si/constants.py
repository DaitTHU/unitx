import math
from decimal import Decimal
from typing import Final

from .._constclass import ConstClass
from ..quantity import Quantity, quantity
from .units import si

__all__ = ['SI', 'Particle']

# Match the constants' Decimal magnitudes when using the approximate value of pi.
_PI = Decimal(str(math.pi))


class SI(ConstClass):
    """
    The `SI` class contains common physical quantitys in SI units,
    like speed of light (_c_), Planck quantity (_h_)...

    For example, the standard acceleration of gravity (_g_) is:
    >>> str(SI.g)
    '9.80665 m/s²'

    These quantitys can be directly used in formulas or code.

    Ref: http://physics.nist.gov/quantitys
    """

    # exact quantitys defined by SI

    DeltanuCs: Final = Quantity(9_192_631_770, si.Hz)
    """hyperfine transition frequency of Cs-133, definition of second."""
    c: Final = Quantity(299_792_458, 'm/s')
    """speed of light in vacuum"""
    h: Final = quantity(6.626_070_15e-34, 'J*s')
    """Planck quantity"""
    e: Final = quantity(1.602_176_634e-19, si.C)
    """elementary charge"""
    kB: Final = quantity(1.380_649e-23, 'J/K')
    """Boltzmann quantity"""
    NA: Final = quantity(6.022_140_76e23, 'mol-1')
    """Avogadro quantity"""
    Kcd: Final = Quantity(683, 'lm/W')
    """luminous efficacy of 540 THz radiation, definition of candela"""

    hbar: Final = h / (2 * _PI)
    """reduced Planck quantity, ħ: Final = h / 2π"""
    g: Final = quantity(9.80665, 'm/s2')
    """standard acceleration of gravity"""
    T0: Final = quantity(273.15, si.K)
    """standard temperature"""

    # universal

    G: Final = quantity(6.674_30e-11, 'm3/(kg*s2)')
    """Newtonian quantity of gravitation"""
    # kappa: Final = 8 * _PI * G / c**4  # simplify=True
    # """Einstein gravitational quantity"""
    # H0: Final = Quantity(70, 'km/s*Mpc')
    # """Hubble quantity, approx"""
    # Lambda: Final = quantity(1.089e-52, 'm-2')
    # """cosmological quantity, approx"""

    # electromagnetic quantitys

    mu0: Final = quantity(1.256_637_061_27e-6, 'H/m')
    """vacuum magnetic permeability, μ₀ ≈ 4π×10⁻⁷ H/m"""
    epsilon0: Final = (1 / (mu0 * c**2)).to('F/m')
    """vacuum electric permittivity"""
    eps0: Final = epsilon0
    """vacuum electric permittivity"""
    Z0: Final = (mu0 * c).to(si.ohm)  # simplify=True
    """characteristic impedance of vacuum"""
    ke: Final = (1 / (4 * _PI * epsilon0)).to('N*m2/C2')
    """Coulomb quantity"""
    KJ: Final = (2 * e / h).to('Hz/V')
    """Josephson quantity"""
    Phi0: Final = (1 / KJ).to(si.Wb)  # simplify=True
    """magnetic flux quantum"""
    G0: Final = (2 * e**2 / h).to(si.S)  # simplify=True
    """conductance quantum"""
    RK: Final = (h / e**2).to(si.ohm)  # simplify=True
    """von Klitzing quantity"""

    # atomic and nuclear

    me: Final = quantity(9.109_383_7139e-31, si.kg)
    """electron mass"""
    mp: Final = quantity(1.672_621_925_95e-27, si.kg)
    """proton mass"""
    mn: Final = quantity(1.674_927_500_56e-27, si.kg)
    """neutron mass"""

    alpha: Final = (e**2 / (2 * epsilon0 * h * c)).to(si.one)  # simplify=True
    """fine-structure quantity, α ≈ 1/137"""
    a0: Final = (hbar / (alpha * me * c)).to(si.m)  # simplify=True
    """Bohr radius"""
    lambdaC: Final = (h / (me * c)).to(si.m)  # simplify=True
    """Compton wavelength"""
    Rinf: Final = alpha**2 / (2 * lambdaC)
    """Rydberg quantity"""
    Eh: Final = (me * (alpha * c) ** 2).to(si.J)  # simplify=True
    """Hartree energy"""
    re: Final = alpha**2 * a0
    """classical electron radius"""
    sigmae: Final = 8 * _PI / 3 * re**2
    """Thomson cross section"""
    muB: Final = (e * hbar / (2 * me)).to('J/T')
    """Bohr magneton"""
    muN: Final = (e * hbar / (2 * mp)).to(muB.unit)
    """nuclear magneton"""

    # physico-chemical

    Mu: Final = (NA * si.Da).to('kg/mol')
    """molar mass quantity"""
    R: Final = kB * NA
    """molar gas quantity"""
    F: Final = NA * e
    """Faraday quantity"""

    sigma: Final = (_PI**2 * kB**4 / (60 * hbar**3 * c**2)).to('W/(m2*K4)')
    """Stefan-Boltzmann quantity"""
    c1L: Final = (2 * h * c**2).to('W*m2/sr')
    """first radiation quantity for spectral radiance"""
    c1: Final = c1L * _PI * si.sr
    """first radiation quantity"""
    c2: Final = h * c / kB
    """second radiation quantity"""
    # b: Final = c2 / Math.WEIN_W
    # """Wien wavelength displacement law quantity"""
    # b_: Final = (Math.WEIN_F * c / c2).to('Hz/K')
    # """Wien frequency displacement law quantity"""

    Vm: Final = (R * T0 / si.bar).to('m3/mol')
    """molar volume of ideal gas (273.15 K, 100 kPa)"""
    # Vmatm: Final = (R * T0 / si.atm).to(Vm.unit)
    # """molar volume of ideal gas (273.15 K, 101.325 kPa)"""
    n0: Final = NA / Vm
    """Loschmidt quantity (273.15 K, 100 kPa)"""
    # n0atm: Final = NA / Vmatm
    # """Loschmidt quantity (273.15 K, 101.32 kPa)"""


class Particle(ConstClass):
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
    """proton (p: Final = ¹H) mass"""
    mn: Final = SI.mn
    """neutron (n) mass"""
    mD: Final = quantity(3.343_583_7768e-27, si.kg)
    """deuteron (D: Final = ²H) mass"""
    mT: Final = quantity(5.007_356_7512e-27, si.kg)
    """triton (T: Final = ³H) mass"""
    mh: Final = quantity(5.006_412_7862e-27, si.kg)
    """helion (h: Final = ³He) mass"""
    malpha: Final = quantity(6.644_657_3450e-27, si.kg)
    """alpha (α: Final = ⁴He) particle mass"""

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
