"""Shared SI roots and physical inputs required during catalogue construction."""

from typing import Final

from .._constnamespace import ConstNamespace
from ..quantity import Quantity, quantity
from ..unit import Unit
from ..unitcatalog import UnitCatalog
from ..unitregistry import default_registry


class _si(UnitCatalog, registry=default_registry):
    """Coherent SI units, following the BIPM SI Brochure, ninth edition.

    https://www.bipm.org/en/publications/si-brochure
    """

    # SI base unit

    one: Final = Unit()
    """Dimensionless multiplicative identity, `1`."""
    s: Final = Unit('s')
    """Second; unperturbed Cs-133 ground-state hyperfine frequency is 9192631770 Hz exactly."""
    m: Final = Unit('m')
    """Metre; vacuum light speed is exactly 299792458 m/s."""
    kg: Final = Unit('kg')
    """Kilogram; Planck constant is exactly `6.62607015e-34 J*s`."""
    A: Final = Unit('A')
    """Ampere; elementary charge is exactly `1.602176634e-19 C`, with `C = A*s`."""
    K: Final = Unit('K')
    """Kelvin; Boltzmann constant is exactly `1.380649e-23 J/K`."""
    mol: Final = Unit('mol')
    """Mole; Avogadro constant is exactly `6.02214076e23 /mol`."""
    cd: Final = Unit('cd')
    """Candela; luminous efficacy at 540 THz is exactly `683 lm/W`."""

    # SI derived unit

    rad: Final = one.define('rad', name='radian', symbol_aliases='㎭')
    """Radian, plane angle given by arc length/radius; `1 rad = 1 m/m`."""
    sr: Final = one.define('sr', name='steradian', symbol_aliases='㏛')
    """Steradian, solid angle given by area/radius squared; `1 sr = 1 m**2/m**2`."""
    Hz: Final = (s**-1).define('Hz', name='hertz', symbol_aliases='㎐')
    """Hertz, `1 Hz = 1/s`; frequency."""
    N: Final = (kg * m / s**2).define('N', name='newton')
    """Newton, `1 N = 1 kg*m/s**2`; force."""
    Pa: Final = (N / m**2).define('Pa', name='pascal', symbol_aliases='㎩')
    """Pascal, `1 Pa = 1 N/m**2`; pressure or stress."""
    J: Final = (N * m).define('J', name='joule')
    """Joule, `1 J = 1 N*m`; energy, work or heat."""
    W: Final = (J / s).define('W', name='watt')
    """Watt, `1 W = 1 J/s`; power."""
    C: Final = (A * s).define('C', name='coulomb')
    """Coulomb, `1 C = 1 A*s`; electric charge."""
    V: Final = (W / A).define('V', name='volt')
    """Volt, `1 V = 1 W/A`; electric potential difference."""
    F: Final = (C / V).define('F', name='farad')
    """Farad, `1 F = 1 C/V`; capacitance."""
    ohm: Final = (V / A).define('Ω', name='ohm', symbol_aliases='Ω')
    """Ohm, `1 ohm = 1 V/A`; resistance or impedance."""
    S: Final = (A / V).define('S', name='siemens')
    """Siemens, `1 S = 1 A/V`; electrical conductance."""
    Wb: Final = (V * s).define('Wb', name='weber', symbol_aliases='㏝')
    """Weber, `1 Wb = 1 V*s`; magnetic flux."""
    T: Final = (Wb / m**2).define('T', name='tesla')
    """Tesla, `1 T = 1 Wb/m**2`; magnetic flux density."""
    H: Final = (Wb / A).define('H', name='henry')
    """Henry, `1 H = 1 Wb/A`; inductance."""
    lm: Final = (cd * sr).define('lm', name='lumen', symbol_aliases='㏐')
    """Lumen, `1 lm = 1 cd*sr`; luminous flux."""
    lx: Final = (lm / m**2).define('lx', name='lux', symbol_aliases='㏓')
    """Lux, `1 lx = 1 lm/m**2`; illuminance."""
    Bq: Final = (s**-1).define('Bq', name='becquerel', symbol_aliases='㏃')
    """Becquerel, `1 Bq = 1/s`; radionuclide activity."""
    Gy: Final = (J / kg).define('Gy', name='gray', symbol_aliases='㏉')
    """Gray, `1 Gy = 1 J/kg`; absorbed dose."""
    Sv: Final = (J / kg).define('Sv', name='sievert', symbol_aliases='㏜')
    """Sievert, `1 Sv = 1 J/kg`; equivalent or effective dose."""
    kat: Final = (mol / s).define('kat', name='katal')
    """Katal, `1 kat = 1 mol/s`; catalytic activity."""


class _SI(ConstNamespace):
    """BIPM defining constants and shared CODATA 2022 catalogue inputs.

    https://www.bipm.org/en/publications/si-brochure
    https://physics.nist.gov/cuu/pdf/all.pdf
    """

    DeltanuCs: Final = Quantity(9_192_631_770, _si.Hz)
    """hyperfine transition frequency ∆ν of Cs-133, definition of second."""
    c: Final = Quantity(299_792_458, _si.m / _si.s)
    """speed of light in vacuum"""
    h: Final = quantity(6.626_070_15e-34, _si.J * _si.s)
    """Planck constant"""
    e: Final = quantity(1.602_176_634e-19, _si.C)
    """elementary charge"""
    kB: Final = quantity(1.380_649e-23, _si.J / _si.K)
    """Boltzmann constant"""
    NA: Final = quantity(6.022_140_76e23, _si.mol**-1)
    """Avogadro constant"""
    Kcd: Final = Quantity(683, _si.lm / _si.W)
    """luminous efficacy of 540 THz radiation, definition of candela"""

    g: Final = quantity(9.80665, _si.m / _si.s**2)  # defined here for si.kgf
    """standard acceleration of gravity"""
    T0: Final = quantity(273.15, _si.K)  # defined here for si.degC
    """standard temperature"""

    mu0: Final = quantity(1.256_637_061_27e-6, _si.H / _si.m)
    """Vacuum permeability, CODATA 2022 central value in H/m; measured."""
