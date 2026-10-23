"""Shared coherent SI units and the seven defining constants."""

from typing import Final

from .._constnamespace import ConstNamespace
from ..quantity import Quantity, quantity
from ..unit import Unit
from ..unitcatalog import UnitCatalog
from ..unitregistry import default_registry


class _si(UnitCatalog, registry=default_registry):
    # SI base unit

    one: Final = Unit()
    """unitless"""
    s: Final = Unit('s')
    """second, time"""
    m: Final = Unit('m')
    """metre/meter, length"""
    kg: Final = Unit('kg')
    """kilogram, mass"""
    A: Final = Unit('A')
    """ampere, electric current"""
    K: Final = Unit('K')
    """kelvin, thermodynamic temperature"""
    mol: Final = Unit('mol')
    """mole, amount of substance"""
    cd: Final = Unit('cd')
    """candela, luminous intensity"""

    # SI derived unit

    rad: Final = one.define('rad', name='radian', symbol_aliases='㎭')
    """radian, plane angle"""
    sr: Final = one.define('sr', name='steradian', symbol_aliases='㏛')
    """steradian, solid angle"""
    Hz: Final = (s**-1).define('Hz', name='hertz', symbol_aliases='㎐')
    """hertz, frequency"""
    N: Final = (kg * m / s**2).define('N', name='newton')
    """newton, force/weight"""
    Pa: Final = (N / m**2).define('Pa', name='pascal', symbol_aliases='㎩')
    """pascal, pressure/stress"""
    J: Final = (N * m).define('J', name='joule')
    """joule, energy/work/heat"""
    W: Final = (J / s).define('W', name='watt')
    """watt, power/radiant flux"""
    C: Final = (A * s).define('C', name='coulomb')
    """coulomb, electric charge"""
    V: Final = (W / A).define('V', name='volt')
    """volt, electric potential/voltage/emf"""
    F: Final = (C / V).define('F', name='farad')
    """farad, capacitance"""
    ohm: Final = (V / A).define('Ω', name='ohm', symbol_aliases='Ω')
    """ohm, resistance/impedance/reactance"""
    S: Final = (A / V).define('S', name='siemens')
    """siemens, electrical conductance"""
    Wb: Final = (V * s).define('Wb', name='weber', symbol_aliases='㏝')
    """weber, magnetic flux"""
    T: Final = (Wb / m**2).define('T', name='tesla')
    """tesla, magnetic flux density"""
    H: Final = (Wb / A).define('H', name='henry')
    """henry, inductance"""
    lm: Final = (cd * sr).define('lm', name='lumen', symbol_aliases='㏐')
    """lumen, luminous flux"""
    lx: Final = (lm / m**2).define('lx', name='lux', symbol_aliases='㏓')
    """lux, illuminance"""
    Bq: Final = (s**-1).define('Bq', name='becquerel', symbol_aliases='㏃')
    """becquerel, activity referred to a radionuclide"""
    Gy: Final = (J / kg).define('Gy', name='gray', symbol_aliases='㏉')
    """gray, absorbed dose"""
    Sv: Final = (J / kg).define('Sv', name='sievert', symbol_aliases='㏜')
    """sievert, equivalent dose"""
    kat: Final = (mol / s).define('kat', name='katal')
    """katal, catalytic activity"""


class _SI(ConstNamespace):
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
