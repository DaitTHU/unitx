"""Shared coherent SI units and the seven defining constants."""

from typing import Final

from .._constnamespace import ConstNamespace
from ..factor import ONE
from ..quantity import Quantity, quantity
from ..unit import Unit


class _si(ConstNamespace):
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

    rad: Final = one.define('rad', name='radian')
    """radian, plane angle"""
    sr: Final = one.define('sr', name='steradian')
    """steradian, solid angle"""
    Hz: Final = (ONE / s).define('Hz', name='hertz')
    """hertz, frequency"""
    N: Final = (kg * m / s**2).define('N', name='newton')
    """newton, force/weight"""
    Pa: Final = (N / m**2).define('Pa', name='pascal')
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
    ohm: Final = (V / A).define('Ω', name='ohm')
    """ohm, resistance/impedance/reactance"""
    S: Final = (A / V).define('S', name='siemens')
    """siemens, electrical conductance"""
    Wb: Final = (V * s).define('Wb', name='weber')
    """weber, magnetic flux"""
    T: Final = (Wb / m**2).define('T', name='tesla')
    """tesla, magnetic flux density"""
    H: Final = (Wb / A).define('H', name='henry')
    """henry, inductance"""
    lm: Final = (cd * sr).define('lm', name='lumen')
    """lumen, luminous flux"""
    lx: Final = (lm / m**2).define('lx', name='lux')
    """lux, illuminance"""
    Bq: Final = (ONE / s).define('Bq', name='becquerel')
    """becquerel, activity referred to a radionuclide"""
    Gy: Final = (J / kg).define('Gy', name='gray')
    """gray, absorbed dose"""
    Sv: Final = (J / kg).define('Sv', name='sievert')
    """sievert, equivalent dose"""
    kat: Final = (mol / s).define('kat', name='katal')
    """katal, catalytic activity"""


class _SI(ConstNamespace):
    DeltanuCs: Final = Quantity(9_192_631_770, _si.Hz)
    """hyperfine transition frequency of Cs-133, definition of second."""
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
