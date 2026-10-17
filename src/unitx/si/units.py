from collections.abc import Iterator
from decimal import Decimal
from typing import Final

from .._constclass import ConstClass
from ..factor import ONE, PI, TEN
from ..unit import Unit

__all__ = ['si']


def _prefixed_units(unit: str, prefixes: str | list[str]) -> Iterator[Unit]:
    """Returns a generator of `Unit` objects with the given prefixes applied to the unit.

    Use `prefixes` as a string when every prefix is an individual character.
    Unit without prefix can be included in the string as a space character.
    Otherwise use a list of strings for multi-character prefixes.
    """
    return (Unit(prefix + unit) for prefix in prefixes)


class si(ConstClass):
    """The `si` constclass offers a comprehensive set of common units
    such as meters (m), kilograms (kg), and more.

    These units can be directly used in formulas or code.
    >>> 0.511 * si.MeV
    Quantity(0.511, Unit('MeV'))
    >>> str(2 / si.m)
    '2/m'
    """

    # SI base unit

    one: Final = Unit()
    """unitless"""
    fs, ps, ns, us, ms, s = _prefixed_units('s', 'fpnum ')
    """second, time"""
    fm, pm, nm, um, mm, cm, dm, m, km = _prefixed_units('m', 'fpnumcd k')
    """metre/meter, length"""
    mg, g, kg = _prefixed_units('g', 'm k')
    """gram, mass"""
    mA, A = _prefixed_units('A', 'm ')
    """ampere, electric current"""
    mK, K = _prefixed_units('K', 'm ')
    """kelvin, thermodynamic temperature"""
    mmol, mol = _prefixed_units('mol', 'm ')
    """mole, amount of substance"""
    cd = Unit('cd')
    """candela, luminous intensity"""

    # SI derived unit

    rad: Final = one.define('rad', name='radian')
    """radian, plane angle"""
    sr: Final = one.define('sr', name='steradian')
    """steradian, solid angle"""
    Hz: Final = (ONE / s).define('Hz', name='hertz')
    """hertz, frequency"""
    kHz, MHz, GHz, THz = _prefixed_units('Hz', 'kMGT')
    """hertz, frequency"""
    N: Final = (kg * m / s**2).define('N', name='newton')
    """newton, force/weight"""
    Pa: Final = (N / m**2).define('Pa', name='pascal')
    """pascal, pressure/stress"""
    kPa, MPa, GPa = _prefixed_units('Pa', 'kMG')
    """pascal, pressure/stress"""
    J: Final = (N * m).define('J', name='joule')
    """joule, energy/work/heat"""
    mJ, kJ, MJ = _prefixed_units('J', 'mkM')
    """joule, energy/work/heat"""
    W: Final = (J / s).define('W', name='watt')
    """watt, power/radiant flux"""
    mW, kW, MW = _prefixed_units('W', 'mkM')
    """watt, power/radiant flux"""
    C: Final = (A * s).define('C', name='coulomb')
    """coulomb, electric charge"""
    pC, nC, uC, mC = _prefixed_units('C', 'pnum')
    """coulomb, electric charge"""
    V: Final = (W / A).define('V', name='volt')
    """volt, electric potential/voltage/emf"""
    mV, kV = _prefixed_units('V', 'mk')
    """volt, electric potential/voltage/emf"""
    F: Final = (C / V).define('F', name='farad')
    """farad, capacitance"""
    pF, nF, uF, mF = _prefixed_units('F', 'pnum')
    """farad, capacitance"""
    ohm: Final = (V / A).define('Ω', name='ohm')
    """ohm, resistance/impedance/reactance"""
    mohm, kohm = _prefixed_units('Ω', 'mk')
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
    kBq, MBq, GBq = _prefixed_units('Bq', 'kMG')
    """becquerel, activity referred to a radionuclide"""
    Gy: Final = (J / kg).define('Gy', name='gray')
    """gray, absorbed dose"""
    Sv: Final = (J / kg).define('Sv', name='sievert')
    """sievert, equivalent dose"""
    nSv, uSv, mSv = _prefixed_units('Sv', 'num')
    """sievert, equivalent dose"""
    kat: Final = (mol / s).define('kat', name='katal')
    """katal, catalytic activity"""

    # non-SI units accepted for use with SI

    min: Final = (60 * s).define('min', name='minute', prefixable=False)
    """minute, 1 min = 60 s, time"""
    h: Final = (60 * min).define('h', name='hour', prefixable=False)
    """hour, 1 h = 60 min, time"""
    d: Final = (24 * h).define('d', name='day', prefixable=False)
    """day, 1 d = 24 h, time"""
    yr: Final = (365 * d).define('yr', name='year', prefixable=False)
    """simple year, 1 yr = 365 d, time"""
    # a = (365.25 * d).define('a', name='Julian year')
    # """Julian year, 1 yr = 365.25 d, time"""
    deg: Final = (PI / 180 * rad).define('°', name='degree', symbol_aliases='deg', prefixable=False)
    """degree, plane angle"""
    arcmin: Final = (deg / 60).define(
        '′', name='arcminute', symbol_aliases=("'", 'arcmin'), prefixable=False
    )
    """arcminute, plane angle"""
    arcsec: Final = (arcmin / 60).define(
        '″', name='arcsecond', symbol_aliases=('"', 'arcsec'), prefixable=False
    )
    """arcsecond, plane angle"""
    # a: Final = (100 * m**2).define('a', name='are')
    # """are, 1 a = 100 m², land area"""
    b: Final = (m**2 / TEN**28).define('b', name='barn')
    """barn, 1 b = 10⁻²⁸ m², effective cross sections."""
    L: Final = (dm**3).define('L', 'liter', 'litre', symbol_aliases='l')
    """liter/litre, 1 L = 1 dm³, volume"""
    mL: Final = Unit('mL')
    """`millilitre, 1 mL = 1 cm³, volume"""
    t: Final = (kg * 1000).define('t', name='tonne', prefixable=False)
    """tonne, 1 t = 1000 kg, mass"""
    # symbol aliases: chr(0xC5), chr(0x212B), 'A'+chr(0x30A)
    angstrom: Final = (m / TEN**10).define('Å', 'ångström', 'angstrom', symbol_aliases=('Å', 'Å'))
    """ångström, 1 Å = 10⁻¹⁰ m,
    lengths of chemical bonds, sizes of atoms and molecules, and wavelengths.
    """
    Gal: Final = (cm / s**2).define('Gal', name='gal')
    """gal, acceleration"""
    bar: Final = (TEN**5 * Pa).define('bar', name='bar')
    """bar, standard-state pressure"""
    Da: Final = (kg * Decimal('1.660_539_068_92e-27')).define('Da', name='dalton')
    """dalton, unified atomic mass unit, 1 Da = m(¹²C) / 12, mass"""
    # u = Da.define('u', name='unified atomic mass unit', symbol_aliases='amu')
    # """unified atomic mass unit, 1 u = m(¹²C) / 12, mass"""
    au: Final = (149_597_870_700 * m).define('au', name='astronomical unit')
    """astronomical unit, length"""
    pc: Final = (au / arcsec).define('pc', name='parsec')
    """parsec, length"""
    # ly: Final = (c * yr).define('ly', name='light year', prefixable=False)
    # """light year, length"""
    nmile: Final = (1852 * m).define('nmi', 'nautical mile', symbol_aliases='NM')
    """nautical mile, length in marine and aerial navigation"""
    knot: Final = (nmile / h).define('kn', 'knot', symbol_aliases='kt')
    """knot, speed in marine and aerial navigation"""
    atm: Final = (101_325 * Pa).define('atm', name='atmosphere')
    """standard atmosphere"""
    mmHg: Final = (atm / 760).define('mmHg', name='millimetre of mercury')
    """millimetre of mercury, 760 mmHg = 1 atm, pressure"""
    Torr: Final = mmHg.define('Torr', 'torr', 'torricelli', prefixable=False)
    """torr, 760 Torr = 1 atm, pressure"""
    Wh: Final = (W * h).define('Wh', name='watthour')
    """watthour, energy"""
    kWh, MWh, GWh = _prefixed_units('Wh', 'kMG')
    """watthour, energy"""
    eV: Final = (Decimal('1.602_176_634e-19') * J).define('eV', name='electronvolt')
    """electron volt, energy"""
    meV, keV, MeV, GeV, TeV = _prefixed_units('eV', 'mkMGT')
    """electron volt, energy"""
    cal: Final = (Decimal('4.184') * J).define('cal', name='calorie')
    """calorie, energy"""
    kcal: Final = Unit('kcal')
    """kilocalorie, energy"""
    # var: Final = (V * A).define('var', name='volt-ampere reactive')
    # """volt-ampere reactive, reactive power"""
    # kvar: Final = Unit('kvar')
    # """kilovolt-ampere reactive, reactive power"""
