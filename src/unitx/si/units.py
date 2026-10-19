from collections.abc import Iterator
from decimal import Decimal
from typing import Final

from ..factor import PI, TEN
from ..unit import Unit
from .bases import _SI, _si

__all__ = ['si']


def _prefixed_units(unit: str, prefixes: str | list[str]) -> Iterator[Unit]:
    """Returns a generator of `Unit` objects with the given prefixes applied to the unit.

    Use `prefixes` as a string when every prefix is an individual character.
    Unit without prefix can be included in the string as a space character.
    Otherwise use a list of strings for multi-character prefixes.
    """
    return (Unit(prefix + unit) for prefix in prefixes)


class si(_si):
    """The `si` constclass offers a comprehensive set of common units
    such as meters (m), kilograms (kg), and more.

    These units can be directly used in formulas or code.
    >>> 0.511 * si.MeV
    Quantity(0.511, Unit('MeV'))
    >>> str(2 / si.m)
    '2/m'
    """

    # SI base unit

    as_, fs, ps, ns, us, ms = _prefixed_units('s', 'afpnum')
    """second, time"""
    fm, pm, nm, um, mm, cm, dm, km = _prefixed_units('m', 'fpnumcdk')
    """metre/meter, length"""
    pg, ng, ug, mg, g = _prefixed_units('g', 'pnum ')
    """gram, mass"""
    pA, nA, uA, mA, kA = _prefixed_units('A', 'pnumk')
    """ampere, electric current"""
    nK, uK, mK = _prefixed_units('K', 'num')
    """kelvin, thermodynamic temperature"""
    pmol, nmol, umol, mmol, kmol = _prefixed_units('mol', 'pnumk')
    """mole, amount of substance"""
    mcd: Final = Unit('mcd')
    """candela, luminous intensity"""

    # SI derived unit

    urad, mrad = _prefixed_units('rad', 'um')
    """radian, plane angle"""
    kHz, MHz, GHz, THz, PHz = _prefixed_units('Hz', 'kMGTP')
    """hertz, frequency"""
    pN, nN, uN, mN, kN, MN = _prefixed_units('N', 'pnumkM')
    """newton, force/weight"""
    mPa, hPa, kPa, MPa, GPa = _prefixed_units('Pa', 'mhkMG')
    """pascal, pressure/stress"""
    nJ, uJ, mJ, kJ, MJ, GJ = _prefixed_units('J', 'numkMG')
    """joule, energy/work/heat"""
    pW, nW, uW, mW, kW, MW, GW, TW = _prefixed_units('W', 'pnumkMGT')
    """watt, power/radiant flux"""
    fC, pC, nC, uC, mC = _prefixed_units('C', 'fpnum')
    """coulomb, electric charge"""
    nV, uV, mV, kV, MV = _prefixed_units('V', 'numkM')
    """volt, electric potential/voltage/emf"""
    fF, pF, nF, uF, mF = _prefixed_units('F', 'fpnum')
    """farad, capacitance"""
    uohm, mohm, kohm, Mohm, Gohm = _prefixed_units('Ω', 'umkMG')
    """ohm, resistance/impedance/reactance"""
    uS, mS = _prefixed_units('S', 'um')
    """siemens, electrical conductance"""
    uWb, mWb = _prefixed_units('Wb', 'um')
    """weber, magnetic flux"""
    nT, uT, mT = _prefixed_units('T', 'num')
    """tesla, magnetic flux density"""
    nH, uH, mH = _prefixed_units('H', 'num')
    """henry, inductance"""
    kBq, MBq, GBq, TBq = _prefixed_units('Bq', 'kMGT')
    """becquerel, activity referred to a radionuclide"""
    uGy, mGy = _prefixed_units('Gy', 'um')
    """gray, absorbed dose"""
    nSv, uSv, mSv = _prefixed_units('Sv', 'num')
    """sievert, equivalent dose"""

    # non-SI units accepted for use with SI

    min: Final = (60 * _si.s).define('min', name='minute', prefixable=False)
    """minute, 1 min = 60 s, time"""
    h: Final = (60 * min).define('h', name='hour', prefixable=False)
    """hour, 1 h = 60 min, time"""
    d: Final = (24 * h).define('d', name='day', prefixable=False)
    """day, 1 d = 24 h, time"""
    yr: Final = (365 * d).define('yr', name='year', prefixable=False)
    """simple year, 1 yr = 365 d, time"""
    a_j: Final = (365.25 * d).define('a_j', name='Julian year')
    """Julian year, 1 a_j = 365.25 d, time"""
    mo: Final = (a_j / 12).define('mo', name='month', prefixable=False)
    """month, 1 mo = 1/12 a_j, time"""
    deg: Final = (PI / 180 * _si.rad).define(
        '°', name='degree', symbol_aliases='deg', prefixable=False
    )
    """degree, plane angle"""
    arcmin: Final = (deg / 60).define(
        '′', name='arcminute', symbol_aliases=("'", 'arcmin'), prefixable=False
    )
    """arcminute, plane angle"""
    arcsec: Final = (arcmin / 60).define(
        '″',
        name='arcsecond',
        symbol_aliases=('"', 'arcsec'),
        prefixable=('arcsec', 'arcsecond'),
    )
    """arcsecond, plane angle"""
    a: Final = (100 * _si.m**2).define('a', name='are', prefixable=False)
    """are, 1 a = 100 m², land area"""
    ha: Final = (100 * a).define('ha', name='hectare', prefixable=False)
    """hectare, 1 ha = 100 a = 10000 m², land area"""
    b: Final = (_si.m**2 / TEN**28).define('b', name='barn')
    """barn, 1 b = 10⁻²⁸ m², effective cross sections."""
    L: Final = (dm**3).define('L', 'liter', 'litre', symbol_aliases='l')
    """liter/litre, 1 L = 1 dm³, volume"""
    nL, uL, mL, cL, dL = _prefixed_units('L', 'numcd')
    """liter/litre, 1 L = 1 dm³, volume"""
    t: Final = (1000 * _si.kg).define('t', name='tonne', prefixable=False)
    """tonne, 1 t = 1000 kg, mass"""
    # symbol aliases: chr(0xC5), chr(0x212B), 'A'+chr(0x30A)
    angstrom: Final = (_si.m / TEN**10).define(
        'Å', 'ångström', 'angstrom', symbol_aliases=('Å', 'Å')
    )
    """ångström, 1 Å = 10⁻¹⁰ m,
    lengths of chemical bonds, sizes of atoms and molecules, and wavelengths.
    """
    Gal: Final = (cm / _si.s**2).define('Gal', name='gal')
    """gal, acceleration"""
    bar: Final = (TEN**5 * _si.Pa).define('bar', name='bar')
    """bar, standard-state pressure"""
    mbar: Final = Unit('mbar')
    """millibar, standard-state pressure"""
    Da: Final = (Decimal('1.660_539_068_92e-27') * _si.kg).define('Da', name='dalton')
    """dalton, unified atomic mass unit, 1 Da = m(¹²C) / 12, mass"""
    u: Final = Da.define('u', name='unified atomic mass unit', prefixable=False)
    """unified atomic mass unit, 1 u = m(¹²C) / 12, mass"""
    au: Final = (149_597_870_700 * _si.m).define('au', name='astronomical unit')
    """astronomical unit, length"""
    pc: Final = (au / arcsec).define('pc', name='parsec')
    """parsec, length"""
    ly: Final = (_SI.c * a_j).define('ly', name='light year', prefixable=False)
    """light year, length"""
    nmile: Final = (1852 * _si.m).define('nmi', 'nautical mile', symbol_aliases='NM')
    """nautical mile, length in marine and aerial navigation"""
    knot: Final = (nmile / h).define('kn', 'knot', symbol_aliases='kt')
    """knot, speed in marine and aerial navigation"""
    atm: Final = (101_325 * _si.Pa).define('atm', name='atmosphere')
    """standard atmosphere"""
    mHg: Final = (1000 * (atm / 760)).define('mHg', name='metre of mercury')
    """metre of mercury, 0.760 mHg = 1 atm, pressure"""
    mmHg: Final = Unit('mmHg')
    """millimetre of mercury, 760 mmHg = 1 atm, pressure"""
    Torr: Final = mmHg.define('Torr', 'torr', 'torricelli', prefixable=False)
    """torr, 760 Torr = 1 atm, pressure"""
    Wh: Final = (_si.W * h).define('Wh', name='watthour')
    """watthour, energy"""
    mWh, kWh, MWh, GWh, TWh = _prefixed_units('Wh', 'mkMGT')
    """watthour, energy"""
    eV: Final = (_SI.e * _si.V).define('eV', name='electronvolt')
    """electron volt, energy"""
    meV, keV, MeV, GeV, TeV, PeV = _prefixed_units('eV', 'mkMGTP')
    """electron volt, energy"""
    cal: Final = (Decimal('4.184') * _si.J).define('cal', name='calorie')
    """calorie, energy"""
    kcal: Final = Unit('kcal')
    """kilocalorie, energy"""
    var: Final = (_si.V * _si.A).define('var', name='volt-ampere reactive')
    """volt-ampere reactive, reactive power"""
    kvar: Final = Unit('kvar')
    """kilovolt-ampere reactive, reactive power"""
