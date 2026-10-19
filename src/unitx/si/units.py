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
    """The `si` namespace offers a comprehensive set of common units
    such as meters (m), kilograms (kg), and more.

    These units can be directly used in formulas or code.
    >>> 0.511 * si.MeV
    Quantity(0.511, Unit('MeV'))
    >>> str(2 / si.m)
    '2/m'
    """

    # SI base unit

    ys, zs, as_, fs, ps, ns, us, ms = _prefixed_units('s', 'yzafpnum')
    """second, time"""
    am, fm, pm, nm, um, mm, cm, dm, km, Mm, Gm = _prefixed_units('m', 'afpnumcdkMG')
    """metre/meter, length"""
    fg, pg, ng, ug, mg, g, Mg, Gg = _prefixed_units('g', 'fpnum MG')
    """gram, mass"""
    fA, pA, nA, uA, mA, kA, MA = _prefixed_units('A', 'fpnumkM')
    """ampere, electric current"""
    nK, uK, mK = _prefixed_units('K', 'num')
    """kelvin, thermodynamic temperature"""
    fmol, pmol, nmol, umol, mmol, kmol = _prefixed_units('mol', 'fpnumk')
    """mole, amount of substance"""
    ucd, mcd, kcd = _prefixed_units('cd', 'umk')
    """candela, luminous intensity"""

    # SI derived unit

    nrad, urad, mrad = _prefixed_units('rad', 'num')
    """radian, plane angle"""
    nHz, uHz, mHz, kHz, MHz, GHz, THz, PHz, EHz = _prefixed_units('Hz', 'numkMGTPE')
    """hertz, frequency"""
    fN, pN, nN, uN, mN, kN, MN, GN = _prefixed_units('N', 'fpnumkMG')
    """newton, force/weight"""
    fPa, nPa, uPa, mPa, hPa, kPa, MPa, GPa, TPa = _prefixed_units('Pa', 'fnumhkMGT')
    """pascal, pressure/stress"""
    fJ, pJ, nJ, uJ, mJ, kJ, MJ, GJ, TJ, PJ = _prefixed_units('J', 'fpnumkMGTP')
    """joule, energy/work/heat"""
    fW, pW, nW, uW, mW, kW, MW, GW, TW, PW = _prefixed_units('W', 'fpnumkMGTP')
    """watt, power/radiant flux"""
    aC, fC, pC, nC, uC, mC, kC = _prefixed_units('C', 'afpnumk')
    """coulomb, electric charge"""
    pV, nV, uV, mV, kV, MV, GV = _prefixed_units('V', 'pnumkMG')
    """volt, electric potential/voltage/emf"""
    aF, fF, pF, nF, uF, mF, kF = _prefixed_units('F', 'afpnumk')
    """farad, capacitance"""
    nohm, uohm, mohm, kohm, Mohm, Gohm, Tohm = _prefixed_units('Ω', 'numkMGT')
    """ohm, resistance/impedance/reactance"""
    pS, nS, uS, mS, kS = _prefixed_units('S', 'pnumk')
    """siemens, electrical conductance"""
    nWb, uWb, mWb, kWb = _prefixed_units('Wb', 'numk')
    """weber, magnetic flux"""
    fT, pT, nT, uT, mT, kT = _prefixed_units('T', 'fpnumk')
    """tesla, magnetic flux density"""
    pH, nH, uH, mH, kH = _prefixed_units('H', 'pnumk')
    """henry, inductance"""
    ulm, mlm, klm, Mlm = _prefixed_units('lm', 'umkM')
    """lumen, luminous flux"""
    mlx, klx = _prefixed_units('lx', 'mk')
    """lux, illuminance"""
    mBq, kBq, MBq, GBq, TBq, PBq = _prefixed_units('Bq', 'mkMGTP')
    """becquerel, activity referred to a radionuclide"""
    nGy, uGy, mGy, kGy, MGy = _prefixed_units('Gy', 'numkM')
    """gray, absorbed dose"""
    pSv, nSv, uSv, mSv, kSv = _prefixed_units('Sv', 'pnumk')
    """sievert, equivalent dose"""
    nkat, ukat, mkat = _prefixed_units('kat', 'num')
    """katal, catalytic activity"""

    # other common units

    # dimensionless
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
    mas, uas = _prefixed_units('arcsec', 'mu')
    """arcsecond, plane angle"""
    percent: Final = (_si.one / 100).define(
        '%', name='percent', symbol_aliases='٪', prefixable=False
    )
    """percent, 1/100, dimensionless"""
    permille: Final = (_si.one / 1000).define('‰', name='permille', prefixable=False)
    """permille, 1/1000, dimensionless"""
    permyriad: Final = (_si.one / 10000).define(
        '‱', 'permyriad', 'per ten thousand', prefixable=False
    )
    """permyriad, 1/10000, dimensionless"""
    # time/frequency related
    min: Final = (60 * _si.s).define('min', name='minute', prefixable=False)
    """minute, 1 min = 60 s, time"""
    h: Final = (60 * min).define('h', name='hour', prefixable=False)
    """hour, 1 h = 60 min, time"""
    d: Final = (24 * h).define('d', name='day', prefixable=False)
    """day, 1 d = 24 h, time"""
    yr: Final = (365 * d).define('yr', name='year', prefixable=False)
    """simple year, 1 yr = 365 d, time"""
    a_j: Final = (365.25 * d).define('a_j', 'Julian year', 'annum')
    """Julian year, 1 a_j = 365.25 d, time"""
    mo: Final = (a_j / 12).define('mo', name='month', prefixable=False)
    """month, 1 mo = 1/12 a_j, time"""
    rpm: Final = (1 / min).define('rpm', name='revolutions per minute', prefixable=False)
    """revoltions per minute, 60 rpm = 1 Hz, rotational speed"""
    # length/area/volume related
    au: Final = (149_597_870_700 * _si.m).define(
        'au', name='astronomical unit', symbol_aliases='AU'
    )
    """astronomical unit, length"""
    pc: Final = (au / arcsec).define('pc', name='parsec')
    """parsec, length"""
    kpc, Mpc, Gpc, Tpc = _prefixed_units('pc', 'kMGT')
    """parsec, length"""
    ly: Final = (_SI.c * a_j).define('ly', name='light year', prefixable=False)
    """light year, length"""
    nmile: Final = (1852 * _si.m).define('nmi', 'nautical mile', symbol_aliases='NM')
    """nautical mile, length in marine and aerial navigation"""
    angstrom: Final = (_si.m / TEN**10).define(
        'Å', 'ångström', 'angstrom', symbol_aliases=('Å', 'Å')
    )  # symbol: chr(0xC5), chr(0x212B), 'A'+chr(0x30A)
    """ångström, 1 Å = 10⁻¹⁰ m,
    lengths of chemical bonds, sizes of atoms and molecules, and wavelengths.
    """
    a: Final = (100 * _si.m**2).define('a', name='are', prefixable=False)
    """are, 1 a = 100 m², land area"""
    ha: Final = (100 * a).define('ha', name='hectare', prefixable=False)
    """hectare, 1 ha = 100 a = 10000 m², land area"""
    b: Final = (_si.m**2 / TEN**28).define('b', name='barn')
    """barn, 1 b = 10⁻²⁸ m², effective cross sections."""
    L: Final = (dm**3).define('L', 'litre', 'liter', symbol_aliases='l')
    """litre/liter, 1 L = 1 dm³, volume"""
    fL, pL, nL, uL, mL, cL, dL, kL, ML = _prefixed_units('L', 'fpnumcdkM')
    """litre/liter, 1 L = 1 dm³, volume"""
    # mass related
    t: Final = (1000 * _si.kg).define('t', name='tonne', prefixable=False)
    """tonne, 1 t = 1000 kg, mass"""
    Da: Final = (Decimal('1.660_539_068_92e-27') * _si.kg).define(
        'Da', 'dalton', 'unified atomic mass unit', symbol_aliases='u', prefixable='Da'
    )
    """dalton, unified atomic mass unit, 1 Da = m(¹²C) / 12, mass"""
    kDa, MDa = _prefixed_units('Da', 'kM')
    """dalton, unified atomic mass unit, 1 Da = m(¹²C) / 12, mass"""
    u: Final = Unit('u')
    """unified atomic mass unit, 1 u = m(¹²C) / 12, mass"""
    # kinetic related
    c: Final = _SI.c.define('c', name='speed of light', prefixable=False)
    """speed of light in vacuum, 299_792_458 m/s"""
    Gal: Final = (cm / _si.s**2).define('Gal', 'gal', 'galileo')
    """gal, acceleration"""
    knot: Final = (nmile / h).define('kn', 'knot', symbol_aliases='kt')
    """knot, speed in marine and aerial navigation"""
    # dynamic related
    gf: Final = (_SI.g * g).define('gf', 'gram-force', 'gram force')
    """gram-force, 1 gf = 1 g * 9.80665 m/s², force"""
    kgf: Final = Unit('kgf')
    """kilogram-force, 1 kgf = 1 kg * 9.80665 m/s², force"""
    bar: Final = (TEN**5 * _si.Pa).define('bar', name='bar')
    """bar, standard-state pressure"""
    ubar, mbar = _prefixed_units('bar', 'um')
    """bar, standard-state pressure"""
    atm: Final = (101_325 * _si.Pa).define('atm', name='atmosphere')
    """standard atmosphere"""
    mHg: Final = (1000 * (atm / 760)).define('mHg', name='metre of mercury')
    """metre of mercury, 0.760 mHg = 1 atm, pressure"""
    mmHg: Final = Unit('mmHg')
    """millimetre of mercury, 760 mmHg = 1 atm, pressure"""
    Torr: Final = mmHg.define('Torr', 'torr', 'torricelli')
    """torr, 760 Torr = 1 atm, pressure"""
    mTorr: Final = Unit('mTorr')
    """millitorr, pressure"""
    Wh: Final = (_si.W * h).define('Wh', name='watthour')
    """watthour, energy"""
    uWh, mWh, kWh, MWh, GWh, TWh, PWh = _prefixed_units('Wh', 'umkMGTP')
    """watthour, energy"""
    eV: Final = (_SI.e * _si.V).define('eV', name='electronvolt')
    """electron volt, energy"""
    ueV, meV, keV, MeV, GeV, TeV, PeV, EeV = _prefixed_units('eV', 'umkMGTPE')
    """electron volt, energy"""
    cal: Final = (4184 * mJ).define('cal', name='calorie')
    """calorie, energy"""
    kcal: Final = Unit('kcal')
    """kilocalorie, energy"""
    VA: Final = (_si.V * _si.A).define('VA', name='volt-ampere')
    """volt-ampere, apparent power"""
    kVA, MVA = _prefixed_units('VA', 'kM')
    """volt-ampere, apparent power"""
    var: Final = (_si.V * _si.A).define('var', name='volt-ampere reactive', symbol_aliases='VAr')
    """volt-ampere reactive, reactive power"""
    kvar, Mvar = _prefixed_units('var', 'kM')
    """volt-ampere reactive, reactive power"""
    # electromagnetic related
    Ah: Final = (_si.A * h).define('Ah', name='ampere-hour')
    """ampere-hour, electric charge"""
    mAh, kAh, MAh = _prefixed_units('Ah', 'mkM')
    """ampere-hour, electric charge"""
    mho: Final = _si.S.define('mho', name='mho', symbol_aliases='℧')
    """mho, electrical conductance"""
    # others
    nt: Final = (_si.cd / _si.m**2).define('nt', name='nit')
    """nit, luminance"""
