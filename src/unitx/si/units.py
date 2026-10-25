import math as _math
from collections.abc import Iterator
from decimal import Decimal
from typing import Final

from .._electromagnetism.coefficients import _EMCoefficients
from .._registry.prefixes import NO_PREFIX, PREFIX_DEFINITIONS, PREFIX_SYMBOL_ALIAS_TO_SYMBOL
from ..affineunit import AffineUnit
from ..factor import pi, ten
from ..logunit import LogUnit
from ..monomial import Monomial
from ..quantity import Quantity
from ..singleunit import SingleUnit
from ..unit import Unit
from ..unitregistry import default_registry
from .bases import _SI, _si

__all__ = ['si']


def _prefixed_units(unit: Unit, prefixes: str | list[str]) -> Iterator[Unit]:
    """Replace a single unit's prefix with each listed prefix.

    Use `prefixes` as a string when every prefix is an individual character.
    Unit without prefix can be included in the string as a space character.
    Otherwise use a list of strings for multi-character prefixes.
    """
    [(base, exponent)] = unit._monomial.components()
    assert exponent == 1
    for spelling in prefixes:
        prefix = (
            PREFIX_DEFINITIONS[PREFIX_SYMBOL_ALIAS_TO_SYMBOL.get(spelling, spelling)]
            if spelling != ' '
            else NO_PREFIX
        )
        yield Unit._from_monomial(Monomial({SingleUnit._from_definitions(prefix, base._unit): 1}))


class si(_si, registry=default_registry):
    """SI units, common non-SI units and named coordinate descriptors.

    Definitions follow the BIPM SI Brochure (ninth edition) and NIST SP 811;
    entry docstrings specify additional library conventions. `em` groups the
    float-valued electromagnetic formula coefficients derived from CODATA 2022.
    `c` remains the speed-of-light Unit; `em.c` is a physical Quantity.
    https://www.bipm.org/en/publications/si-brochure
    https://www.nist.gov/pml/special-publication-811/nist-guide-si-chapter-5-units-outside-si
    https://physics.nist.gov/cuu/pdf/all.pdf

    These units can be directly used in formulas or code.
    >>> 0.511 * si.MeV
    Quantity(0.511, Unit('MeV'))
    >>> str(2 / si.m)
    '2/m'
    """

    em: Final[_EMCoefficients] = _EMCoefficients(
        k1=Quantity(
            float(_SI.mu0.value) * float(_SI.c.value) ** 2 / (4 * _math.pi),
            _si.N * _si.m**2 / _si.C**2,
        ),
        alpha=Quantity(1.0, _si.one),
        c=Quantity(float(_SI.c.value), _SI.c.unit),
    )
    """SI formula coefficients: `k1=1/(4*pi*epsilon0)`, `alpha=1`,
    `k2=mu0/(4*pi)`. Derived from CODATA 2022 mu0 and exact BIPM c;
    float Quantities, without uncertainty propagation.
    https://physics.nist.gov/cuu/pdf/all.pdf
    """

    # SI base unit

    ys, zs, as_, fs, ps, ns, us, ms = _prefixed_units(_si.s, 'yzafpnum')
    """SI-prefixed seconds; each prefix supplies its standard power-of-ten factor."""
    am, fm, pm, nm, um, mm, cm, dm, km, Mm, Gm = _prefixed_units(_si.m, 'afpnumcdkMG')
    """SI-prefixed metres; each prefix supplies its standard power-of-ten factor."""
    fg, pg, ng, ug, mg, g, Mg, Gg = _prefixed_units(_si.kg, 'fpnum MG')
    """Gram and SI-prefixed grams; `1 g = 0.001 kg`."""
    fA, pA, nA, uA, mA, kA, MA = _prefixed_units(_si.A, 'fpnumkM')
    """SI-prefixed amperes; `1 A = 1 C/s`."""
    nK, uK, mK = _prefixed_units(_si.K, 'num')
    """SI-prefixed kelvins; temperature differences scale from K."""
    fmol, pmol, nmol, umol, mmol, kmol = _prefixed_units(_si.mol, 'fpnumk')
    """SI-prefixed moles; `1 mol` contains exactly `6.02214076e23` entities."""
    ucd, mcd, kcd = _prefixed_units(_si.cd, 'umk')
    """SI-prefixed candelas; luminous intensity scales from cd."""

    # SI derived unit

    nrad, urad, mrad = _prefixed_units(_si.rad, 'num')
    """SI-prefixed radians; angle is arc length/radius."""
    nHz, uHz, mHz, kHz, MHz, GHz, THz, PHz, EHz = _prefixed_units(_si.Hz, 'numkMGTPE')
    """SI-prefixed hertz; `1 Hz = 1/s`."""
    fN, pN, nN, uN, mN, kN, MN, GN = _prefixed_units(_si.N, 'fpnumkMG')
    """SI-prefixed newtons; `1 N = 1 kg*m/s**2`."""
    fPa, nPa, uPa, mPa, hPa, kPa, MPa, GPa, TPa = _prefixed_units(_si.Pa, 'fnumhkMGT')
    """SI-prefixed pascals; `1 Pa = 1 N/m**2`."""
    fJ, pJ, nJ, uJ, mJ, kJ, MJ, GJ, TJ, PJ = _prefixed_units(_si.J, 'fpnumkMGTP')
    """SI-prefixed joules; `1 J = 1 N*m`."""
    fW, pW, nW, uW, mW, kW, MW, GW, TW, PW = _prefixed_units(_si.W, 'fpnumkMGTP')
    """SI-prefixed watts; `1 W = 1 J/s`."""
    aC, fC, pC, nC, uC, mC, kC = _prefixed_units(_si.C, 'afpnumk')
    """SI-prefixed coulombs; `1 C = 1 A*s`."""
    pV, nV, uV, mV, kV, MV, GV = _prefixed_units(_si.V, 'pnumkMG')
    """SI-prefixed volts; `1 V = 1 W/A`."""
    aF, fF, pF, nF, uF, mF, kF = _prefixed_units(_si.F, 'afpnumk')
    """SI-prefixed farads; `1 F = 1 C/V`."""
    nohm, uohm, mohm, kohm, Mohm, Gohm, Tohm = _prefixed_units(_si.ohm, 'numkMGT')
    """SI-prefixed ohms; `1 ohm = 1 V/A`."""
    pS, nS, uS, mS, kS = _prefixed_units(_si.S, 'pnumk')
    """SI-prefixed siemens; `1 S = 1 A/V`."""
    nWb, uWb, mWb, kWb = _prefixed_units(_si.Wb, 'numk')
    """SI-prefixed webers; `1 Wb = 1 V*s`."""
    fT, pT, nT, uT, mT, kT = _prefixed_units(_si.T, 'fpnumk')
    """SI-prefixed teslas; `1 T = 1 Wb/m**2`."""
    pH, nH, uH, mH, kH = _prefixed_units(_si.H, 'pnumk')
    """SI-prefixed henries; `1 H = 1 Wb/A`."""
    ulm, mlm, klm, Mlm = _prefixed_units(_si.lm, 'umkM')
    """SI-prefixed lumens; `1 lm = 1 cd*sr`."""
    mlx, klx = _prefixed_units(_si.lx, 'mk')
    """SI-prefixed lux; `1 lx = 1 lm/m**2`."""
    mBq, kBq, MBq, GBq, TBq, PBq = _prefixed_units(_si.Bq, 'mkMGTP')
    """SI-prefixed becquerels; `1 Bq = 1/s` for radionuclide activity."""
    nGy, uGy, mGy, kGy, MGy = _prefixed_units(_si.Gy, 'numkM')
    """SI-prefixed grays; `1 Gy = 1 J/kg` of absorbed dose."""
    pSv, nSv, uSv, mSv, kSv = _prefixed_units(_si.Sv, 'pnumk')
    """SI-prefixed sieverts; `1 Sv = 1 J/kg` of equivalent or effective dose."""
    nkat, ukat, mkat = _prefixed_units(_si.kat, 'num')
    """SI-prefixed katals; `1 kat = 1 mol/s`."""

    # other common units

    # dimensionless
    deg: Final = (pi / 180 * _si.rad).define(
        '°', name='degree', symbol_aliases='deg', prefixable=False
    )
    """Degree, `1 degree = pi/180 rad`; plane angle."""
    arcmin: Final = (deg / 60).define(
        '′', name='arcminute', symbol_aliases=("'", 'arcmin'), prefixable=False
    )
    """Arcminute, `1 arcmin = 1/60 degree`; plane angle."""
    arcsec: Final = (arcmin / 60).define(
        '″',
        name='arcsecond',
        symbol_aliases=('"', 'arcsec'),
        prefixable=('arcsec', 'arcsecond'),
    )
    """Arcsecond or its SI prefixes; `1 arcsec = 1/3600 degree`."""
    mas, uas = _prefixed_units(arcsec, 'mu')
    """Arcsecond or its SI prefixes; `1 arcsec = 1/3600 degree`."""
    percent: Final = (_si.one / 100).define(
        '%', name='percent', symbol_aliases='٪', prefixable=False
    )
    """Percent, `1 percent = 1/100`; dimensionless ratio."""
    permille: Final = (_si.one / 1000).define('‰', name='permille', prefixable=False)
    """Permille, `1 permille = 1/1000`; dimensionless ratio."""
    permyriad: Final = (_si.one / 10000).define(
        '‱', 'permyriad', 'per ten thousand', prefixable=False
    )
    """Permyriad, `1 permyriad = 1/10000`; dimensionless ratio."""
    # time/frequency related
    min: Final = (60 * _si.s).define('min', name='minute', prefixable=False)
    """Minute, `1 min = 60 s`; time."""
    h: Final = (60 * min).define('h', name='hour', prefixable=False)
    """Hour, `1 h = 3600 s`; time."""
    d: Final = (24 * h).define('d', name='day', prefixable=False)
    """Day, `1 d = 86400 s`; time."""
    yr: Final = (365 * d).define('yr', name='year', prefixable=False)
    """Simple year, `1 yr = 365 d`; fixed library interval, not a calendar year."""
    a_j: Final = (365.25 * d).define('a_j', 'Julian year', 'annum')
    """Julian year, `1 a_j = 365.25 d = 31557600 s`."""
    mo: Final = (a_j / 12).define('mo', name='month', prefixable=False)
    """Mean month, `1 mo = 1/12 Julian year`; fixed library interval."""
    rpm: Final = (1 / min).define('rpm', name='revolutions per minute', prefixable=False)
    """Revolutions per minute, `1 rpm = 1/60 Hz`; rotational frequency, not rad/s."""
    # length/area/volume related
    au: Final = (149_597_870_700 * _si.m).define(
        'au', name='astronomical unit', symbol_aliases=('AU', '㍳')
    )
    """Astronomical unit, `1 au = 149597870700 m` exactly."""
    pc: Final = (au / arcsec).define('pc', name='parsec', symbol_aliases='㍶')
    """Parsec or its SI prefixes; `1 pc = (648000/pi) au`."""
    kpc, Mpc, Gpc, Tpc = _prefixed_units(pc, 'kMGT')
    """Parsec or its SI prefixes; `1 pc = (648000/pi) au`."""
    ly: Final = (_SI.c * a_j).define('ly', name='light year', prefixable=False)
    """Light-year, distance travelled by vacuum light in one Julian year."""
    nmile: Final = (1852 * _si.m).define('nmi', 'nautical mile', symbol_aliases='NM')
    """International nautical mile, `1 nmi = 1852 m` exactly."""
    angstrom: Final = (_si.m / ten**10).define(
        'Å', 'ångström', 'angstrom', symbol_aliases=('Å', 'Å')
    )  # symbol: chr(0xC5), chr(0x212B), 'A'+chr(0x30A)
    """Angstrom, `1 angstrom = 1e-10 m` exactly; microscopic length."""
    a: Final = (100 * _si.m**2).define('a', name='are', prefixable=False)
    """Are, `1 a = 100 m**2`; land area."""
    ha: Final = (100 * a).define('ha', name='hectare', symbol_aliases='㏊', prefixable=False)
    """Hectare, `1 ha = 10000 m**2`; land area."""
    b: Final = (_si.m**2 / ten**28).define('b', name='barn')
    """Barn, `1 b = 1e-28 m**2`; interaction cross section."""
    L: Final = (dm**3).define('L', 'litre', 'liter', symbol_aliases=('l', 'ℓ'))
    """Litre or its SI prefixes; `1 L = 1 dm**3 = 0.001 m**3`."""
    fL, pL, nL, uL, mL, cL, dL, kL, ML = _prefixed_units(L, 'fpnumcdkM')
    """Litre or its SI prefixes; `1 L = 1 dm**3 = 0.001 m**3`."""
    # mass related
    t: Final = (1000 * _si.kg).define('t', name='tonne', prefixable=False)
    """Tonne, `1 t = 1000 kg`; mass."""
    Da: Final = (Decimal('1.660_539_068_92e-27') * _si.kg).define(
        'Da', 'dalton', 'unified atomic mass unit', symbol_aliases='u', prefixable='Da'
    )
    """Dalton, `m(carbon-12)/12`; nominal CODATA 2022 scale `1.66053906892e-27 kg`."""
    kDa, MDa = _prefixed_units(Da, 'kM')
    """Dalton, `m(carbon-12)/12`; nominal CODATA 2022 scale `1.66053906892e-27 kg`."""
    u: Final = Da
    """Unified atomic mass unit; the same definition object as `Da`."""
    # kinetic related
    c: Final = _SI.c.define('c', name='speed of light', prefixable=False)
    """Speed unit, `1 c = 299792458 m/s`; `SI.c` and `si.em.c` are Quantities."""
    Gal: Final = (cm / _si.s**2).define('Gal', 'gal', 'galileo')
    """Galileo, `1 Gal = 1 cm/s**2`; acceleration."""
    knot: Final = (nmile / h).define('kn', 'knot', symbol_aliases=('kt', '㏏'))
    """Knot, `1 kn = 1 nautical mile/hour = 1852/3600 m/s`."""
    # dynamic related
    gf: Final = (_SI.g * g).define('gf', 'gram-force', 'gram force')
    """Gram-force, `1 gf = 0.00980665 N`, using standard gravity."""
    kgf: Final = next(_prefixed_units(gf, 'k'))
    """Kilogram-force, `1 kgf = 9.80665 N`, using standard gravity."""
    bar: Final = (ten**5 * _si.Pa).define('bar', name='bar', symbol_aliases='㍴')
    """Bar or its SI prefixes; `1 bar = 100000 Pa` exactly."""
    ubar, mbar = _prefixed_units(bar, 'um')
    """Bar or its SI prefixes; `1 bar = 100000 Pa` exactly."""
    atm: Final = (101_325 * _si.Pa).define('atm', name='atmosphere', prefixable=False)
    """Standard atmosphere, `1 atm = 101325 Pa` exactly."""
    mHg: Final = (1000 * (atm / 760)).define('mHg', name='metre of mercury')
    """Metre of mercury; library convention `1 mHg = (1000/760) atm`."""
    mmHg: Final = next(_prefixed_units(mHg, 'm'))
    """Millimetre of mercury; library convention `1 mmHg = 1/760 atm`."""
    Torr: Final = mmHg.define('Torr', 'torr', 'torricelli')
    """Torr, `1 Torr = 1/760 atm`; independent named definition."""
    mTorr: Final = next(_prefixed_units(Torr, 'm'))
    """Millitorr, `1 mTorr = 0.001 Torr`; pressure."""
    Wh: Final = (_si.W * h).define('Wh', name='watthour')
    """Watt-hour or its SI prefixes; `1 Wh = 3600 J`."""
    uWh, mWh, kWh, MWh, GWh, TWh, PWh = _prefixed_units(Wh, 'umkMGTP')
    """Watt-hour or its SI prefixes; `1 Wh = 3600 J`."""
    eV: Final = (_SI.e * _si.V).define('eV', name='electronvolt')
    """Electronvolt or its SI prefixes; `1 eV = 1.602176634e-19 J` exactly."""
    ueV, meV, keV, MeV, GeV, TeV, PeV, EeV = _prefixed_units(eV, 'umkMGTPE')
    """Electronvolt or its SI prefixes; `1 eV = 1.602176634e-19 J` exactly."""
    cal: Final = (4184 * mJ).define('cal', name='calorie', symbol_aliases='㎈')
    """Thermochemical calorie, `1 cal = 4.184 J` exactly."""
    kcal: Final = next(_prefixed_units(cal, 'k'))
    """Thermochemical kilocalorie, `1 kcal = 4184 J` exactly."""
    VA: Final = (_si.V * _si.A).define('VA', name='volt-ampere')
    """Volt-ampere or its SI prefixes; `1 VA = 1 V*A`; apparent power."""
    kVA, MVA = _prefixed_units(VA, 'kM')
    """Volt-ampere or its SI prefixes; `1 VA = 1 V*A`; apparent power."""
    var: Final = (_si.V * _si.A).define('var', name='volt-ampere reactive', symbol_aliases='VAr')
    """Var or its SI prefixes; `1 var = 1 V*A`; reactive power."""
    kvar, Mvar = _prefixed_units(var, 'kM')
    """Var or its SI prefixes; `1 var = 1 V*A`; reactive power."""
    # electromagnetic related
    Ah: Final = (_si.A * h).define('Ah', name='ampere-hour')
    """Ampere-hour or its SI prefixes; `1 Ah = 3600 C`."""
    mAh, kAh, MAh = _prefixed_units(Ah, 'mkM')
    """Ampere-hour or its SI prefixes; `1 Ah = 3600 C`."""
    mho: Final = _si.S.define('mho', name='mho', symbol_aliases='℧')
    """Mho, `1 mho = 1 siemens`; independent named definition."""
    # temperature
    degC: Final = AffineUnit(_SI.T0).define(
        '°C', 'degree Celsius', 'celsius', symbol_aliases=('degC', '℃')
    )
    """Celsius coordinate: difference `1 degC = 1 K`; reading zero is `273.15 K`."""
    # others
    nt: Final = (_si.cd / _si.m**2).define('nt', name='nit')
    """Nit, `1 nt = 1 cd/m**2`; luminance."""
    # Logarithmic units
    Np: Final = Unit('Np')
    """Neper, log difference: amplitude ratio `exp(x)` for value `x`."""
    B: Final = Unit('B')
    """Bel, log difference: power ratio `10**x` for value `x`."""
    dB: Final = Unit('dB')
    """Decibel, `0.1 B`: power ratio `10**(x/10)`, amplitude ratio `10**(x/20)`."""
    dBm: Final = LogUnit(1 * mW, delta_unit=dB, kind='power').define('dBm', 'decibel-milliwatt')
    """Log power coordinate: reading `x` dBm represents `10**(x/10) mW`."""
    dBW: Final = LogUnit(1 * _si.W, delta_unit=dB, kind='power').define('dBW', 'decibel-watt')
    """Log power coordinate: reading `x` dBW represents `10**(x/10) W`."""
    dBV: Final = LogUnit(1 * _si.V, delta_unit=dB, kind='amplitude').define('dBV', 'decibel-volt')
    """Log amplitude coordinate: reading `x` dBV represents `10**(x/20) V`."""
