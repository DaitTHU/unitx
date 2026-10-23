"""British Imperial, historical English, and US customary units.

Each catalogue has its own parsing scope. Shared international units live in
`imperial`; `us` contains US-specific definitions only.
Historical measures use the stated convention, not a universal definition.
Use `imperial.degF` for Fahrenheit readings and `imperial.degR` for Rankine intervals.
Rational scales and circular areas stay exact; definitions do not depend on the
active Decimal precision. See `docs/catalogues/imperial.md` for conventions.
"""

from decimal import Decimal
from fractions import Fraction
from typing import Final

from ..affineunit import AffineUnit
from ..factor import pi
from ..si import SI, si
from ..unitcatalog import UnitCatalog
from ..unitregistry import Registry

__all__ = ['imperial', 'imperial_historical', 'us']


with Registry(prefixable=False).defining() as _imperial_registry:

    class imperial(
        UnitCatalog,
        units=(si.m, si.s, si.kg, si.K, si.N, si.Pa, si.L, si.h),
        registry=_imperial_registry,
    ):
        """Modern British Imperial units and shared international definitions."""

        # Length
        inch: Final = (Decimal('0.0254') * si.m).define('in', 'inch', symbol_aliases='″')
        """international inch, 1 in = 0.0254 m, length"""
        ft: Final = (12 * inch).define('ft', 'foot', 'feet')
        """international foot, 1 ft = 12 in, length"""
        yd: Final = (3 * ft).define('yd', 'yard')
        """international yard, 1 yd = 3 ft, length"""
        mi: Final = (1760 * yd).define('mi', 'mile')
        """international mile, 1 mi = 1760 yd, length"""
        th: Final = (inch / 1000).define('th', 'thou', 'mil')
        """thou/mil, 1 th = 1/1000 in, length"""
        hh: Final = (4 * inch).define('hh', 'hand')
        """hand, 1 hh = 4 in, length"""
        link: Final = (33 * (ft / 50)).define('li', 'link')
        """Gunter link, 1 li = 33/50 ft, length"""
        rod: Final = (33 * ft / 2).define('rd', 'rod', 'pole', 'perch')
        """rod/pole/perch, 1 rd = 16.5 ft, length"""
        fathom: Final = (6 * ft).define('fth', 'fathom')
        """international fathom, 1 fth = 6 ft, length"""
        chain: Final = (66 * ft).define('ch', 'chain', 'Gunter chain')
        """Gunter chain, 1 ch = 66 ft, length"""
        furlong: Final = (10 * chain).define('fur', 'furlong')
        """furlong, 1 fur = 10 ch, length"""
        nmile: Final = si.nmile.define('nmi', 'nautical mile')
        """international nautical mile, 1 nmi = 1852 m, length"""

        # Typography (modern DTP conventions)
        pica: Final = (inch / 6).define('pc', 'pica')
        """DTP pica, 1 pc = 1/6 in, typographic length"""
        point: Final = (inch / 72).define('point')  # 'pt' is pint
        """DTP point, 1 point = 1/72 in, typographic length"""
        twip: Final = (point / 20).define('twip')
        """twip, 1 twip = 1/20 point, typographic length"""

        # Land and timber
        acre: Final = (10 * chain**2).define('ac', 'acre')
        """international acre, 1 ac = 43560 ft², land area"""
        board_foot: Final = (144 * inch**3).define('bd_ft', 'board foot')
        """board foot, 1 bd_ft = 144 in³, timber volume"""

        # Capacity
        gal: Final = (Decimal('4.54609') * si.L).define('gal', 'gallon')
        """Imperial gallon, 1 gal = 4.54609 L, volume"""
        qt: Final = (gal / 4).define('qt', 'quart')
        """Imperial quart, 1 qt = 1/4 Imperial gal, volume"""
        pt: Final = (qt / 2).define('pt', 'pint')
        """Imperial pint, 1 pt = 1/8 Imperial gal, volume"""
        fl_oz: Final = (pt / 20).define('fl_oz', 'fluid ounce')
        """Imperial fluid ounce, 1 fl_oz = 1/20 Imperial pt, volume"""
        gill: Final = (pt / 4).define('gi', 'gill')
        """Imperial gill, 1 gi = 1/4 Imperial pt, volume"""
        peck: Final = (2 * gal).define('pk', 'peck')
        """Imperial peck, 1 pk = 2 Imperial gal, volume"""
        bushel: Final = (8 * gal).define('bu', 'bushel')
        """Imperial bushel, 1 bu = 8 Imperial gal, volume"""

        # Mass (avoirdupois and troy)
        lb: Final = (Decimal('0.45359237') * si.kg).define('lb', 'pound')
        """avoirdupois pound, 1 lb = 0.45359237 kg, mass"""
        oz: Final = (lb / 16).define('oz', 'ounce')
        """avoirdupois ounce, 1 oz = 1/16 lb, mass"""
        grain: Final = (lb / 7000).define('gr', 'grain')
        """grain, 1 gr = 1/7000 lb, mass"""
        dram: Final = (lb / 256).define('dr', 'avoirdupois dram')
        """avoirdupois dram, 1 dr = 1/256 lb, mass"""
        st: Final = (14 * lb).define('st', 'stone')
        """stone, 1 st = 14 lb, mass"""
        cwt: Final = (112 * lb).define('cwt', 'long hundredweight')
        """long hundredweight, 1 cwt = 112 lb, mass"""
        ton: Final = (2240 * lb).define('ton', 'long ton')
        """long ton, 1 ton = 2240 lb, mass"""
        pennyweight: Final = (24 * grain).define('dwt', 'pennyweight')
        """pennyweight, 1 dwt = 24 grains, mass"""
        troy_oz: Final = (480 * grain).define('ozt', 'troy ounce')
        """troy ounce, 1 ozt = 480 grains, mass"""

        # Force, pressure, temperature intervals
        lbf: Final = (lb * SI.g).define('lbf', 'pound-force')
        """pound-force, 1 lbf = 4.4482216152605 N at standard gravity, force"""
        poundal: Final = (lb * ft / si.s**2).define('pdl', 'poundal')
        """poundal, 1 pdl = 1 lb·ft/s², force"""
        psi: Final = (lbf / inch**2).define('psi', 'pound per square inch')
        """pound-force per square inch, 1 psi = 1 lbf/in², pressure"""
        degR: Final = (5 * (si.K / 9)).define(
            '°R', 'degree Rankine', 'rankine', symbol_aliases='degR'
        )
        """degree Rankine: difference 1 °R = 5/9 K; point 0 °R = 0 K, temperature"""
        degF: Final = AffineUnit(0 @ si.degC, at=32, delta_unit=degR).define(
            '°F',
            'degree Fahrenheit',
            'fahrenheit',
            symbol_aliases=('degF', '℉'),
        )
        """degree Fahrenheit: difference 1 °F = 5/9 K; point 32 °F = 0 °C, temperature"""

        # Navigation, heat and power
        knot: Final = si.knot.define('kn', 'knot')
        """knot, 1 kn = 1 international nautical mile per hour, speed"""
        btu_IT: Final = (Decimal('1055.05585262') * si.N * si.m).define(
            'Btu_IT', 'International Table British thermal unit'
        )
        """International Table British thermal unit, 1 Btu_IT = 1055.05585262 J, energy"""
        # si.cal is the thermochemical calorie: 4.184 J exactly.
        btu_th: Final = (si.cal * (lb / si.g) * (degR / si.K)).define(
            'Btu_th', 'thermochemical British thermal unit'
        )
        """thermochemical Btu, 1 Btu_th = (453.59237 × 5/9) cal_th, energy"""
        therm_IT: Final = (100000 * btu_IT).define('therm_IT', 'IT therm')
        """IT therm, 1 therm_IT = 100000 Btu_IT, energy"""
        horsepower: Final = (550 * ft * lbf / si.s).define('hp', 'mechanical horsepower')
        """mechanical horsepower, 1 hp = 550 ft·lbf/s, power"""


with Registry(prefixable=False).defining() as _imperial_historical_registry:

    class imperial_historical(
        UnitCatalog, units=(si.m, si.s, si.kg, si.L), registry=_imperial_historical_registry
    ):
        """Historical English measures; some values depend on era or trade."""

        # Length (specified English conventions)
        barleycorn: Final = (imperial.inch / 3).define('barleycorn')
        """English barleycorn convention, 1 barleycorn = 1/3 in, length"""
        line: Final = (imperial.inch / 12).define('line')
        """English line convention, 1 line = 1/12 in, length"""
        digit: Final = (3 * imperial.inch / 4).define('digit')
        """digit convention, 1 digit = 3/4 in, length"""
        nail: Final = (9 * imperial.inch / 4).define('nail')
        """cloth nail, 1 nail = 9/4 in, length"""
        palm: Final = (3 * imperial.inch).define('palm')
        """palm convention, 1 palm = 3 in, length"""
        shaftment: Final = (6 * imperial.inch).define('shaftment')
        """shaftment convention, 1 shaftment = 6 in, length"""
        span: Final = (9 * imperial.inch).define('span')
        """span convention, 1 span = 9 in, length"""
        cubit: Final = (18 * imperial.inch).define('cubit')
        """cubit convention, 1 cubit = 18 in, length"""
        ell: Final = (45 * imperial.inch).define('ell', 'English ell')
        """English ell, 1 ell = 45 in, length"""
        pace: Final = (5 * imperial.ft / 2).define('pace')
        """pace convention, 1 pace = 2.5 international ft, length"""
        roman_pace_chart: Final = (5 * imperial.ft).define(
            'roman_pace_chart', 'Roman pace chart convention'
        )
        """Roman pace chart convention, 1 chart pace = 5 international ft, length"""
        league: Final = (3 * imperial.mi).define('lea', 'land league')
        """land league convention, 1 lea = 3 international mi, length"""
        ramsden_chain: Final = (100 * imperial.ft).define('ramsden_ch', 'Ramsden chain')
        """Ramsden chain, 1 ramsden_ch = 100 international ft, length"""
        rope: Final = (20 * imperial.ft).define('rope')
        """rope convention, 1 rope = 20 international ft, length"""
        shackle: Final = (15 * imperial.fathom).define('shackle')
        """shackle, 1 shackle = 15 international fathoms, length"""
        skein: Final = (4 * shackle).define('skein', 'skein chart convention')
        """skein chart convention, 1 skein = 120 international yd, length"""
        int_cable: Final = (imperial.nmile / 10).define('int_cable', 'international cable length')
        """international cable, 1 int_cable = 1/10 international nautical mile, length"""

        # Land, cargo and capacity
        rood: Final = (imperial.acre / 4).define('rood')
        """rood, 1 rood = 1/4 international acre, land area"""
        mass_quarter: Final = (28 * imperial.lb).define('qr_mass', 'mass quarter')
        """mass quarter, 1 qr_mass = 28 lb, mass"""
        cental: Final = (100 * imperial.lb).define('cental')
        """cental, 1 cental = 100 lb, mass"""
        capacity_quarter: Final = (8 * imperial.bushel).define('cap_qr', 'quarter of grain')
        """quarter of grain, 1 cap_qr = 8 Imperial bushels, volume"""
        pottle: Final = (2 * imperial.qt).define('pottle')
        """pottle, 1 pottle = 2 Imperial quarts, volume"""
        fl_dr: Final = (imperial.fl_oz / 8).define('fl_dr', 'fluid drachm')
        """Imperial fluid drachm, 1 fl_dr = 1/8 Imperial fluid ounce, volume"""
        minim: Final = (fl_dr / 60).define('minim')
        """Imperial minim, 1 minim = 1/60 Imperial fluid drachm, volume"""
        fluid_scruple: Final = (20 * minim).define('fl_scruple', 'fluid scruple')
        """Imperial fluid scruple, 1 fl_scruple = 20 Imperial minims, volume"""
        troy_lb: Final = (12 * imperial.troy_oz).define('lb_t', 'troy pound')
        """troy pound, 1 lb_t = 12 troy ounces = 5760 grains, mass"""
        ap_scruple: Final = (20 * imperial.grain).define('ap_scruple', 'apothecaries scruple')
        """apothecaries scruple, 1 ap_scruple = 20 grains, mass"""
        ap_dram: Final = (3 * ap_scruple).define('ap_dr', 'apothecaries dram')
        """apothecaries dram, 1 ap_dr = 60 grains, mass"""
        ap_oz: Final = (8 * ap_dram).define('ap_oz', 'apothecaries ounce')
        """apothecaries ounce, 1 ap_oz = 480 grains, mass"""
        ap_lb: Final = (12 * ap_oz).define('ap_lb', 'apothecaries pound')
        """apothecaries pound, 1 ap_lb = 5760 grains, mass"""
        register_ton: Final = (100 * imperial.ft**3).define('register_ton', 'register ton')
        """register ton, 1 register_ton = 100 ft³, cargo volume"""
        beer_firkin: Final = (9 * imperial.gal).define('beer_firkin', 'beer firkin')
        """Imperial beer firkin, 1 beer_firkin = 9 Imperial gal, volume"""
        beer_kilderkin: Final = (2 * beer_firkin).define('beer_kilderkin', 'beer kilderkin')
        """Imperial beer kilderkin, 1 beer_kilderkin = 18 Imperial gal, volume"""
        beer_barrel: Final = (2 * beer_kilderkin).define('beer_bbl', 'Imperial beer barrel')
        """Imperial beer barrel, 1 beer_bbl = 36 Imperial gal, volume"""
        beer_hogshead: Final = (54 * imperial.gal).define('beer_hhd', 'Imperial beer hogshead')
        """Imperial beer hogshead, 1 beer_hhd = 54 Imperial gal, volume"""
        beer_butt: Final = (108 * imperial.gal).define('beer_butt', 'Imperial beer butt')
        """Imperial beer butt, 1 beer_butt = 108 Imperial gal, volume"""
        beer_tun: Final = (216 * imperial.gal).define('beer_tun', 'Imperial beer tun')
        """Imperial beer tun, 1 beer_tun = 216 Imperial gal, volume"""


with Registry(prefixable=False).defining() as _us_registry:

    class us(
        UnitCatalog, units=(si.m, si.s, si.kg, si.K, si.N, si.Pa, si.L, si.h), registry=_us_registry
    ):
        """US-specific measures; shared inch, foot and pound are in `imperial`."""

        # Historical US survey measures (obsolete since 2023)
        us_survey_foot: Final = (Fraction(1200, 3937) * si.m).define(
            'us_survey_ft', 'US survey foot'
        )
        """legacy US survey foot, 1 us_survey_ft = 1200/3937 m exactly, length"""
        us_survey_mile: Final = (5280 * us_survey_foot).define('us_survey_mi', 'US survey mile')
        """legacy US survey mile, 1 us_survey_mi = 5280 US survey feet, length"""

        # The modern US cable uses the international foot, not the survey foot.
        us_cable: Final = (720 * imperial.ft).define('us_cable', 'US cable length')
        """modern US cable, 1 us_cable = 720 international ft = 219.456 m, length"""

        # Land and lumber
        circular_mil: Final = (pi / 4 * imperial.th**2).define('cmil', 'circular mil')
        """circular mil, area of a circle one mil in diameter: π/4 × (1/1000 in)²"""
        section: Final = (imperial.mi**2).define('section_nominal', 'survey section nominal')
        """nominal survey section, 1 section_nominal = 1 international mi², land area"""
        township: Final = (36 * section).define('township_nominal', 'survey township nominal')
        """nominal survey township, 1 township_nominal = 36 sections, land area"""
        roofing_square: Final = (100 * imperial.ft**2).define('roof_sq', 'roofing square')
        """roofing square, 1 roof_sq = 100 ft², area"""
        cord: Final = (128 * imperial.ft**3).define('cord')
        """cord, 1 cord = 128 ft³, stacked-wood volume"""
        acre_foot: Final = (imperial.acre * imperial.ft).define('ac_ft', 'acre-foot')
        """international acre-foot, 1 ac_ft = 43560 ft³, volume"""

        # Liquid measures
        us_gal: Final = (231 * imperial.inch**3).define('US_gal', 'US liquid gallon')
        """US liquid gallon, 1 US_gal = 231 in³, volume"""
        us_qt: Final = (us_gal / 4).define('US_qt', 'US liquid quart')
        """US liquid quart, 1 US_qt = 1/4 US liquid gal, volume"""
        us_pt: Final = (us_gal / 8).define('US_pt', 'US liquid pint')
        """US liquid pint, 1 US_pt = 1/8 US liquid gal, volume"""
        us_fl_oz: Final = (us_gal / 128).define('US_fl_oz', 'US fluid ounce')
        """US fluid ounce, 1 US_fl_oz = 1/128 US liquid gal, volume"""
        us_gill: Final = (us_pt / 4).define('US_gill', 'US gill')
        """US gill, 1 US_gill = 4 US fluid ounces, volume"""
        us_cup: Final = (us_pt / 2).define('US_cup', 'US customary cup')
        """US customary cup, 1 US_cup = 8 US fluid ounces, volume"""
        us_tbsp: Final = (us_fl_oz / 2).define('US_tbsp', 'US customary tablespoon')
        """US customary tablespoon, 1 US_tbsp = 1/2 US fluid ounce, volume"""
        us_tsp: Final = (us_tbsp / 3).define('US_tsp', 'US customary teaspoon')
        """US customary teaspoon, 1 US_tsp = 1/3 US tablespoon, volume"""
        us_fl_dr: Final = (us_fl_oz / 8).define('US_fl_dr', 'US fluid dram')
        """US fluid dram, 1 US_fl_dr = 1/8 US fluid ounce, volume"""
        us_minim: Final = (us_fl_dr / 60).define('US_minim', 'US minim')
        """US minim, 1 US_minim = 1/60 US fluid dram, volume"""

        # Dry measures and commodity barrels
        us_dry_bushel: Final = (Decimal('2150.42') * imperial.inch**3).define(
            'US_dry_bu', 'US dry bushel'
        )
        """US dry bushel, 1 US_dry_bu = 2150.42 in³, volume"""
        us_dry_peck: Final = (us_dry_bushel / 4).define('US_dry_pk', 'US dry peck')
        """US dry peck, 1 US_dry_pk = 1/4 US dry bushel, volume"""
        us_dry_gal: Final = (us_dry_bushel / 8).define('US_dry_gal', 'US dry gallon')
        """US dry gallon, 1 US_dry_gal = 1/8 US dry bushel, volume"""
        us_dry_qt: Final = (us_dry_bushel / 32).define('US_dry_qt', 'US dry quart')
        """US dry quart, 1 US_dry_qt = 1/32 US dry bushel, volume"""
        us_dry_pt: Final = (us_dry_bushel / 64).define('US_dry_pt', 'US dry pint')
        """US dry pint, 1 US_dry_pt = 1/64 US dry bushel, volume"""
        us_standard_dry_barrel: Final = (7056 * imperial.inch**3).define(
            'US_dry_bbl', 'US standard dry barrel'
        )
        """US standard dry barrel, 1 US_dry_bbl = 7056 in³, volume"""
        us_cranberry_barrel: Final = (5826 * imperial.inch**3).define(
            'US_cranberry_bbl', 'US cranberry barrel'
        )
        """US cranberry barrel, 1 US_cranberry_bbl = 5826 in³, volume"""
        us_petroleum_barrel: Final = (42 * us_gal).define('US_oil_bbl', 'US petroleum barrel')
        """US petroleum barrel, 1 US_oil_bbl = 42 US liquid gal, volume"""
        us_beer_barrel: Final = (31 * us_gal).define('US_beer_bbl', 'US beer barrel')
        """US beer barrel, 1 US_beer_bbl = 31 US liquid gal, volume"""

        # Mass, force, pressure and power
        us_cwt: Final = (100 * imperial.lb).define('US_cwt', 'US short hundredweight')
        """US short hundredweight, 1 US_cwt = 100 lb, mass"""
        us_short_ton: Final = (2000 * imperial.lb).define('US_ton', 'US short ton')
        """US short ton, 1 US_ton = 2000 lb, mass"""
        kip: Final = (1000 * imperial.lbf).define('kip')
        """kip, 1 kip = 1000 lbf, force"""
        slug: Final = (imperial.lbf * si.s**2 / imperial.ft).define('slug')
        """slug, 1 slug = 1 lbf·s²/ft, mass"""
        us_short_ton_force: Final = (2000 * imperial.lbf).define('US_tonf', 'US short ton-force')
        """US short ton-force, 1 US_tonf = 2000 lbf, force"""
        ksi: Final = (1000 * imperial.psi).define('ksi', 'kilopound per square inch')
        """kilopound-force per square inch, 1 ksi = 1000 psi, pressure"""
        # Conventional column pressures: rho * g_n * inch, with defined densities
        # 13595.1 and 1000 kg/m3. These are not temperature-specific measurements.
        inHg: Final = (Decimal('3386.388640341') * si.Pa).define(
            'inHg', 'inch of mercury', 'conventional inch of mercury'
        )
        """conventional inch of mercury, 1 inHg = 3386.388640341 Pa, pressure"""
        inH2O: Final = (Decimal('249.08891') * si.Pa).define(
            'inH₂O', 'inch of water', 'conventional inch of water'
        )
        """conventional inch of water, 1 inH₂O = 249.08891 Pa, pressure"""
        electrical_horsepower: Final = (746 * si.N * si.m / si.s).define(
            'hp_e', 'electrical horsepower'
        )
        """electrical horsepower, 1 hp_e = 746 W, power"""
        refrigeration_ton: Final = (12000 * imperial.btu_IT / si.h).define(
            'RT', 'ton of refrigeration'
        )
        """US ton of refrigeration, 1 RT = 12000 Btu_IT/h, cooling power"""
        quad_IT: Final = (10**15 * imperial.btu_IT).define('quad_IT', 'IT quad')
        """IT quad, 1 quad_IT = 10¹⁵ Btu_IT, energy"""
