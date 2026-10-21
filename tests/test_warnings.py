"""Warning attribution at public call sites, independent of internal call depth."""

import warnings
from collections import Counter
from contextlib import nullcontext
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

import unitx._numerics as numerics_module
import unitx.factor as factor_module
from unitx.dimension import Dimension
from unitx.exceptions import (
    IncompleteFactorWarning,
    InexactFloatWarning,
    UnitSymbolError,
    UnitSyntaxError,
)
from unitx.factor import PI, Factor
from unitx.monomial import Monomial
from unitx.singleunit import SingleUnit
from unitx.unitparser import UnitParser

_INEXACT = (InexactFloatWarning,)
_INCOMPLETE = (IncompleteFactorWarning,)


@pytest.mark.parametrize(
    ('operation', 'expected', 'error'),
    [
        # Valid spellings and invalid prefixes never emit numeric warnings.
        ("SingleUnit('um')", (), None),
        ("SingleUnit('μm')", (), None),
        ("SingleUnit('Km')", (), UnitSymbolError),
        ("UnitParser('Km/s').parse()", (), UnitSymbolError),
        ("UnitParser('K m').parse()", (), None),
        ("UnitParser('m 2').parse()", (), UnitSyntaxError),
        ("UnitParser('m 1/s').parse()", (), None),
        ("UnitParser('m^0.1').parse()", (), None),
        # Numeric scales warn on rationalization; formal exponents use decimal text.
        ('as_fraction(2.54)', _INEXACT, None),
        ('as_fraction(1e-10)', _INEXACT, None),
        ("Monomial({'x': 2.54})", (), None),
        ("Monomial({'x': 2.54, 'y': 2.54})", (), None),
        ('monomial ** 2.54', (), None),
        ('monomial **= 2.54', (), None),
        ('Dimension(L=2.54)', (), None),
        ('Dimension(2.54)', (), None),
        ('Dimension(2.54, 2.54, 2.54, 2.54, 2.54, 2.54, 2.54)', (), None),
        ('dimension ** 2.54', (), None),
        ('dimension **= 2.54', (), None),
        ('Factor(2.54)', _INEXACT, None),
        ('Factor.from_dict({2: 2.54})', (), None),
        ('Factor.from_dict({2: 2.54, 3: 2.54})', (), None),
        ('factor * 2.54', _INEXACT, None),
        ('2.54 * factor', _INEXACT, None),
        ('factor / 2.54', _INEXACT, None),
        ('2.54 / factor', _INEXACT, None),
        ('factor ** 2.54', (), None),
        ('factor *= 2.54', _INEXACT, None),
        ('factor /= 2.54', _INEXACT, None),
        ('factor **= 2.54', (), None),
        ('PI * 2.54', _INEXACT, None),
        ('2.54 * PI', _INEXACT, None),
        ('PI / 2.54', _INEXACT, None),
        ('2.54 / PI', _INEXACT, None),
        ('PI ** 2.54', (), None),
        ('PI *= 2.54', _INEXACT, None),
        ('PI /= 2.54', _INEXACT, None),
        ('PI **= 2.54', (), None),
        # Conversion can warn before positivity or division checks raise.
        ('Factor(-2.54)', _INEXACT, ValueError),
        ('Factor(1e-10)', _INEXACT, ValueError),
        ('factor * 1e-10', _INEXACT, ValueError),
        ('factor / 1e-10', _INEXACT, ZeroDivisionError),
        ('1e-10 / factor', _INEXACT, ValueError),
        ('PI * 1e-10', _INEXACT, ValueError),
        ('PI / 1e-10', _INEXACT, ZeroDivisionError),
        ('1e-10 / PI', _INEXACT, ValueError),
        # Built-in factoring, including recursive powers and both rational parts.
        ('factorint(143, limit=5)', _INCOMPLETE, None),
        ('factorint(143**9, limit=5)', _INCOMPLETE, None),
        ('isprime(2**89 - 1)', _INCOMPLETE, None),
        ('Factor(n)', _INCOMPLETE, None),
        ('Factor(Fraction(1, n))', _INCOMPLETE, None),
        ('Factor(n**2)', _INCOMPLETE, None),
        ('Factor(Fraction(1, n**3))', _INCOMPLETE, None),
        ('Factor(Fraction(n, d))', _INCOMPLETE * 2, None),
        ('Factor.from_dict({2**89 - 1: 1})', _INCOMPLETE, None),
        ('Factor.from_dict({marked_n: 1})', _INCOMPLETE, None),
        ('Factor.from_dict({marked_n: 1, marked_d: -1})', _INCOMPLETE * 2, None),
        ('factor * n', _INCOMPLETE, None),
        ('n * factor', _INCOMPLETE, None),
        ('factor / n', _INCOMPLETE, None),
        ('n / factor', _INCOMPLETE, None),
        ('factor *= n', _INCOMPLETE, None),
        ('factor /= n', _INCOMPLETE, None),
        ('PI * n', _INCOMPLETE, None),
        ('n * PI', _INCOMPLETE, None),
        ('PI / n', _INCOMPLETE, None),
        ('n / PI', _INCOMPLETE, None),
        ('PI *= n', _INCOMPLETE, None),
        ('PI /= n', _INCOMPLETE, None),
        # The first warning can be correctly attributed while a later one is not.
        ('Factor.from_dict({2**89 - 1: 2.54})', _INCOMPLETE, None),
        ('Factor.from_dict({marked_n: 2.54})', _INCOMPLETE, None),
        (
            'Factor.from_dict({2**89 - 1: 2.54, marked_n: 2.54})',
            _INCOMPLETE * 2,
            None,
        ),
        # Exact inputs must not manufacture either warning.
        ('as_fraction(0.5)', (), None),
        ('Factor(0.5)', (), None),
        ("Factor(Decimal('0.1'))", (), None),
        ('Factor(Fraction(1, 3))', (), None),
        ('Factor.from_dict({2: 0.5})', (), None),
        ("Monomial({'x': 0.5})", (), None),
        ('Dimension(L=0.5)', (), None),
        ('factor * 0.5', (), None),
        ('PI ** 0.5', (), None),
    ],
)
def test_unitx_warning_user_location(
    operation: str,
    expected: tuple[type[Warning], ...],
    error: type[Exception] | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SymPy must not hide the fallback warnings being tested.
    monkeypatch.setattr(factor_module, 'factorint', numerics_module.factorint)
    monkeypatch.setattr(factor_module, 'isprime', numerics_module.isprime)
    n = 1000003 * 1000033
    d = 1000037 * 1000039
    namespace = {
        '__name__': 'user_code',
        'as_fraction': numerics_module.as_fraction,
        'factorint': numerics_module.factorint,
        'isprime': numerics_module.isprime,
        'Factor': Factor,
        'PI': PI,
        'Monomial': Monomial,
        'Dimension': Dimension,
        'SingleUnit': SingleUnit,
        'UnitParser': UnitParser,
        'Fraction': Fraction,
        'Decimal': Decimal,
        'factor': Factor(2),
        'monomial': Monomial({'x': 1}),
        'dimension': Dimension(L=1),
        'n': n,
        'd': d,
        'marked_n': numerics_module.UnfactoredInteger(n, 1000000),
        'marked_d': numerics_module.UnfactoredInteger(d, 1000000),
    }
    user_file = tmp_path / 'user_code.py'
    source = f'# Simulated user entry point.\n\n{operation}\n'
    user_file.write_text(source)
    code = compile(source, str(user_file), 'exec')

    with warnings.catch_warnings(record=True) as recorded:
        # Capture repeated warnings as well as unexpected categories.
        warnings.simplefilter('always')
        with pytest.raises(error) if error is not None else nullcontext():
            exec(code, namespace)

    assert Counter(warning.category for warning in recorded) == Counter(expected)
    for warning in recorded:
        assert (warning.filename, warning.lineno) == (str(user_file), 3)
