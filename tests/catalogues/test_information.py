"""Information scales, independent spelling, and exact binary multiples."""

import copy
import json
import math
import os
import pickle
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from unitx import DIMENSIONLESS, Quantity, Registry, Unit
from unitx.exceptions import UnitRegistryError, UnitSymbolError
from unitx.factor import ln2, ln10
from unitx.information import information
from unitx.serialization import from_dict, to_dict
from unitx.si import si
from unitx.singleunit import SingleUnit


def test_information_scales() -> None:
    assert information.nat.dimension == information.bit.dimension == DIMENSIONLESS
    assert information.nat.factor == 1
    assert information.bit.factor == ln2
    assert information.Hart.factor == ln10
    assert information.B is information.byte
    assert information.Sh is information.bit
    assert 1 * information.B == 8 * information.bit
    assert 1 * information.nat == 1 * si.one
    assert float((1 * information.Hart).value_in(information.bit)) == pytest.approx(math.log2(10))
    # Information units have no logarithmic ratio degree and are not bel/neper.
    assert information.bit.dimension != si.B.dimension


@pytest.mark.parametrize(
    ('prefix', 'name', 'exponent'),
    [
        ('Ki', 'kibi', 10),
        ('Mi', 'mebi', 20),
        ('Gi', 'gibi', 30),
        ('Ti', 'tebi', 40),
        ('Pi', 'pebi', 50),
        ('Ei', 'exbi', 60),
        ('Zi', 'zebi', 70),
        ('Yi', 'yobi', 80),
        ('Ri', 'robi', 90),
        ('Qi', 'quebi', 100),
    ],
)
def test_information_binary_prefixes(prefix: str, name: str, exponent: int) -> None:
    for symbol, full_name, base in (('B', 'byte', information.B), ('bit', 'bit', information.bit)):
        unit = information.unit(prefix + symbol)
        assert unit == information.unit(name + full_name) == getattr(information, prefix + symbol)
        assert (1 * unit).value_in(base) == 2**exponent
        assert unit.factor / base.factor == 2**exponent
        assert information.unit(unit.expression()) == unit
        assert (Fraction(1, 2**exponent) * unit).value_in(base) == 1


def test_information_decimal_prefixes() -> None:
    assert (1 * information.MB).value_in(information.B) == 10**6
    assert (1 * information.MiB).value_in(information.MB) == Fraction(16384, 15625)
    assert information.unit('megabyte') == information.MB
    assert information.unit('kibishannon') == information.unit('KiSh')
    assert information.unit('millinat').factor == information.nat.factor / 1000
    assert information.unit('kilohartley').factor == 1000 * information.Hart.factor


def test_information_registry_isolation() -> None:
    assert Unit('B') == si.B
    assert Unit('b') == si.b
    assert Unit('MB') != information.MB
    assert information.unit('B') == information.B
    assert information.unit('bit/s') == information.bit / si.s
    registry = Registry((si.m,))
    with pytest.raises(UnitRegistryError):
        registry.include(si.B, information.B)
    assert registry.parse('m') == si.m
    with pytest.raises(UnitSymbolError):
        registry.parse('B')


@pytest.mark.parametrize('text', ['Kim', 'kibimetre', 'KiB', 'bit', 'byte', 'nat', 'Hart'])
def test_information_default_registry_invalid(text: str) -> None:
    with pytest.raises(UnitSymbolError):
        Unit(text)


@pytest.mark.parametrize('text', ['Kinat', 'kibinat', 'KiHart', 'kibihartley', 'KiKiB', 'kMiB'])
def test_information_prefix_invalid(text: str) -> None:
    with pytest.raises(UnitSymbolError):
        information.unit(text)


def test_information_prefix_metadata() -> None:
    registry = Registry((information.bit,))
    unit = registry.parse('Kibit')
    [(single, _)] = unit._monomial.components()
    assert single.prefix_symbol == 'Ki'
    assert single.prefix_name == 'kibi'
    assert single.name == 'kibibit'
    assert single.without_prefix().factor == ln2
    assert single.without_prefix().unprefixed_symbol == 'bit'
    assert single.prefix_factor == 1024


@pytest.mark.parametrize('attribute', ['nat', 'bit', 'Hart', 'B', 'MiB', 'QiB'])
def test_information_roundtrip(attribute: str) -> None:
    unit = getattr(information, attribute)
    assert copy.copy(unit) == copy.deepcopy(unit) == unit
    assert eval(repr(unit), {'Unit': Unit, 'SingleUnit': SingleUnit}) == unit
    restored = pickle.loads(pickle.dumps(unit))
    assert restored == unit
    assert hash(restored) == hash(unit)
    registry = Registry((information.nat, information.bit, information.Hart, information.B))
    quantity = Fraction(3, 7) * unit
    assert from_dict(json.loads(json.dumps(to_dict(quantity))), registry=registry) == quantity


def test_information_fresh_process_roundtrip(tmp_path: Path) -> None:
    unit = information.QiB
    payload = tmp_path / 'information.pickle'
    payload.write_bytes(pickle.dumps(unit))
    record = tmp_path / 'information.json'
    record.write_text(json.dumps(to_dict(unit)))
    script = """
import json, pickle, sys
from pathlib import Path
from unitx import Registry
from unitx.information import information
from unitx.serialization import from_dict
assert pickle.loads(Path(sys.argv[1]).read_bytes()) == information.QiB
registry = Registry((information.B,))
assert from_dict(json.loads(Path(sys.argv[2]).read_text()), registry=registry) == information.QiB
assert 'numpy' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, '-c', script, str(payload), str(record)],
        cwd=tmp_path,
        env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[2] / 'src')},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_information_compact() -> None:
    result = (4096 * information.B).compact()
    assert result.unit == information.KiB
    assert result.value == 4
    assert (10**6 * information.B).compact().unit == information.MB
    assert (1024 * si.m).compact().unit == si.km
    assert isinstance(result, Quantity)
