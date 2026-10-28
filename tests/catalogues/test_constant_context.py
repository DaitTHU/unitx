"""Fresh imports must not depend on or alter the caller's Decimal context."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    'precision,rounding,traps', [(6, 'ROUND_UP', True), (50, 'ROUND_DOWN', False)]
)
def test_si_constants_context(precision: int, rounding: str, traps: bool) -> None:
    source = """
import decimal
import json
{setup}
context = decimal.getcontext()
before = repr(context)
from unitx import Quantity
from unitx.si import SI, Particle, Planck, au, nu
assert repr(context) == before
assert decimal.getcontext() is context
print(json.dumps({{
    cls.__name__: {{name: str(q.value) for name, q in vars(cls).items() if isinstance(q, Quantity)}}
    for cls in (SI, Particle, Planck, au, nu)
}}, sort_keys=True))
"""
    setup = f"""
c = decimal.getcontext()
c.prec = {precision}
c.rounding = decimal.{rounding}
c.Emin, c.Emax = -9, 9
c.traps[decimal.Inexact] = {traps}
c.traps[decimal.Rounded] = {traps}
c.flags[decimal.Underflow] = True
decimal.DefaultContext.prec = 3
decimal.DefaultContext.traps[decimal.Inexact] = True
"""
    outputs = []
    for settings in ('', setup):
        result = subprocess.run(
            [sys.executable, '-c', source.format(setup=settings)],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[2] / 'src')},
        )
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1]
