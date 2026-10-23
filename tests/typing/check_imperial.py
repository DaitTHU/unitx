from typing import assert_type

from unitx import Quantity, Unit
from unitx.imperial import imperial, imperial_historical, us

assert_type(imperial.btu_th, Unit)
assert_type(us.us_survey_foot, Unit)
assert_type(us.circular_mil, Unit)
assert_type(us.inHg, Unit)
assert_type(us.inH2O, Unit)
assert_type(us.parse('inH₂O'), Unit)
assert_type(3 * us.circular_mil, Quantity[int])
assert_type(imperial_historical.ramsden_chain, Unit)
