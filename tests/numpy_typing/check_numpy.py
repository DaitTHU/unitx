"""Installed-consumer array types; no promise of arbitrary shape/dtype inference."""

from fractions import Fraction
from typing import Any, assert_type
from unittest import SkipTest

try:
    import numpy as np
    import numpy.typing as npt
except ModuleNotFoundError:
    raise SkipTest('NumPy is optional')

from unitx._numerics import Array, Magnitude, Scalar
from unitx.factor import Factor
from unitx.quantity import Quantity
from unitx.unit import Unit


def contracts(a: npt.NDArray[np.float64], scalar: np.float64, plain: int = 1) -> None:
    magnitude: Magnitude = a
    scalar_magnitude: Scalar = plain
    assert_type(magnitude, npt.NDArray[np.float64])
    assert_type(scalar_magnitude, int)
    unit = Unit('m')
    q = Quantity(a, unit)
    assert_type(q, Quantity[npt.NDArray[np.float64]])
    assert_type(q.value, npt.NDArray[np.float64])
    assert_type(unit * a, Quantity[npt.NDArray[np.float64]])
    # NumPy's own left-operand stubs cannot model unit-aware operator returns;
    # explicit constructors and Unit on the left provide precise static types.
    assert_type(Quantity(scalar, unit), Quantity[np.float64])
    assert_type(q * unit, Quantity[npt.NDArray[np.float64]])
    assert_type(q / unit, Quantity[npt.NDArray[np.float64]])
    assert_type(q * Factor(2), Quantity[npt.NDArray[np.float64]])
    assert_type(q.to('cm'), Quantity[Array])
    assert_type(q.value_in('cm'), Array)
    converted: npt.NDArray[Any] = np.asarray(q.value_in('cm'))
    assert_type(converted.ndim, int)
    assert_type(q + q, Quantity[Array])
    assert_type(Quantity(1, 'm') + q, Quantity[Array])
    assert_type(q**2, Quantity[Array])
    assert_type(q == q, Array)
    assert_type(Quantity(1, 'm') == q, Array)
    assert_type(q[0], Quantity[Array])
    assert_type(q.T, Quantity[Array])
    assert_type(q.shape, tuple[int, ...])
    assert_type(q.isclose(q), Array)
    assert_type(Quantity(1, 'm') == Quantity(2, 'm'), bool)
    assert_type(Quantity(1, 'm').to('cm'), Quantity[int | float | Fraction])


def generic_contracts(q: Quantity[Magnitude], array_q: Quantity[Array]) -> None:
    assert_type(q.to('cm'), Quantity[Magnitude])
    assert_type(q.value_in('cm'), Magnitude)
    assert_type(q * 2, Quantity[Magnitude])
    assert_type(q + 2, Quantity[Magnitude])
    assert_type(q**2, Quantity[Magnitude])
    assert_type(q == 2, bool | Array)
    assert_type(Unit('s') / array_q, Quantity[Array])
    assert_type(Quantity(2, 'm') ** np.array(2), Quantity[Array])
    assert_type(Quantity(2, 'm').isclose(Quantity(2, 'm'), rel_tol=np.array([0.0])), Array)
