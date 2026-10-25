"""Installed-consumer array types; no promise of arbitrary shape/dtype inference."""

from fractions import Fraction
from typing import Any, TypeAlias, assert_type
from unittest import SkipTest

try:
    import numpy as np
    import numpy.typing as npt
except ModuleNotFoundError:
    raise SkipTest('NumPy is optional')

from unitx._numerics import Array, Magnitude, Scalar
from unitx.factor import Factor
from unitx.quantity import Quantity
from unitx.si import SI, si
from unitx.unit import Unit

NumpyValue: TypeAlias = npt.NDArray[Any] | np.number[Any]


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
    assert_type(q.to('cm'), Quantity[NumpyValue])
    assert_type(q.value_in('cm'), NumpyValue)
    converted: npt.NDArray[Any] = np.asarray(q.value_in('cm'))
    assert_type(converted.ndim, int)
    assert_type(q + q, Quantity[NumpyValue])
    assert_type(Quantity(1, 'm') + q, Quantity[NumpyValue])
    assert_type(q**2, Quantity[NumpyValue])
    assert_type(q == q, Array)
    assert_type(Quantity(1, 'm') == q, Array)
    assert_type(q[0], Quantity[NumpyValue])
    assert_type(q.T, Quantity[npt.NDArray[np.float64]])
    assert_type(Quantity(scalar, unit).T, Quantity[np.float64])
    assert_type(q.flatten(), Quantity[npt.NDArray[Any]])
    assert_type(q.dtype, np.dtype[Any])
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
    assert_type(Unit('s') / array_q, Quantity[NumpyValue])
    assert_type(Quantity(2, 'm') ** np.array(2), Quantity[NumpyValue])
    assert_type(Quantity(2, 'm').isclose(Quantity(2, 'm'), rel_tol=np.array([0.0])), Array)


def array_alias(q: Quantity[Array]) -> Quantity[Array]:
    return q


def mutation_contracts(a: npt.NDArray[np.float64], scalar: np.float64) -> None:
    q = Quantity(a, 'm')
    widened = array_alias(q)
    # Element writes never replace the object or dtype held by either alias.
    widened[0] = Quantity(100, 'cm')
    q[:] = Quantity(a, 'cm')
    assert_type(q.value, npt.NDArray[np.float64])
    q += Quantity(1, 'm')
    assert_type(q, Quantity[npt.NDArray[np.float64]])
    q -= Quantity(a, 'cm')
    assert_type(q, Quantity[npt.NDArray[np.float64]])
    q *= 2
    assert_type(q, Quantity[npt.NDArray[np.float64]])
    q /= 2
    assert_type(q, Quantity[npt.NDArray[np.float64]])
    q **= 1
    assert_type(q, Quantity[npt.NDArray[np.float64]])
    q @= a
    assert_type(q, Quantity[npt.NDArray[np.float64]])
    widened += Quantity(1, 'm')
    # A protocol-only alias could hold an immutable NumPy scalar too.
    assert_type(widened.value, NumpyValue)
    assert_type(q.value.dtype, np.dtype[np.float64])
    numpy_scalar = Quantity(scalar, 'm')
    numpy_scalar += Quantity(scalar, 'm')
    assert_type(numpy_scalar, Quantity[NumpyValue])


def output_contracts(a: npt.NDArray[np.float64]) -> None:
    q = Quantity(a, 'm')
    widened = array_alias(q)
    # Covariant aliases may be output targets: only ndarray elements change,
    # never the magnitude object or dtype described by the narrower type.
    # NumPy's own out stubs accept ndarray only; check the protocol entry here.
    q.__array_ufunc__(np.add, '__call__', q, q, out=(widened,))
    q.__array_ufunc__(np.add, 'accumulate', q, out=(widened,))
    q.__array_ufunc__(np.modf, '__call__', q, out=(widened, None))
    exponents = np.empty(a.shape, dtype=np.int32)
    q.__array_ufunc__(np.frexp, '__call__', q, out=(widened, exponents))
    # NumPy versions infer different shape parameters for this plain output array.
    assert_type(exponents.dtype, np.dtype[np.int32])
    q.__array_function__(np.cumsum, (Quantity,), (q,), {'out': widened})
    q.__array_function__(np.clip, (Quantity,), (q, Quantity(0, 'm'), None), {'out': widened})
    q.__array_ufunc__(np.minimum, '__call__', q, Quantity(1, 'm'), out=(widened,))
    q.__array_function__(np.where, (Quantity,), (a > 0, q, Quantity(0, 'm')), {})
    scalar_target = Quantity(np.empty(()), 'cm')
    q.__array_ufunc__(np.add, 'reduce', q, out=(scalar_target,))
    q.__array_function__(np.sum, (Quantity,), (q,), {'out': scalar_target})
    for operation in (np.quantile, np.percentile):
        q.__array_function__(operation, (Quantity,), (q, 0.5), {'out': scalar_target})
    mask = np.array([True, False, True])
    q.__array_ufunc__(
        np.add, 'reduce', q, out=(scalar_target,), where=mask, initial=Quantity(2, 'cm')
    )
    for operation in (np.sum, np.mean, np.std):
        q.__array_function__(operation, (Quantity,), (q,), {'where': mask, 'out': scalar_target})
    q.__array_function__(
        np.var, (Quantity,), (q,), {'where': mask, 'out': Quantity(np.empty(()), 'm2')}
    )
    q.__array_function__(np.diff, (Quantity,), (q,), {'prepend': Quantity(2, 'cm')})
    q.__array_function__(np.gradient, (Quantity,), (q, Quantity(2, 's')), {'axis': 0})
    product_target = Quantity(np.empty(()), 'm2')
    q.__array_ufunc__(
        np.multiply, 'reduce', q, out=(product_target,), where=np.array([True, False, True])
    )
    q.__array_function__(
        np.prod,
        (Quantity,),
        (q,),
        {'initial': Quantity(2, 's'), 'out': Quantity(np.empty(()), 'm2 s')},
    )
    assert_type(product_target.value.dtype, np.dtype[np.float64])
    assert_type(scalar_target.value.dtype, np.dtype[np.float64])
    assert_type(widened.value, Array)
    assert_type(q.value, npt.NDArray[np.float64])
    assert_type(q * Unit('s'), Quantity[npt.NDArray[np.float64]])


def array_expression_contracts(a: npt.NDArray[np.float64]) -> None:
    distance = np.linspace(0.1, 3, 100) * si.m
    typed_distance = si.m * a
    temperature = si.K * np.array([3000, 4000, 5000], dtype=np.float64)
    default_temperature = si.K * np.array([3000, 4000, 5000])
    ratio = distance / (2 * si.m)
    occupation = np.expm1(np.asarray(ratio.value_in(si.one)))
    explicit = distance / (distance**2 * occupation)
    direct = distance / (distance**2 * np.expm1(ratio))
    assert_type(distance.unit, Unit)
    assert_type(distance.value.ndim, int)
    assert_type(typed_distance, Quantity[npt.NDArray[np.float64]])
    assert_type(temperature, Quantity[npt.NDArray[np.float64]])
    assert_type(default_temperature, Quantity[npt.NDArray[Any]])
    assert_type(ratio, Quantity[NumpyValue])
    assert_type(explicit, Quantity[NumpyValue])
    assert_type(direct, Quantity[NumpyValue])
    assert_type(direct.value, NumpyValue)
    assert_type(direct.to('1/cm'), Quantity[NumpyValue])
    assert_type(SI.h * distance, Quantity[NumpyValue])


def numpy_result_methods(a: npt.NDArray[np.float64]) -> None:
    q = Quantity(a, 'm')
    for result in (q.to('cm'), q + q, q * 2, q / 2, q**2, q[0], next(iter(q)), abs(q)):
        # Exercise actual ndarray/scalar interfaces after unitx operations,
        # rather than accepting a result widened to Any or a small protocol.
        assert_type(result.value, NumpyValue)
        assert_type(result.value.ndim, int)
        assert_type(result.value.dtype.kind.startswith('f'), bool)
        result.value.reshape((1, -1))
        result.value.astype(np.float64)
        result.value.sum()
