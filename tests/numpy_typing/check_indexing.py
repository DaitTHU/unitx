"""NumPy preparation calls retain usable wrappers at the supported boundary."""

from typing import assert_type, cast

import numpy as np
from numpy.typing import NDArray

from unitx import AffineQuantity, LogQuantity, Quantity, Unit
from unitx._numerics import Array

type ArrayQuantity = Quantity[Array] | AffineQuantity[Array] | LogQuantity[Array]


def check_indexing(values: NDArray[np.float64], sample: ArrayQuantity) -> None:
    # NumPy's stubs describe ndarray results, so assert the wrapper domain at
    # this boundary, as for other dispatched NumPy functions. Physical units
    # and wrapper families are checked at runtime, not by NumPy's stubs.
    results = (
        np.hstack((sample, sample)),
        np.vstack((sample, sample)),
        np.dstack((sample, sample)),
        np.column_stack((sample, sample)),
        np.append(sample, sample),
        np.insert(sample, 0, sample),
        np.take(sample, [0, 1]),
        np.compress([True, False, True], sample),
        np.extract([True, False, True], sample),
        np.delete(sample, 0),
        np.diag(sample),
        np.diagflat(sample),
        np.diagonal(np.atleast_2d(sample)),
        np.tril(sample),
        np.triu(sample),
    )
    for result in results:
        wrapped = cast(ArrayQuantity, result)
        assert_type(wrapped.value, Array)
    # take_along_axis's ndarray-only input annotation rejects wrappers despite
    # NumPy dispatch. Check that entry through the supported protocol instead.
    sample.__array_function__(
        np.take_along_axis, (type(sample),), (sample, np.array([0, 1]), 0), {}
    )
    for result in (
        np.split(sample, 3),
        np.array_split(sample, 2),
        np.hsplit(sample, 3),
        np.vsplit(np.atleast_2d(sample), 1),
        np.dsplit(np.atleast_3d(sample), 1),
    ):
        for wrapped in cast(list[ArrayQuantity], result):
            assert_type(wrapped.value, Array)
    target = Quantity(np.empty(2), 'cm')
    q = Quantity(values, 'm')
    q.__array_function__(np.take, (Quantity,), (q, [0, 1]), {'out': target})
    q.__array_function__(np.compress, (Quantity,), ([True, False, True], q), {'out': target})
    assert_type(target.value.dtype, np.dtype[np.float64])
    assert_type(target.unit, Unit)


def check_families(values: NDArray[np.float64]) -> None:
    check_indexing(values, Quantity(values, 'm'))
    check_indexing(values, AffineQuantity(values, 'degC'))
    check_indexing(values, LogQuantity(values, 'dBm'))
