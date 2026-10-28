"""Public magnitude annotations; importing this module does not import NumPy.

Protocols describe input capabilities, not support for additional array backends.
Concrete NumPy result aliases are for static analysis with NumPy installed.
"""

from ._numerics import Array as Array
from ._numerics import Magnitude as Magnitude
from ._numerics import Scalar as Scalar
from ._numerics.numpy_typing import NumpyArray as NumpyArray
from ._numerics.numpy_typing import NumpyValue as NumpyValue

__all__ = ['Array', 'Magnitude', 'NumpyArray', 'NumpyValue', 'Scalar']
