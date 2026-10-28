"""A scalar Quantity extension for series and parallel complex impedances.

Run `python examples/impedance.py` from the checkout. See `impedance.md` for the
fixed-frequency model. Fraction inputs retain exact rational arithmetic;
float and complex calculations follow ordinary Python numerical rules.
"""

from __future__ import annotations

import cmath
import math
from fractions import Fraction
from typing import cast, overload

from unitx import Quantity, Unit
from unitx.exceptions import DimensionError
from unitx.si import si
from unitx.typing import Scalar

type Number = int | float | Fraction | complex


class Impedance(Quantity[Number]):
    """A finite scalar impedance with `+` series and `//` parallel composition.

    Values being combined must describe the same angular frequency. This class
    stores the impedance value and unit, without frequency or circuit topology.
    Open circuits (infinity), Decimal and arrays are outside this small model.
    """

    __slots__ = ()

    def __init__(self, value: Number, unit: str | Unit = si.ohm) -> None:
        if type(value) not in (int, float, Fraction, complex):
            error_message = 'Impedance requires an int, float, Fraction or complex scalar.'
            raise TypeError(error_message)
        if isinstance(value, (float, complex)) and not cmath.isfinite(value):
            error_message = 'Impedance must be finite.'
            raise ValueError(error_message)
        super().__init__(value, unit)
        if self.dimension != si.ohm.dimension:
            error_message = 'Impedance requires an impedance unit.'
            raise DimensionError(error_message)

    # This small extension intentionally specializes the broader Quantity
    # signatures to scalar impedances and str/Unit conversion targets.
    def value_in(self, unit: str | Unit, /) -> Number:  # pyright: ignore[reportIncompatibleMethodOverride]
        # Scaling the explicitly supported numeric domain stays in that domain.
        return cast(Number, super().value_in(unit))

    @overload
    def __add__(self, other: Impedance, /) -> Impedance: ...

    @overload
    def __add__(self, other: Quantity[Scalar] | Scalar, /) -> Quantity[Scalar]: ...

    def __add__(self, other: object, /) -> Quantity[Scalar]:  # pyright: ignore[reportIncompatibleMethodOverride]
        if not isinstance(other, Impedance):
            return NotImplemented
        return Impedance(self.value + other.value_in(self.unit), self.unit)

    @overload
    def __iadd__(self, other: Impedance, /) -> Impedance: ...

    @overload
    def __iadd__(self, other: Quantity[Scalar] | Scalar, /) -> Quantity[Scalar]: ...

    def __iadd__(self, other: Quantity[Scalar] | Scalar, /) -> Quantity[Scalar]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self + other

    def __floordiv__(self, other: object, /) -> Impedance:
        if not isinstance(other, Impedance):
            return NotImplemented
        left, right = self.value, other.value_in(self.unit)
        if left == 0 or right == 0:
            return Impedance(0, self.unit)
        if left + right == 0:
            error_message = 'Parallel admittances cancel; the equivalent impedance is infinite.'
            raise ZeroDivisionError(error_message)
        # Z1*Z2/(Z1+Z2), avoiding a large product. Complex values use modulus.
        smaller, larger = (left, right) if abs(left) <= abs(right) else (right, left)
        return Impedance(smaller / (1 + smaller / larger), self.unit)

    def to(self, unit: str | Unit, /) -> Impedance:  # pyright: ignore[reportIncompatibleMethodOverride]
        return Impedance(self.value_in(unit), unit)


def main() -> None:
    """Evaluate an RLC branch in parallel with a resistor, using RMS phasors."""
    first = Impedance(Fraction(100))
    second = Impedance(Fraction(1, 5), si.kohm)
    parallel_resistors = first // second
    assert parallel_resistors == Fraction(200, 3) * si.ohm
    print(f'({first}) // ({second.to(first.unit)}) = {parallel_resistors}')

    frequency = 1 * si.kHz
    omega = 2 * math.pi * frequency
    inductance = 10 * si.mH
    capacitance = 1 * si.uF
    inductor = Impedance(1j * (omega * inductance).value_in(si.ohm))
    capacitor = Impedance(-1j * (1 / (omega * capacitance)).value_in(si.ohm))
    series = first + inductor + capacitor
    shunt = Impedance(300)
    network = series // shunt
    voltage = 12 * si.V  # Real RMS voltage phasor; the current may be complex.
    current = voltage / network
    current_value = current.value
    assert isinstance(current_value, complex)
    complex_power = voltage * Quantity(current_value.conjugate(), current.unit)

    assert network.dimension == si.ohm.dimension
    assert current.dimension == si.A.dimension
    assert complex_power.dimension == si.W.dimension
    assert current_value.real > 0 and current_value.imag > 0  # Net capacitive branch.
    print(f'At {frequency}: series RLC = {series.to(si.ohm):.3f}')
    print(f'With a {shunt} parallel branch: Z = {network.to(si.ohm):.3f}')
    print(f'At {voltage} RMS: I = {current.to(si.mA):.3f}, S = {complex_power.to(si.W):.3f}')


if __name__ == '__main__':
    main()
