# Extending Quantity: complex impedances

[impedance.py](impedance.py) is a standalone example, available from this checkout
rather than the installed unitx API. Run it without optional dependencies:

```sh
python examples/impedance.py
```

Open [the Smith-chart notebook](smith_chart.ipynb) alongside this module to sweep
an RLC network over frequency and plot its reflection coefficient. Keep both
files in the same directory; the notebook imports the class from `impedance.py`.

`Impedance` subclasses `Quantity` to represent the complex impedance of an ideal
two-terminal circuit at a given angular frequency. A real value describes an
ordinary resistor. The class adds `//` for parallel composition, while `+`
represents series composition. Both return another Impedance and keep the left
operand's unit. Conversion with `.to()` also returns an Impedance, allowing
`(z1 + z2).to(si.kohm) // z3` and `z1 // z2 // z3`.

For the convention $e^{i\omega t}$, ideal components have impedances

$$Z_R=R, \qquad Z_L=i\omega L, \qquad Z_C=\frac{1}{i\omega C}.$$

Series and parallel combinations follow

$$Z_{\mathrm{series}}=Z_1+Z_2, \qquad
Z_{\mathrm{parallel}}=\frac{Z_1Z_2}{Z_1+Z_2}.$$

All impedances being combined must correspond to the **same frequency**. This
class stores a value and unit, without recording or checking frequency. It does
not model a frequency-dependent component or preserve the circuit topology.
The script evaluates a 100 ohm, 10 mH, 1 microfarad series branch at 1 kHz,
puts a 300 ohm resistor in parallel and applies a 12 V RMS voltage phasor.
It calculates the current and complex power $S=VI^*$, retaining quantities
throughout the circuit calculation. $\operatorname{Re}S$ is active power and
$\operatorname{Im}S$ is reactive power; the latter retains its sign.

The example accepts finite scalar `int`, `float`, `Fraction` and `complex`
magnitudes, preserving their types at construction. Fraction inputs keep purely
rational calculations exact; float and complex calculations follow Python's
rounding rules. Decimal and arrays are outside this small model. Negative real
and imaginary parts are accepted; the class imposes no passivity assumption.

Parallel composition handles zero impedance explicitly, including two shorts.
It rearranges the formula to avoid multiplying large magnitudes. Nonzero
impedances with cancelling admittances, such as `100j` and `-100j` ohm at ideal
parallel resonance, produce an infinite equivalent impedance. Infinity is
outside this finite-value example, so that operation raises ZeroDivisionError.
Use parentheses when mixing `+` and `//`: Python gives `//` higher precedence.

Inherited multiplication and division return ordinary Quantity: voltage,
current, admittance and power have their usual units. Ordinary operations such
as subtraction also retain the base class's result behavior. Only addition of
two Impedances, parallel composition and conversion preserve the extension
type. Quantity's general subclass and serialization contracts stay unchanged.
Mixing an Impedance with an ordinary Quantity in addition returns an ordinary
Quantity in either operand order. For example, `z + 50 * si.ohm` loses the
extension's `//` operation; construct `Impedance(50)` to compose circuit elements.
The type annotations distinguish this mixed result from `z1 + z2`.
The scalar `+=` operation follows the same rule, returning a new result rather
than changing the original object. `//=` follows parallel composition.
The example deliberately narrows the inherited addition and conversion
signatures to its scalar model and str/Unit targets. Local Pyright annotations
mark that specialization; it is not a general template for preserving all of
Quantity's array and coordinate overloads.
