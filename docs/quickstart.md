# Your first ten minutes with unitx

Start with a physical calculation, then use the same quantities in an array
and a plot. The scalar examples need only Python 3.12 or later:

```sh
python -m pip install unitx
```

## 1. Write a formula and choose its output unit

Import `si` for units. Multiplying a number by a unit creates a quantity;
arithmetic combines both the numbers and their units.

```pycon
>>> from unitx.si import si
>>> distance = 100.0 * si.m
>>> duration = 9.58 * si.s
>>> speed = distance / duration
>>> print(f'{speed:.2f}')
10.44 m/s
>>> print(f'{speed.to(si.km / si.h):.2f}')
37.58 km/h

```

`.to()` returns a quantity in the requested unit. Assign its result when you
want to keep that representation. The original quantity is unchanged.
For physical constants, import the uppercase `SI` namespace:

```pycon
>>> from unitx.si import SI
>>> rest_energy = (SI.me * SI.c**2).to(si.MeV)
>>> print(f'{rest_energy:.6f}')
0.510999 MeV

```

## 2. Correct a unit error

A distance cannot be converted to a time. The exception identifies both units
and dimensions (`L` for length, `T` for time):

```pycon
>>> distance.to(si.s)
Traceback (most recent call last):
    ...
unitx.exceptions.DimensionError: Cannot convert unit 'm' (dimension L) to unit 's' (dimension T): incompatible dimensions.
>>> travel_time = distance / speed
>>> travel_time.isclose(duration)
True

```

Check the formula when dimensions disagree. Changing a unit label alone cannot
turn a distance into a duration.

You can also construct from data with `Quantity(value, 'm/s')`. If you mistype
an expression, the traceback marks the location and may suggest a spelling:

```text
UnitSymbolError: 'milimeter' is not a valid unit symbol. Did you mean 'millimeter', 'millimetre'?
Unit expression, line 1, column 6:
  kg / milimeter
       ^
```

Suggestions use the selected registry and never change the input automatically.
Put a numerical magnitude in `Quantity(1000, 'm')`, rather than `Unit('1000 m')`.

## 3. Know when exactness matters

Use float inputs for ordinary floating-point calculations. Integer unit
alignment can produce an exact Fraction:

```pycon
>>> print(1 * si.m + 50 * si.cm)
3/2 m
>>> print(1.0 * si.m + 50 * si.cm)
1.5 m
>>> 0.1 * si.m == 10 * si.cm
False
>>> (0.1 * si.m).isclose(10 * si.cm)
True

```

Exact comparison uses the float's stored binary value. Use `.isclose()` when
you intend a comparison with tolerance; it does not change the stored data.
See [scalar numerical behavior](scalars.md) for Decimal, result types and tolerances.

## 4. Distinguish readings from differences

Use `@` for an absolute temperature reading and `*` for a temperature difference:

```pycon
>>> room = 20 @ si.degC
>>> warmer = room + 5 * si.K
>>> print(warmer)
25 °C
>>> print(warmer - room)
5 K
>>> room.value_in(si.K)
Fraction(5863, 20)

```

Here the final value is exactly 293.15 K. Adding two readings is invalid;
subtract them to get a difference. Read [points and differences](affine.md)
when working with temperatures or other coordinate origins.

## 5. Calculate with an array and plot it

Install the optional plotting dependencies:

```sh
python -m pip install 'unitx[matplotlib]'
```

Save this complete example as `motion.py` and run `python motion.py`.
It models motion from an initial speed with constant acceleration.

```python
import numpy as np
from matplotlib import pyplot as plt

from unitx import Quantity
from unitx.si import si

time = Quantity(np.linspace(0.0, 4.0, 41), si.s)
initial_speed = 2.0 * si.m / si.s
acceleration = 0.5 * si.m / si.s**2
position = initial_speed * time + acceleration * time**2 / 2
print(f'Final distance: {position[-1]:.1f}')
print(f'Mean sampled position: {position.mean():.2f}')

fig, ax = plt.subplots(layout='constrained')
ax.plot(time.value_in(si.s), position.value_in(si.m))
ax.set(xlabel=f'Time / {si.s}', ylabel=f'Position / {si.m}')
ax.grid(True, alpha=0.25)
plt.show()
```

The final distance is 12.0 m. One Quantity holds the whole array and one unit;
NumPy handles shapes and broadcasting. The constructor and `.mean()` keep the
Quantity interface visible to type checkers. Supported `np.mean(position)` also
works at runtime, though NumPy's annotations may lose the wrapper.

`.value_in(unit)` is the explicit numerical boundary used by the plot.
`.value` reads numbers in the stored unit. `np.asarray(position)` raises an
error because it would silently discard the unit. See the
[Matplotlib adapter](matplotlib.md) for automatic axis conversion.

## Continue with a complete workflow

The [three scientific workflows](workflows.md) cover a formula using constants,
a weighted fit with error bars, and sampled integration with saving and loading.
Each has a runnable script and runtime/type checks. Use the
[units and quantities reference](units.md) when you need the complete API, or
the [catalogue](catalogues/README.md) to find units and constants.
