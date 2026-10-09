# unitx

[![PyPI version](https://img.shields.io/pypi/v/unitx)](https://pypi.org/project/unitx/)
[![Python versions](https://img.shields.io/pypi/pyversions/unitx)](https://pypi.org/project/unitx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**A lightweight Python library for physical units, quantities, and dimensional analysis.**

`unitx` provides a small core for composing unit expressions, converting quantities, and working with physical dimensions. It has no third-party runtime dependencies and is designed around explicit, composable operations.

> **Status: Alpha.** The public API and supported unit set may change between releases. Please validate the behavior you rely on before using `unitx` in critical scientific or production workflows.

## Features

* **Unit-aware quantities** — represent values together with their units using `Quantity`.
* **Unit conversion** — convert between compatible units, such as meters and kilometers.
* **Dimensional analysis** — inspect and combine dimensions with `Dimension`.
* **Composable unit expressions** — supports multiplication, division, parentheses, and integer or rational powers.
* **SI base units and prefixes** — includes the seven SI base-unit dimensions and a set of SI prefixes.
* **Rational symbolic representation** — conversion factors are represented internally using prime-factor decompositions and rational exponents where possible.
* **Lightweight installation** — no third-party runtime dependencies.

## Requirements

* Python 3.11 or newer

## Installation

Install the latest release from PyPI:

```bash
python -m pip install unitx
```

## Quick start

### Convert a quantity

```python
from unitx import Quantity

distance = Quantity(1500, "m")
distance_km = distance.to("km")

print(distance_km)  # 1.5 km
```

### Calculate with quantities

```python
from unitx import Quantity

distance = Quantity(1500, "m")
elapsed = Quantity(30, "s")

speed = distance / elapsed
print(speed)  # 50.0 m/s
```

Multiplication and division combine the underlying units as well as the numeric values.

### Compose and inspect units

```python
from unitx import Unit

velocity = Unit("m/s")
acceleration = Unit("m/s^2")

print(velocity)
print(acceleration)
print(velocity.dimension == acceleration.dimension)  # False
```

`Unit` supports multiplication, division, and exponentiation:

```python
from unitx import Unit

area = Unit("m") ** 2
velocity = Unit("m") / Unit("s")

assert area == Unit("m^2")
assert velocity == Unit("m/s")
```

### Work with dimensions directly

```python
from unitx import Dimension, Unit

velocity_dimension = Dimension(L=1, T=-1)

assert Unit("m/s").dimension == velocity_dimension
```

The base-dimension symbols are:

| Symbol  | Physical dimension                                        |
| ------- | --------------------------------------------------------- |
| `T`     | Time                                                      |
| `L`     | Length                                                    |
| `M`     | Mass                                                      |
| `I`     | Electric current                                          |
| `Theta` | Thermodynamic temperature                                 |
| `N`     | Amount of substance                                       |
| `J`     | Luminous intensity (dimension symbol, not the joule unit) |

For example, velocity has dimension `L T⁻¹`, and force has dimension `M L T⁻²`.

## Unit-expression syntax

`Unit` accepts unit symbols and compound expressions. Examples include:

| Expression | Meaning                             |
| ---------- | ----------------------------------- |
| `m`        | meter                               |
| `km`       | kilometer                           |
| `cm`       | centimeter                          |
| `kg`       | kilogram                            |
| `m/s`      | meter per second                    |
| `kg*m/s^2` | mass × length / time²               |
| `(m/s)^2`  | squared velocity unit               |
| `m^(1/2)`  | square root of a length unit        |
| `s^-(2/3)` | a negative fractional power of time |

The parser supports:

* Multiplication: `m*s` or `m s`
* Division: `m/s`
* Parentheses: `(m/s)^2`
* Exponents: `m^2`, `m**2`, or `m2`
* Fractional exponents: `m^(1/2)` and `s^-(2/3)`
* Leading division: `/s` (equivalent to `1/s`)

Unit symbols are case-sensitive. Use symbols rather than unit names when constructing `Unit` objects.

## Built-in unit scope

The current built-in unit set covers the seven SI base dimensions. Mass is represented using `g` with a scale factor of 10⁻³, so `kg` is available through the `k` prefix:

| Symbol | Unit    |
| ------ | ------- |
| `s`    | second  |
| `m`    | meter   |
| `g`    | gram    |
| `A`    | ampere  |
| `K`    | kelvin  |
| `mol`  | mole    |
| `cd`   | candela |

Supported SI prefixes can be combined with prefixable units, for example `km`, `cm`, `mg`, and `μm`.

Named derived units such as `N`, `J`, and `Pa`, customary units, and user-defined unit registration are **not currently part of the documented built-in unit set**. Check the supported symbols in the source before relying on a symbol not listed above.

## Errors and incompatible dimensions

Conversions and arithmetic operations should use dimensionally compatible quantities. For example, converting a length to a time unit is invalid:

```python
from unitx import Quantity
from unitx.exceptions import DimensionError

distance = Quantity(10, "m")

try:
    distance.to("s")
except DimensionError:
    print("Cannot convert length to time")
```

Invalid syntax and unknown unit symbols raise exceptions from `unitx.exceptions`, including `UnitSyntaxError` and `UnitSymbolError`.

## Numerical considerations and limitations

* Conversion factors are represented symbolically where possible, but applying a factor to a numeric value can involve floating-point arithmetic. Do not assume every conversion result is exact.
* The current model represents units through multiplicative factors. Units requiring an additive offset, such as degrees Celsius converted to kelvin, are not supported by this model.
* NumPy array integration, automatic custom-unit registration, and a comprehensive catalogue of named derived units are not currently documented as supported features.
* The project is in alpha. Review the tests and validate the specific operations your application depends on.

## Development

Clone the repository and install the development dependencies:

```bash
git clone https://github.com/DaitTHU/unitx.git
cd unitx

python -m venv .venv
```

Activate the virtual environment, then run:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install -e ".[test]"
pytest
ruff check .
```

## Contributing

Bug reports, focused feature proposals, and pull requests are welcome.

When reporting a bug, include:

* The Python version and `unitx` version.
* A minimal reproducible example.
* The expected result and the actual result.

For changes to unit parsing or arithmetic, please add tests that cover both the intended behavior and relevant edge cases.

## License

`unitx` is distributed under the [MIT License](LICENSE).
