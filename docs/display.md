# Display and formatting

`Unit`, `ScaledUnit`, and `Quantity` implement `to_latex()` and `to_html()`,
plus `_repr_latex_()` and `_repr_html_()` for Notebook frontends. LaTeX methods
return math fragments with slash unit fractions by default; `fraction=False`
uses signed unit powers instead. The Notebook hook adds `$` delimiters. HTML uses text,
`sup` exponents and explicit denominator grouping, without MathJax. Frontends
choose which available MIME representation to show. IPython/Jupyter, NumPy
and SymPy are not required to render scalar quantities.

Rendering traverses the formal unit factors: units are never silently converted
to SI or simplified by physical equivalence. Rational exponents stay rational;
ScaledUnit's extra exact multiplier remains visible, including for dimensionless
units. A Quantity groups extra scale factors with its unit. Custom symbols are
literal text, escaped independently for HTML and LaTeX; they are not executable
markup or raw TeX commands. Unicode symbols remain Unicode and require font
support in the renderer. These strings are for presentation, not serialization.
`repr()` and `Unit.expression()` keep their existing reconstruction contracts.

## Quantity format syntax

`format(q, 'numeric_spec;style')`, or `f'{q:numeric_spec;style}'`, selects:

| Style | Result |
| --- | --- |
| `text` (default) | Readable magnitude and unit |
| `latex` | LaTeX math fragment with slash unit fractions, without dollar delimiters |
| `html` | Escaped HTML fragment |
| `siunitx` | siunitx v3 `\qty`, `\num`, or `\unit` commands |

For example, `f'{Quantity(12.345, "m/s"):.2f}'` is `12.35 m/s`.
`f'{q:.3e;latex}'` typesets ordinary real-number scientific notation with
`\times 10^{...}`, followed by a structural LaTeX unit. `q.to_latex('.3e')` and
`q.to_html('.3e')` provide the same rich fragments directly.

Width, fill and alignment apply only to the magnitude, before escaping. A
semicolon used as the Python fill character (such as `;>12.2f`) is preserved;
a later semicolon can select style. Unknown styles, empty explicit styles,
and extra style separators raise ValueError. Empty overall specs and `;text`
match `str(q)` exactly. Format the resulting string separately to align the
whole Quantity, for example `f'{str(q):>20}'`.

Formatting delegates to `format(q.value, numeric_spec)`: no implicit float
coercion, unit conversion, rounding of the stored value or global state changes.
Decimal and custom scalars retain their own formatting behavior. Fraction
formatting capabilities depend on Python version; default LaTeX renders Fraction
magnitudes as fractions. Complex magnitudes are grouped. Built-in real numbers
and NumPy numeric scalars use mathematical scientific notation and NaN/Inf
symbols in LaTeX and HTML. The decimal mantissa comes directly from Python's
formatted string; only the exponent spelling is normalized. No numeric value
is recomputed. Padded/grouped strings, complex magnitudes and custom numeric
subclasses remain escaped literal text. Fraction subclasses retain their own
formatting, rather than being forced into a fraction. An unscaled empty unit
is omitted; scaled dimensionless expressions such as `km/m` remain visible.

Arrays retain NumPy's own string summarization and precision settings. HTML
preserves magnitude spaces and line breaks; LaTeX displays the summarized
literal text with line breaks replaced by spaces. Empty and zero-dimensional
arrays are supported. Nonempty numeric specs such as `.2f` are passed directly
to ndarray and raise its native TypeError; no per-element format language is
introduced. To control array precision, use NumPy's print options explicitly.
Rendering does not copy, write to, or implicitly extract an array's units.

Protocol reference: [IPython rich display](https://ipython.readthedocs.io/en/stable/config/integrating.html).
Numeric format syntax: [Python format specification](https://docs.python.org/3/library/string.html#format-specification-mini-language).


## Unit formatting and unit fractions

Unit and ScaledUnit support `format(unit, 'latex')`, `'html'` and `'siunitx'`. Empty specs and `'text'` use `str(unit)`; ordinary Python string
specs such as `'>20'` align the complete unit label. Numeric specs such as `.2f`
and unknown display names raise ValueError.

```python
from unitx.quantity import Quantity
from unitx.unit import Unit

q = Quantity(1.234e-6, 'm/s')
assert format(Unit('m/s^2'), 'latex') == r'\text{m}/\text{s}^{2}'
assert Unit('m/s^2').to_latex(fraction=False) == r'\text{m}\,\text{s}^{-2}'
assert q.to_latex('.3e') == format(q, '.3e;latex')
```

Default LaTeX unit fractions follow the numerator/denominator structure of
`str(unit)`, using a slash rather than a stacked fraction. Multiple denominator
factors are grouped, for example `kg m/(s^2 A)` becomes
`\text{kg}\,\text{m}/(\text{s}^{2}\,\text{A})`. Notebook hooks use this
same default.

Use `to_latex(fraction=False)` explicitly to render the unit expression with
signed powers. This also applies to extra scale factors in a ScaledUnit; it does
not change units, stored magnitude or the independent rendering of a Fraction
magnitude. No stacked-unit-fraction option or separate inline format is provided.
The accepted format names are `text`, `latex`, `html` and `siunitx`.
Unknown display names raise ValueError.

## siunitx export

`Unit.to_siunitx()`, `ScaledUnit.to_siunitx()` and
`Quantity.to_siunitx(format_spec='')` generate LaTeX for siunitx v3.
Quantity also accepts the `;siunitx` format option:

```python
from unitx.quantity import Quantity
from unitx.unit import Unit

assert Unit('kg m/s^2').to_siunitx() == r'\unit{kg.m.s^{-2}}'
q = Quantity(12.3456, 'm/s')
assert f'{q:.3f;siunitx}' == r'\qty{12.346}{m.s^{-1}}'
assert Quantity(2, '').to_siunitx() == r'\num{2}'
```

Load `\usepackage{siunitx}` (v3) in your document. Escaped text/fraction
fallbacks also use `amsmath`. The output has no math delimiters; siunitx commands
work in text or math mode. Standard Notebook MathJax does not provide siunitx;
Notebook hooks continue to return the ordinary LaTeX representation.

Simple ASCII alphabetic unit symbols use siunitx's native **literal unit** syntax,
with dots between factors and exact signed rational exponents. For example,
`m^(1/3)` exports as `\unit{m^{1/3}}`, without replacing 1/3 by a rounded decimal.
The exporter does not invent named unit macros, declare custom units in the
LaTeX document, expand units to SI, or map literal units to semantic macros such
as `\metre`. Literal input consequently does not offer every customization
available for siunitx's semantic unit macros.

Extra ScaledUnit multipliers and other unit symbols use the existing escaped
structural LaTeX with `parse-units=false`. The rendered unit argument is grouped
to protect its slashes from siunitx literal formatting. Quantity groups an extra multiplier
with its unit, preserving the stored magnitude. Unicode symbols require an
appropriate engine/font, as with ordinary LaTeX export.

Magnitude formatting remains delegated to Python. ASCII decimal and scientific
notation use siunitx's number parser. Other results (including padded/grouped
numbers, NaN/Inf, complex numbers, arrays and custom strings) are printed
with `parse-numbers=false`. NaN/Inf and default Fraction values use the same
mathematical representation as ordinary LaTeX; other strings are escaped. Array precision specs remain subject to ndarray's native limitations.
Width is applied before escaping; rendered TeX whitespace is not a fixed-width
layout guarantee. No magnitude is forced to float and no buffer is modified.

Reference: [siunitx official manual source](https://github.com/josephwright/siunitx/blob/main/siunitx.tex).

## Limits

Quantity width/alignment applies to the magnitude. To align the complete
quantity, format `str(q)` separately. Native array precision specifications,
complex-number mathematical typesetting, and siunitx semantic macro mapping
remain outside this formatter's scope. Custom labels are escaped literal
text, not trusted TeX. Rendering has no global defaults, automatic unit
conversion or formatter-plugin registration.

[Documentation index](README.md) · [Units and quantities](units.md)
