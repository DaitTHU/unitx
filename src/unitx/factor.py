from __future__ import annotations

import math
import operator
import re
import warnings
from collections.abc import Callable, Mapping
from decimal import MAX_EMAX, MIN_EMIN, Context, Decimal, Overflow, localcontext
from fractions import Fraction
from typing import Any, overload

from ._numerics import Fractionable, Magnitude, UnfactoredInteger, as_fraction, as_int_if_integral
from .exceptions import _WARNING_SKIP_PREFIXES, IncompleteFactorWarning
from .monomial import Monomial

try:
    from sympy import (  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
        factorint,
        isprime,
    )
except ModuleNotFoundError as exc:
    if exc.name != 'sympy':
        raise
    from ._numerics import factorint, isprime

__all__ = ['ONE', 'PI', 'TEN', 'TWO', 'Factor', 'SymbolicConstant']

_NUMBER_RE = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')
_INVALID_CHARS = re.compile(r'[\s' + re.escape('+-−*/×÷⋅·∗⁄∕^()[]{}=') + ']')


class SymbolicConstant:
    """
    An immutable symbolic representation of a mathematical constant.

    A new symbol must denote a positive mathematical constant that cannot be
    expressed as a finite product of prime integers and previously defined
    SymbolicConstants, all raised to rational exponents. Use Factor for numbers
    already expressible in that form: 2, sqrt(2), and products such as 2 * PI
    must not be introduced as new symbolic constants.

    This mathematical construction contract is the caller's responsibility.
    The constructor validates the symbol and a positive finite float `value`;
    it does not prove nonrepresentability or keep a global constant registry.
    `value` is only a numerical approximation of the represented constant,
    not its exact definition. The contract concerns the mathematical constant,
    not the float, whose stored binary value is itself rational.

    Instances are identified by (symbol, value). Reconstructing the same pair
    is supported for equality, copying, and serialization; it does not define
    an additional independent constant. Different symbols must not be used to
    introduce aliases of the same constant. Equal float approximations alone
    do not establish equality or independence of mathematical constants.

    Symbols must be nonempty names without whitespace, arithmetic operators,
    or grouping delimiters. Numeric strings and nonprintable characters are
    rejected.

    Ordering compares float approximations, not exact mathematical values.
    Equality compares symbolic identity: a constant can compare both <= and
    >= its float approximation while remaining unequal to that rational float.
    """

    __slots__ = ('_symbol', '_value')

    def __new__(cls, symbol: str, value: float) -> SymbolicConstant:
        if not isinstance(symbol, str):
            raise TypeError(f'Symbol must be str, got {type(symbol)!r}.')
        if not isinstance(value, float):
            raise TypeError(f'Value must be float, got {type(value)!r}.')
        if not math.isfinite(value) or value <= 0:
            raise ValueError('SymbolicConstant value must be finite and positive.')
        if symbol == '':
            raise ValueError('SymbolicConstant symbol cannot be empty.')
        if symbol.isnumeric() or _NUMBER_RE.fullmatch(symbol):
            raise ValueError('SymbolicConstant symbol cannot be numeric.')
        if not symbol.isprintable():
            raise ValueError('SymbolicConstant symbol must contain only printable characters.')
        if _INVALID_CHARS.search(symbol) is not None:
            raise ValueError('Invalid SymbolicConstant symbol: whitespace or operators/delimiters.')
        obj = super().__new__(cls)
        object.__setattr__(obj, '_symbol', symbol)
        object.__setattr__(obj, '_value', value)
        return obj

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError('SymbolicConstant objects are immutable.')

    def __delattr__(self, name: str) -> None:
        raise AttributeError('SymbolicConstant objects are immutable.')

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self._symbol!r}, {self._value!r})'

    def __str__(self) -> str:
        return self._symbol

    def __float__(self) -> float:
        return self._value

    def __hash__(self) -> int:
        return hash((self._symbol, self._value))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SymbolicConstant):
            return self._symbol == other._symbol and self._value == other._value
        return NotImplemented

    def __reduce__(self) -> tuple[type[SymbolicConstant], tuple[str, float]]:
        return SymbolicConstant, (self._symbol, self._value)

    @staticmethod
    def _comparison_op(op) -> Callable[[SymbolicConstant, object], bool]:
        def comparison(self: SymbolicConstant, other: object) -> bool:
            if isinstance(other, SymbolicConstant):
                return op(self._value, other._value)
            if isinstance(other, Fractionable):
                return op(self._value, other)
            return NotImplemented

        return comparison

    __lt__ = _comparison_op(operator.lt)
    __le__ = _comparison_op(operator.le)
    __gt__ = _comparison_op(operator.gt)
    __ge__ = _comparison_op(operator.ge)

    def __mul__(self, other: Fractionable | SymbolicConstant) -> Factor:
        if isinstance(other, Fractionable):
            fraction = as_fraction(other)
            operand = Factor(fraction)
        elif isinstance(other, SymbolicConstant):
            operand = other
        else:
            return NotImplemented
        return Factor(self) * operand

    __rmul__ = __mul__

    def __truediv__(self, other: Fractionable | SymbolicConstant) -> Factor:
        if isinstance(other, Fractionable):
            fraction = as_fraction(other)
            if fraction == 0:
                raise ZeroDivisionError('division by zero')
            operand = Factor(fraction)
        elif isinstance(other, SymbolicConstant):
            operand = other
        else:
            return NotImplemented
        return Factor(self) / operand

    def __rtruediv__(self, other: Fractionable | SymbolicConstant) -> Factor:
        if isinstance(other, Fractionable):
            fraction = as_fraction(other)
            operand = Factor(fraction)
        elif isinstance(other, SymbolicConstant):
            operand = Factor(other)
        else:
            return NotImplemented
        return operand / Factor(self)

    def __pow__(self, other: Fractionable) -> Factor:
        if not isinstance(other, Fractionable):
            return NotImplemented
        return Factor._from_monomial(Monomial({self: as_fraction(other)}))

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def value(self) -> float:
        return self._value


class Factor:
    """
    An exact representation of scaling factors for unit conversions.

    This class uses `Monomial` to preserve the exact multiplicative expression.
    Instead of using floats which suffer from precision drift during repeated
    multiplications and roots, `Factor` decomposes numbers into a mapping of
    prime bases to their rational exponents.

    Algebraic Mapping:
    - Bases: Prime integers (e.g., 2, 3, 5) or SymbolicConstants (e.g., π).
    - Exponents: `Fraction` representing powers.

    Symbolic bases must satisfy SymbolicConstant's mathematical construction
    contract. Numbers expressible with prime bases or existing symbolic bases
    must be built through Factor arithmetic, not new SymbolicConstants. Neither
    construction nor arithmetic attempts to prove the contract from floats.
    Symbolic bases within one Factor must not share a symbol with different values.

    If the built-in backend cannot completely factor an input or prove a base
    prime, it emits IncompleteFactorWarning and retains unresolved bases.
    Unresolved bases are retained without further normalization. Simplification
    and classification may be incomplete, and equivalent expressions may compare
    unequal. Install SymPy for more complete support.

    Construction and arithmetic convert float inputs and exponents to Fractions
    with denominators at most 1,000,000. Inexact conversions emit
    InexactFloatWarning. Small positive floats can round to zero and fail the
    positivity check: Factor(1e-10) warns and raises ValueError. Use Decimal or
    Fraction for exact inputs and exponents.

    Factor equality compares stored bases and exponents. Numeric operands are
    compared as exact rational values, including the stored value of floats.
    Hashes match equal rational numbers and individual SymbolicConstants;
    other expressions are hashed by their stored bases and exponents.
    Decimal inputs are converted exactly to Fraction; evaluation returns int,
    Fraction, or float. Evaluation of symbolic and other irrational expressions
    is approximate, using the symbolic bases' float values. The stored algebraic
    expression remains exact. Arbitrary-precision evaluation is not supported.
    Use float() and an approximate comparison for numerical approximations.

    Examples
    --------
    >>> Factor(12)
    Factor.from_dict({2: 2, 3: 1})
    >>> Factor(2.54)  # 127 / 50
    Factor.from_dict({127: 1, 2: -1, 5: -2})
    >>> (Factor(3) / 10).value
    Fraction(3, 10)
    >>> Factor(2.54).value  # InexactFloatWarning
    Fraction(127, 50)
    """

    __slots__ = ('_monomial', '_value_cache', '_decompose_cache')
    _monomial: Monomial[int | SymbolicConstant]
    _value_cache: int | Fraction | float | None
    _decompose_cache: tuple[Fraction, Factor] | None

    def __init__(self, value: Fractionable | SymbolicConstant, /) -> None:
        self._decompose_cache = None
        if isinstance(value, SymbolicConstant):
            self._monomial = Monomial({value: 1})
            self._value_cache = float(value)
            return
        try:
            fraction = as_fraction(value)
        except TypeError:
            raise TypeError('Factor must be a number.')
        if fraction <= 0:
            raise ValueError('Factor must be positive.')
        self._monomial = Monomial(factorint(fraction.numerator))  # pyright: ignore[reportAttributeAccessIssue]
        if fraction.denominator == 1:
            self._value_cache = fraction.numerator
            return
        self._monomial /= Monomial(factorint(fraction.denominator))  # pyright: ignore[reportOperatorIssue]
        self._value_cache = fraction

    @classmethod
    def _from_monomial(cls, monomial: Monomial[int | SymbolicConstant], /) -> Factor:
        """Direct constructor from Monomial without copy. Internal use only."""
        symbols: dict[str, float] = {}
        for base in monomial.bases():
            if isinstance(base, SymbolicConstant):
                previous_value = symbols.setdefault(base.symbol, base.value)
                if previous_value != base.value:
                    error_message = (
                        f'Conflicting values for symbol {base.symbol!r}: '
                        f'{previous_value!r} != {base.value!r}.'
                    )
                    raise ValueError(error_message)
        obj = object.__new__(cls)
        obj._monomial = monomial
        obj._value_cache = None
        obj._decompose_cache = None
        return obj

    @classmethod
    def from_dict(cls, components: Mapping[int | SymbolicConstant, Fractionable], /) -> Factor:
        """Construct from prime bases and symbolic bases satisfying their contract.

        Symbolic bases with conflicting values for one symbol are rejected;
        mathematical independence is assumed, not inferred from their float
        approximations.
        """
        if not isinstance(components, Mapping):
            error_message = f'Expected a mapping for components, got {type(components).__name__}.'
            raise TypeError(error_message)
        normalized: dict[int | SymbolicConstant, Fraction] = {}
        for base, exponent in components.items():
            normalized[base] = as_fraction(exponent)
        monomial = Monomial(normalized)
        if not all(isinstance(base, (int, SymbolicConstant)) for base in monomial.bases()):
            raise TypeError('All bases must be int or SymbolicConstant.')
        for base in monomial.bases():
            if isinstance(base, UnfactoredInteger):
                if base < 2:
                    raise ValueError('Integer bases must be greater than one.')
                warnings.warn(
                    f'Integer base {int(base)} has not been fully factored. '
                    'Simplification may be incomplete. Install `sympy` for support.',
                    IncompleteFactorWarning,
                    stacklevel=2,
                    skip_file_prefixes=_WARNING_SKIP_PREFIXES,
                )
            elif isinstance(base, int) and not isprime(base):
                raise ValueError('All integer bases must be prime numbers.')
        return cls._from_monomial(monomial)

    def is_integer(self) -> bool:
        return all(
            isinstance(base, int) and exp > 0 and exp.denominator == 1
            for base, exp in self._monomial.components()
        )

    def is_rational(self) -> bool:
        return all(
            isinstance(base, int) and exp.denominator == 1
            for base, exp in self._monomial.components()
        )

    def decompose(self, *, rationalize_denominator: bool = False) -> tuple[Fraction, Factor]:
        """
        Decompose the factor into the rational part and irrational part.

        The rational part is a Fraction, and the irrational part is a Factor
        containing only SymbolicConstant bases and integer bases with *proper*
        fraction exponents. The product of the two parts equals the original factor.
        Rationalized decompositions are cached lazily on this factor; default
        decompositions are computed on each call, independently of the value cache.

        Parameters
        ----------
        rationalize_denominator : bool, optional
            If True, exponents of integer bases in the irrational part are all
            positive to rationalize the denominator, e.g. 2^(-3/2) to
            2^(1/2) / 4.

        Returns
        -------
        tuple[Fraction, Factor]
            A tuple containing the rational part and the irrational part.

        >>> (Factor(2) ** Fraction(3, 2)).decompose()
        (Fraction(2, 1), Factor.from_dict({2: Fraction(1, 2)}))
        >>> (Factor(2) ** Fraction(-3, 2)).decompose()
        (Fraction(1, 2), Factor.from_dict({2: Fraction(-1, 2)}))
        >>> (Factor(2) ** Fraction(-3, 2)).decompose(rationalize_denominator=True)
        (Fraction(1, 4), Factor.from_dict({2: Fraction(1, 2)}))
        """
        if rationalize_denominator and self._decompose_cache is not None:
            return self._decompose_cache
        rational = Fraction(1)
        irrational: dict[int | SymbolicConstant, Fraction] = {}
        for base, exponent in self._monomial.components():
            if isinstance(base, SymbolicConstant):
                irrational[base] = exponent
            elif exponent.denominator == 1:
                rational *= Fraction(base) ** exponent.numerator
            else:
                integer_part = math.floor(exponent) if rationalize_denominator else int(exponent)
                rational *= Fraction(base) ** integer_part
                irrational[base] = exponent - integer_part
        result = rational, Factor._from_monomial(Monomial._from_dict(irrational))
        if rationalize_denominator:
            self._decompose_cache = result
        return result

    @property
    def value(self) -> int | Fraction | float:
        """Evaluate exactly when rational, otherwise using float approximations."""
        if self._value_cache is not None:
            return self._value_cache
        rational, irrational = self.decompose()
        if not irrational._monomial.bases():
            self._value_cache = as_int_if_integral(rational)
            return self._value_cache
        # Keep intermediate powers outside the float range until final conversion.
        try:
            with localcontext(Context(prec=50, Emax=MAX_EMAX, Emin=MIN_EMIN)):
                value = Decimal(rational.numerator) / rational.denominator
                for base, exponent in irrational._monomial.components():
                    base_value = (
                        Decimal.from_float(base.value)
                        if isinstance(base, SymbolicConstant)
                        else Decimal(base)
                    )
                    value *= base_value ** (Decimal(exponent.numerator) / exponent.denominator)
                result = float(value)
        except Overflow:
            raise OverflowError('Factor value is too large for a float.') from None
        if not math.isfinite(result):
            raise OverflowError('Factor value is too large for a float.')
        self._value_cache = result
        return self._value_cache

    def __float__(self) -> float:
        return float(self.value)

    def format(self, *, compact: bool = True) -> str:
        """
        Format the factor as a string.

        `compact=True` indicates that the factor is formatted in a compact form,
        with the rational part collected into a numerator and denominator
        rather than a product of prime powers. Integer bases with the
        same remaining fractional exponent are multiplied into a single base.
        Both forms use Monomial's string representation, with negative powers
        displayed as denominator terms.
        >>> (Factor(5040) ** Fraction(1, 2)).format()
        '12⋅35¹ᐟ²'
        >>> (Factor(5040) ** Fraction(1, 2)).format(compact=False)
        '2²⋅3⋅5¹ᐟ²⋅7¹ᐟ²'
        """
        if not compact:
            return str(self._monomial)
        return str(self._display_monomial())

    def _display_monomial(self) -> Monomial[int | str]:
        """Collect numeric bases for compact display, without changing exact state."""
        rational, irrational = self.decompose(rationalize_denominator=True)
        display_components: dict[int | str, Fraction] = {}
        if rational.numerator > 1:
            display_components[rational.numerator] = Fraction(1)
        if rational.denominator > 1:
            display_components[rational.denominator] = Fraction(-1)
        merged_bases: dict[Fraction, int] = {}
        for base, exponent in irrational._monomial.components():
            if isinstance(base, int):
                merged_bases[exponent] = merged_bases.get(exponent, 1) * base
        for base, exponent in irrational._monomial.components():
            if isinstance(base, int):
                merged_base = merged_bases.pop(exponent, None)
                if merged_base is None:
                    continue
                base = merged_base
            # Numeric roots use string keys to stay distinct from rational coefficients.
            # Symbol validation and Factor construction prevent collisions with symbols.
            display_components[str(base)] = exponent
        return Monomial._from_dict(display_components)

    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        kv = ', '.join(
            f'{base!r}: {as_int_if_integral(exp)!r}' for base, exp in self._monomial.components()
        )
        return f'{type(self).__name__}.from_dict({{{kv}}})'

    def __hash__(self) -> int:
        if self.is_rational():
            return hash(self.value)
        components = self._monomial.components()
        if len(components) == 1:
            base, exponent = next(iter(components))
            if isinstance(base, SymbolicConstant) and exponent == 1:
                return hash(base)
        return hash(self._monomial)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Factor):
            return self._monomial == other._monomial
        if isinstance(other, SymbolicConstant):
            return self == Factor(other)
        if isinstance(other, Fractionable):
            try:
                rational = Fraction(other)
            except (ValueError, OverflowError):
                return False
            if rational <= 0 or not self.is_rational():
                return False
            return self.value == rational
        return NotImplemented

    def __mul__(self, other: Fractionable | SymbolicConstant | Factor) -> Factor:
        if isinstance(other, Factor):
            return Factor._from_monomial(self._monomial * other._monomial)
        if isinstance(other, Fractionable):
            fraction = as_fraction(other)
            return Factor._from_monomial(self._monomial * Factor(fraction)._monomial)
        if isinstance(other, SymbolicConstant):
            return Factor._from_monomial(self._monomial * Factor(other)._monomial)
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other: Fractionable | SymbolicConstant | Factor) -> Factor:
        if isinstance(other, Factor):
            return Factor._from_monomial(self._monomial / other._monomial)
        if isinstance(other, Fractionable):
            other = as_fraction(other)
            if other == 0:
                raise ZeroDivisionError('division by zero')
            return Factor._from_monomial(self._monomial / Factor(other)._monomial)
        if isinstance(other, SymbolicConstant):
            return Factor._from_monomial(self._monomial / Factor(other)._monomial)
        return NotImplemented

    def __rtruediv__(self, other: Fractionable | SymbolicConstant) -> Factor:
        if isinstance(other, Fractionable):
            fraction = as_fraction(other)
            return Factor(fraction) / self
        if isinstance(other, SymbolicConstant):
            return Factor(other) / self
        return NotImplemented

    def __pow__(self, other: Fractionable) -> Factor:
        return Factor._from_monomial(self._monomial ** as_fraction(other))

    @overload
    def __call__(self, magnitude: Fraction) -> Fraction | float: ...

    @overload
    def __call__(self, magnitude: int | float | Decimal) -> int | Fraction | float: ...

    @overload
    def __call__(self, magnitude: Magnitude) -> Magnitude: ...

    @overload
    def __call__(self, magnitude: Any) -> Any: ...

    def __call__(self, magnitude: Any) -> Any:
        """Scale a magnitude using the evaluated int, Fraction, or float value.

        Finite Decimal magnitudes are converted exactly to Fraction, without
        Decimal context rounding. An int or Fraction coefficient returns int
        or Fraction; a float coefficient returns a float approximation.
        Multiplication uses the float coefficient's exact binary ratio before
        converting the final result to float, avoiding intermediate range loss.
        Other magnitudes use native multiplication with the evaluated value.
        Zero and negative magnitudes are allowed; unsupported combinations
        raise TypeError from that multiplication.

        >>> Factor(Fraction(1, 3))(Decimal('1.5'))
        Fraction(1, 2)
        """
        if isinstance(magnitude, Decimal):
            value = self.value
            result = Fraction(value) * Fraction(magnitude)
            if isinstance(value, float):
                return float(result)
            return as_int_if_integral(result)
        return self.value * magnitude


ONE = Factor(1)  # identity
TWO = Factor(2)  # base 2, for binary scaling
TEN = Factor(10)  # base 10, for decimal scaling
PI = SymbolicConstant('π', math.pi)
"""The ratio of the circumference of a circle to its diameter."""
