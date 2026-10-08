from __future__ import annotations

from fractions import Fraction
from typing import Generic, Literal, Self, TypeVar
from collections.abc import Iterator, ItemsView, KeysView, ValuesView, Callable

from .utils.number import common_fraction, ZERO
from .utils.special_char import superscript


def _sort_components(components: ItemsView[K, Fraction], /, *, reverse=False) -> list[tuple[K, Fraction]]:
    try:
        return sorted(components, key=lambda x: x[0], reverse=reverse)  # type: ignore
    except TypeError:
        return sorted(components, key=lambda x: (type(x[0]).__module__, type(x[0]).__qualname__, repr(x[0])), reverse=reverse)


def _make_exponent_formatter(op: Literal['^', '**']) -> Callable[[Fraction], str]:
    def format_exponent(e: Fraction) -> str:
        return '' if e == 1 else f'{op}{e}' if e.denominator == 1 else f'{op}({e})'
    return format_exponent


_EXP_STYLE: dict[str, Callable[[Fraction], str]] = {
    'sup': superscript,
    '^': _make_exponent_formatter('^'),
    '**': _make_exponent_formatter('**')
}

K = TypeVar('K')


class Monomial(Generic[K]):
    '''
    A sparse, zero-purged dictionary representing a generalized mathematical monomial.

    This class serves as the core algebraic engine for physical unit calculations.
    It maps base generators (keys) to their rational exponents (values). By exploiting 
    the isomorphism between dictionary operations and free abelian groups, unit 
    multiplication/division is seamlessly handled via dictionary addition/subtraction, 
    and exponentiation is handled via scalar multiplication.

    Any key whose value evaluates to zero is automatically purged, ensuring 
    a strictly canonical representation (e.g., m² * m⁻² -> 1 -> empty dict).

    Algebraic Mapping:
    | Operator | Dictionary Semantic | Physical Unit Semantic |
    |----------|---------------------|------------------------|
    | `*`, `/` | Exponent Add/Sub    | Unit Multiply / Divide |
    | `**`     | Exponent Scaling    | Unit Exponentiation    |

    Parameters
    ----------
    elements: dict[K, int | Fraction] | None
        Initial mapping of base elements to their exponents.
        Default is an empty Monomial.

    Examples
    --------
    >>> m = Monomial({'m': 1})
    >>> s = Monomial({'s': 1})
    >>> m / s  # velocity
    Monomial({'m': 1, 's': -1})
    >>> area = m**2
    >>> area
    Monomial({'m': 2})
    >>> area / m**2
    Monomial({})
    >>> area['missing']
    Fraction(0, 1)
    '''
    __slots__ = ('_elements',)
    _elements: dict[K, Fraction]  # Mapping of base elements to their exponents

    def __init__(self, elements: dict[K, int | Fraction] | None = None, /) -> None:
        if elements is None:
            self._elements = {}
            return
        if not isinstance(elements, dict):
            raise TypeError(f'elements must be dict, got {type(elements)}.')
        self._elements = {k: common_fraction(v)
                          for k, v in elements.items() if v != 0}

    @classmethod
    def _from_dict(cls, elements: dict[K, Fraction], /) -> Self:
        '''Direct constructor from dict without copy.'''
        assert all(isinstance(v, Fraction) for v in elements.values()), 'All values in elements must be of type Fraction.'
        assert all(v != 0 for v in elements.values()), 'All values in elements must be non-zero.'
        obj = object.__new__(cls)
        obj._elements = elements
        return obj

    def __delattr__(self, name: str) -> None:
        raise AttributeError('Cannot delete attribute of Monomial.')

    def __contains__(self, key: K) -> bool: return key in self._elements

    def __getitem__(self, key: K) -> Fraction: return self._elements.get(key, ZERO)

    def __setitem__(self, key: K, value: int | Fraction | float) -> None:
        if value == 0:
            self._elements.pop(key, None)
        else:
            self._elements[key] = common_fraction(value, stacklevel=2)

    def __delitem__(self, key: K) -> None: del self._elements[key]

    def __iter__(self) -> Iterator[K]: return iter(self._elements)

    def __repr__(self) -> str:
        kv = ', '.join(f'{k!r}: {v}' for k, v in self._elements.items())
        return f'{type(self).__name__}(' + '{' + kv + '})'

    def __str__(self) -> str: return self.format()

    def format(self, *, mul: Literal['⋅', '*', ' * ', ' '] = '⋅',
               exp: Literal['sup', '^', '**'] = 'sup', frac: bool = True, 
               order: Literal['insertion', 'ascending', 'descending'] = 'insertion') -> str:
        '''
        Format specification for Monomial:
        - `mul` specifies the symbol used to separate factors.
        - `exp` specifies the style of exponent representation:
            - `'sup'`: superscript (default)
            - `'^'`: caret notation (e.g., x^2)
            - `'**'`: Python-style exponentiation (e.g., x**2)
        - `frac=True` indicates that the monomial is formatted as a fraction,
            with the factors having negative exponents placed in the denominator.
            If omitted, the monomial is formatted as a product of factors with
            both positive and negative exponents.
        - `order` specifies the order of factors: ascending and descending sort
            by the keys when they are mutually comparable; otherwise, they fall
            back to their string representations.
            - `'insertion'`: insertion order (default)
            - `'ascending'`: ascending order
            - `'descending'`: descending order
            
        '''
        if mul not in ('⋅', '*', ' * ', ' '):
            raise ValueError(f'Invalid format specification: mul={mul!r}.')
        if exp not in _EXP_STYLE:
            raise ValueError(f'Invalid format specification: exp={exp!r}.')
        exp_style = _EXP_STYLE[exp]
        if order == 'ascending':
            components = _sort_components(self.components())
        elif order == 'descending':
            components = _sort_components(self.components(), reverse=True)
        elif order == 'insertion':
            components = self.components()
        else:
            raise ValueError(f'Invalid format specification: order={order!r}.')
        if frac:
            numerator = mul.join(f'{b}{exp_style(e)}' for b, e in components if e > 0) or '1'
            denominator = mul.join(f'{b}{exp_style(-e)}' for b, e in components if e < 0)
            if sum(1 for _, e in components if e < 0) > 1:
                denominator = f'({denominator})'
            div = ' / ' if mul == ' * ' else '/'
            return f'{numerator}{div}{denominator}' if denominator else numerator
        return mul.join(f'{b}{exp_style(e)}' for b, e in components) or '1'

    def __len__(self) -> int: return len(self._elements)

    def copy(self) -> Self: return self._from_dict(self._elements.copy())

    def bases(self) -> KeysView[K]: return self._elements.keys()

    def exponents(self) -> ValuesView[Fraction]: return self._elements.values()

    def components(self) -> ItemsView[K, Fraction]: return self._elements.items()

    @property
    def numerator(self) -> Self:
        return self._from_dict({k: v for k, v in self.components() if v > 0})

    @property
    def denominator(self) -> Self:
        return self._from_dict({k: -v for k, v in self.components() if v < 0})

    def pop(self, key: K, default=ZERO) -> Fraction:
        return self._elements.pop(key, default)

    def clear(self): self._elements.clear()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._elements == other._elements

    def __mul__(self, other: Self) -> Self:
        if not isinstance(other, Monomial):
            return NotImplemented
        result = self.copy()
        result *= other
        return result

    def __truediv__(self, other: Self) -> Self:
        if not isinstance(other, Monomial):
            return NotImplemented
        result = self.copy()
        result /= other
        return result

    def __pow__(self, other: int | Fraction | float) -> Self:
        if other == 0:
            return self._from_dict({})
        other = common_fraction(other, stacklevel=2)
        return self._from_dict({k: v * other for k, v in self.components()})

    def __rtruediv__(self, other: Literal[1]) -> Self:
        if other == 1:
            return self._from_dict({k: -v for k, v in self.components()})
        return NotImplemented

    def __imul__(self, other: Self) -> Self:
        if not isinstance(other, Monomial):
            return NotImplemented
        for k, v in other.components():
            if k in self._elements:
                if (new_v := self._elements[k] + v) == 0:
                    del self._elements[k]
                else:
                    self._elements[k] = new_v
            else:
                self._elements[k] = v
        return self

    def __itruediv__(self, other: Self) -> Self:
        if not isinstance(other, Monomial):
            return NotImplemented
        if other is self:
            self._elements.clear()
            return self
        for k, v in other.components():
            if k in self._elements:
                if (new_v := self._elements[k] - v) == 0:
                    del self._elements[k]
                else:
                    self._elements[k] = new_v
            else:
                self._elements[k] = -v
        return self

    def __ipow__(self, other: int | Fraction | float) -> Self:
        if other == 0:
            self._elements.clear()
            return self
        if other == 1:
            return self
        other = common_fraction(other, stacklevel=2)
        for k, v in self.components():
            self._elements[k] = v * other
        return self
