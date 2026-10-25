from __future__ import annotations

from collections.abc import Hashable, ItemsView, KeysView, Mapping, ValuesView
from fractions import Fraction
from types import MappingProxyType
from typing import Literal

from ._formatting.unicode import DOT, superscript
from ._immutable import _Immutable
from ._numerics import Fractionable, as_exponent

__all__ = ['Monomial']

_ZERO = Fraction(0)


class Monomial[BaseT: Hashable](_Immutable):
    """An immutable formal product of bases raised to rational exponents.

    A monomial represents `product(base ** exponent)` as a sparse mapping from
    hashable bases to Fraction exponents. An empty monomial represents the
    multiplicative identity.

    Zero exponents are omitted. `exponent_of(base)` returns zero for a missing
    base. Equality and hashing depend on the base-exponent pairs, regardless
    of insertion order.

    Parameters
    ----------
    components : Mapping[BaseT, Fractionable], optional
        Bases and exponents. The mapping is copied and zero exponents are
        removed. Defaults to an empty monomial. Use int, Fraction, or Decimal
        for exact exponents. Float exponents use their shortest decimal text,
        so 0.1 means Fraction(1, 10); no denominator limit is applied.

    Notes
    -----
    Use `bases()`, `exponents()`, or `components()` to inspect or iterate
    over the stored factors in insertion order. Use `len(monomial.bases())`
    to count bases.

    Monomials can be dictionary keys. Their bases must retain stable hashes
    and equality for the lifetime of the monomial. Unit and Factor can contain
    a Monomial and build their domain-specific rules around this formal algebra.

    Examples
    --------
    >>> m = Monomial({'m': 1})
    >>> s = Monomial({'s': 1})
    >>> m / s
    Monomial({'m': 1, 's': -1})
    >>> m**2
    Monomial({'m': 2})
    >>> m / m
    Monomial()
    >>> (m / s) ** Fraction(1, 2)
    Monomial({'m': Fraction(1, 2), 's': Fraction(-1, 2)})
    >>> m.exponent_of('missing')
    Fraction(0, 1)
    >>> str(m / s)
    'm/s'
    >>> bool(Monomial())
    True
    """

    __slots__ = ('_components',)
    _components: MappingProxyType[BaseT, Fraction]

    def __new__(cls, components: Mapping[BaseT, Fractionable] | None = None, /) -> Monomial[BaseT]:
        """Construct a new Monomial from a mapping of bases to their exponents."""
        if components is None:
            components = {}
        if not isinstance(components, Mapping):
            error_message = f'Expected a mapping for components, got {type(components).__name__}.'
            raise TypeError(error_message)
        normalized: dict[BaseT, Fraction] = {}
        for base, exp in components.items():
            if exponent := as_exponent(exp):
                normalized[base] = exponent
        obj = super().__new__(cls)
        object.__setattr__(obj, '_components', MappingProxyType(normalized))
        return obj

    @staticmethod
    def _from_dict(components: dict[BaseT, Fraction], /) -> Monomial[BaseT]:
        """Take ownership of a normalized dictionary. Internal use only."""
        assert all(type(exp) is Fraction for exp in components.values()), 'Exponent not Fraction'
        assert all(exp != 0 for exp in components.values()), 'Zero exponent'
        obj = object.__new__(Monomial)
        object.__setattr__(obj, '_components', MappingProxyType(components))
        return obj

    def __contains__(self, base: BaseT) -> bool:
        return base in self._components

    def exponent_of(self, base: BaseT) -> Fraction:
        """Return the base's exponent, or Fraction(0) if the base is absent."""
        return self._components.get(base, _ZERO)

    def __repr__(self) -> str:
        components = ', '.join(
            f'{base!r}: {exp}' if exp.denominator == 1 else f'{base!r}: {exp!r}'
            for base, exp in self.components()
        )
        if not components:
            return f'{type(self).__name__}()'
        return f'{type(self).__name__}({{{components}}})'

    def __str__(self) -> str:
        numerator = DOT.join(
            f'{base}{superscript(exp)}' for base, exp in self.components() if exp > 0
        )
        denominator = DOT.join(
            f'{base}{superscript(-exp)}' for base, exp in self.components() if exp < 0
        )
        if not numerator:
            numerator = '1'
        if not denominator:
            return numerator
        if sum(exp < 0 for exp in self.exponents()) > 1:
            denominator = f'({denominator})'
        return f'{numerator}/{denominator}'

    def __copy__(self) -> Monomial[BaseT]:
        return self

    def __reduce__(self) -> tuple[type[Monomial[BaseT]], tuple[dict[BaseT, Fraction]]]:
        return Monomial, (dict(self._components),)

    def bases(self) -> KeysView[BaseT]:
        """Return the bases as dict-keys of the monomial."""
        return self._components.keys()

    def exponents(self) -> ValuesView[Fraction]:
        """Return the exponents as dict-values of the monomial."""
        return self._components.values()

    def components(self) -> ItemsView[BaseT, Fraction]:
        """Return the base-exponent pairs as dict-items of the monomial."""
        return self._components.items()

    @property
    def numerator(self) -> Monomial[BaseT]:
        """A new Monomial containing only the bases with positive exponents."""
        return self._from_dict({base: exp for base, exp in self.components() if exp > 0})

    @property
    def denominator(self) -> Monomial[BaseT]:
        """A new Monomial containing denominator bases with positive exponents."""
        return self._from_dict({base: -exp for base, exp in self.components() if exp < 0})

    def __hash__(self) -> int:
        return hash(frozenset(self.components()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Monomial):
            return NotImplemented
        return self._components == other._components

    def __mul__(self, other: Monomial[BaseT]) -> Monomial[BaseT]:
        if not isinstance(other, Monomial):
            return NotImplemented
        components = self._components.copy()
        for base, exponent in other.components():
            if (current_exponent := components.get(base)) is None:
                components[base] = exponent
            elif updated_exponent := current_exponent + exponent:
                components[base] = updated_exponent
            else:
                del components[base]
        return self._from_dict(components)

    def __truediv__(self, other: Monomial[BaseT]) -> Monomial[BaseT]:
        if not isinstance(other, Monomial):
            return NotImplemented
        components = self._components.copy()
        for base, exponent in other.components():
            if (current_exponent := components.get(base)) is None:
                components[base] = -exponent
            elif updated_exponent := current_exponent - exponent:
                components[base] = updated_exponent
            else:
                del components[base]
        return self._from_dict(components)

    def __pow__(self, other: Fractionable) -> Monomial[BaseT]:
        if not isinstance(other, Fractionable):
            return NotImplemented
        other_as_fraction = as_exponent(other)
        if other_as_fraction == 0:
            return self._from_dict({})
        return self._from_dict({base: exp * other_as_fraction for base, exp in self.components()})

    def __rtruediv__(self, other: Literal[1]) -> Monomial[BaseT]:
        if type(other) is int and other == 1:
            return self._from_dict({base: -exp for base, exp in self.components()})
        return NotImplemented
