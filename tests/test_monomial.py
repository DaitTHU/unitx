import pytest

from unitx.monomial import Monomial


def test_monomial_creation() -> None:
    m1 = Monomial({'x': 2, 'y': 3})
    m2 = Monomial({'x': 2, 'y': 3})
    m3 = Monomial({'x': 0, 'y': 3})

    assert m1 == m2
    assert m1 != m3
    assert len(m1) == 2
    assert set(m1.bases()) == {'x', 'y'}
    assert set(m1.exponents()) == {2, 3}
    assert set(m1.components()) == {('x', 2), ('y', 3)}
    assert repr(m1) == "Monomial({'x': 2, 'y': 3})"
    assert str(m1) == "x²⋅y³"
    assert set(m3.bases()) == {'y'}
    assert str(m3) == "y³"


def test_monomial_invalid_creation() -> None:
    with pytest.raises(TypeError):
        Monomial({'x': 2, 'y': None})  # type: ignore

    with pytest.raises(TypeError):
        Monomial({'x', 'y'})  # type: ignore


def test_monomial_operations() -> None:
    m1 = Monomial({'x': 2, 'y': 3})
    m2 = Monomial({'x': 1, 'z': 4})

    m_mul = m1 * m2
    m_div = m1 / m2
    m_pow = m1 ** 2
    m_inv = 1 / m1

    assert m_mul == Monomial({'x': 3, 'y': 3, 'z': 4})
    assert m_div == Monomial({'x': 1, 'y': 3, 'z': -4})
    assert m_pow == Monomial({'x': 4, 'y': 6})
    assert str(m_inv) == "1/(x²⋅y³)"


def test_monomial_invalid_operations() -> None:
    with pytest.raises(TypeError):
        Monomial({'x': 2}) * 2  # type: ignore

    with pytest.raises(TypeError):
        Monomial({'x': 2}) / 0  # type: ignore


def test_float_conversion_warning() -> None:
    u = Monomial({'x': 1})
    with pytest.warns(UserWarning):
        u **= 1/3

    with pytest.warns(UserWarning):
        u['x'] = 0.2

