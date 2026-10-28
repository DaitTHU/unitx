"""Matplotlib owns numeric extraction and preserves axis unit semantics."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity
from unitx.factor import Factor, pi
from unitx.si import si

np = pytest.importorskip('numpy')
Figure = pytest.importorskip('matplotlib.figure').Figure
units = pytest.importorskip('matplotlib.units')
ConversionError, registry = units.ConversionError, units.registry
register = pytest.importorskip('unitx.matplotlib').register


@pytest.fixture(autouse=True)
def converters() -> Iterator[None]:
    before = dict(registry)
    register()
    yield
    registry.clear()
    registry.update(before)


def test_matplotlib_register_repeated() -> None:
    converter = registry[Quantity]
    register()
    assert registry[Quantity] is registry[AffineQuantity] is registry[LogQuantity] is converter


def test_matplotlib_plot_units() -> None:
    figure = Figure()
    axes = figure.subplots()
    time = Quantity(np.array([0.0, 1, 2]), si.ms)
    length = Quantity(np.array([1.0, 2, 3]), si.m)
    (line,) = axes.plot(time, length)
    np.testing.assert_array_equal(line.get_xdata(orig=False), [0, 1, 2])
    np.testing.assert_array_equal(line.get_ydata(orig=False), [1, 2, 3])
    assert isinstance(line.get_ydata(orig=True), Quantity)
    assert axes.get_xlabel() == 'ms' and axes.get_ylabel() == 'm'
    assert axes.xaxis.get_units() == si.ms and axes.yaxis.get_units() == si.m
    axes.yaxis.set_units(si.cm)
    np.testing.assert_array_equal(line.get_ydata(orig=False), [100, 200, 300])
    assert axes.get_ylabel() == 'cm'
    with pytest.raises(TypeError, match='Implicit unit removal'):
        np.asarray(length)


def test_matplotlib_plot_explicit_units() -> None:
    axes = Figure().subplots()
    axes.set_ylabel('Distance')
    (line,) = axes.plot(
        Quantity(np.array([0.0, 1]), si.s),
        Quantity(np.array([1.0, 2]), si.m),
        xunits=si.ms,
        yunits=si.cm,
    )
    np.testing.assert_array_equal(line.get_xdata(orig=False), [0, 1000])
    np.testing.assert_array_equal(line.get_ydata(orig=False), [100, 200])
    assert axes.get_ylabel() == 'Distance'
    axes.set_xlim(Quantity(100, si.ms), Quantity(800, si.ms))
    np.testing.assert_allclose(axes.get_xlim(), [100, 800])


def test_matplotlib_plot_complete_scale() -> None:
    axes = Figure().subplots()
    q = Quantity(np.array([1.0, 2]), Factor(2) * pi * si.cm)
    (line,) = axes.plot(q, yunits=si.m)
    np.testing.assert_allclose(line.get_ydata(orig=False), [0.02 * np.pi, 0.04 * np.pi])


def test_matplotlib_plot_aligned_lines() -> None:
    axes = Figure().subplots()
    axes.plot(Quantity(np.array([1.0, 2]), si.m))
    (line,) = axes.plot(Quantity(np.array([100.0, 200]), si.cm))
    np.testing.assert_array_equal(line.get_ydata(orig=False), [1, 2])


@pytest.mark.parametrize(
    'point,target,expected',
    [
        (
            AffineQuantity(np.array([0.0, 10]), si.degC),
            AffineQuantity(0, si.K).unit,
            [273.15, 283.15],
        ),
        (LogQuantity(np.array([0.0, 10]), si.dBW), si.dBm, [30.0, 40.0]),
    ],
)
def test_matplotlib_plot_point(point: Any, target: Any, expected: list[float]) -> None:
    axes = Figure().subplots()
    (line,) = axes.plot(point, yunits=target)
    np.testing.assert_allclose(line.get_ydata(orig=False), expected)
    assert axes.yaxis.get_units() == target


def test_matplotlib_plot_scalar() -> None:
    axes = Figure().subplots()
    (line,) = axes.plot(Quantity(1.0, si.s), Quantity(2.0, si.m))
    np.testing.assert_array_equal(line.get_xydata(), [[1, 2]])


@pytest.mark.parametrize('q', [AffineQuantity(1.0, si.degC), LogQuantity(1.0, si.dBm)])
def test_matplotlib_point_scalar_limit(q: Any) -> None:
    axes = Figure().subplots()
    axes.plot(q)
    axes.set_ylim(q, q + Quantity(1, q.unit.delta_unit))
    np.testing.assert_allclose(axes.get_ylim(), [1, 2])


def test_matplotlib_plot_empty() -> None:
    axes = Figure().subplots()
    (line,) = axes.plot(Quantity(np.array([]), si.m))
    assert line.get_ydata(orig=False).size == 0
    assert axes.yaxis.get_units() == si.m


@pytest.mark.parametrize(
    'other', [Quantity(np.array([1.0, 2]), si.s), AffineQuantity(np.array([1.0, 2]), si.K)]
)
def test_matplotlib_plot_invalid_dimension(other: Any) -> None:
    axes = Figure().subplots()
    axes.plot(Quantity(np.array([1.0, 2]), si.m))
    with pytest.raises(ConversionError):
        axes.plot(other)


def test_matplotlib_plot_invalid_point_difference() -> None:
    axes = Figure().subplots()
    axes.plot(AffineQuantity(np.array([0.0, 10]), si.degC))
    with pytest.raises(ConversionError):
        axes.plot(Quantity(np.array([0.0, 10]), si.K))


def test_matplotlib_plot_invalid_complex() -> None:
    with pytest.raises(ConversionError):
        Figure().subplots().plot(Quantity(np.array([1 + 2j]), si.V))


def test_matplotlib_plot_invalid_convention() -> None:
    axes = Figure().subplots()
    axes.plot(LogQuantity(np.array([0.0]), si.dB(si.mW)))
    with pytest.raises(ConversionError):
        axes.plot(LogQuantity(np.array([0.0]), si.dB(si.mW, kind='amplitude')))


def test_matplotlib_plot_render(tmp_path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figure = Figure()
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.plot(Quantity(np.array([0.0, 1, 2]), si.s), Quantity(np.array([1.0, 2, 1]), si.m))
    figure.savefig(tmp_path / 'quantities.png')
    assert (tmp_path / 'quantities.png').stat().st_size > 1000


@pytest.mark.parametrize(
    'cls,unit,target,expected',
    [
        (Quantity, si.m, si.cm, [100, 200, 300]),
        (AffineQuantity, si.degC, AffineQuantity(0, si.K).unit, [274.15, 275.15, 276.15]),
        (LogQuantity, si.dBW, si.dBm, [31, 32, 33]),
    ],
)
def test_matplotlib_collections(cls: Any, unit: Any, target: Any, expected: list[float]) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    time = Quantity(np.array([0.0, 1, 2]), si.s)
    q = cls(np.array([1.0, 2, 3]), unit)
    delta = q.unit if cls is Quantity else q.unit.delta_unit
    figure = Figure()
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    dots = axes.scatter(time, q, xunits=si.ms, yunits=target)
    np.testing.assert_allclose(dots.get_offsets()[:, 0], [0, 1000, 2000])
    np.testing.assert_allclose(dots.get_offsets()[:, 1], expected)
    band = axes.fill_between(time, q, q + 1 * delta)
    vertices = np.concatenate([path.vertices for path in band.get_paths()])
    np.testing.assert_allclose(vertices[:, 0].min(), 0)
    np.testing.assert_allclose(vertices[:, 0].max(), 2000)
    np.testing.assert_allclose(vertices[:, 1].min(), expected[0])
    np.testing.assert_allclose(vertices[:, 1].max(), (q + 1 * delta).value_in(target)[-1])
    figure.canvas.draw()


def test_matplotlib_errorbar_boundary() -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figure = Figure()
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    time = Quantity(np.array([0.0, 1.0]), si.s)
    readings = AffineQuantity(np.array([20.0, 21.0]), si.degC)
    errors = Quantity(np.array([0.2, 0.3]), si.K)
    container = axes.errorbar(
        time.value_in(si.s), readings.value_in(si.degC), yerr=errors.value_in(si.degC.delta_unit)
    )
    segments = container.lines[2][0].get_segments()
    np.testing.assert_allclose(segments, [[[0, 19.8], [0, 20.2]], [[1, 20.7], [1, 21.3]]])
    figure.canvas.draw()
