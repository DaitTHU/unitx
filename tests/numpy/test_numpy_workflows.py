"""Physical workflows across coordinate statistics, circuit solves and spectra."""

import pytest

from unitx import AffineQuantity, Quantity, Unit
from unitx.exceptions import DimensionError
from unitx.factor import Factor, pi

np = pytest.importorskip('numpy')


@pytest.mark.parametrize('coordinate', ['degC', 'K'])
def test_numpy_workflow_coordinate_statistics(coordinate: str) -> None:
    readings = AffineQuantity(
        np.array([[20.0, 20.2, 19.8], [30.0, 30.4, 29.6], [40.0, 40.6, 39.4]]),
        'degC',
    ).to(coordinate)
    average = np.mean(readings, axis=1, keepdims=True)
    residuals = readings - average
    assert isinstance(average, AffineQuantity) and isinstance(residuals, Quantity)
    np.testing.assert_allclose(average.to('degC').value[:, 0], [20, 30, 40], atol=1e-12)
    spread = np.std(readings, axis=1, mean=average, correction=1)
    np.testing.assert_allclose(spread.value_in('K'), [0.2, 0.4, 0.6], atol=1e-12)

    accepted = np.abs(residuals) <= Quantity(0.25, 'K')
    cleaned = np.where(accepted, readings, average)
    cleaned_average = np.mean(cleaned, axis=1)
    np.testing.assert_allclose(cleaned_average.to('degC').value, [20, 30, 40], atol=1e-12)
    maximum = np.maximum.accumulate(readings, axis=1)
    np.testing.assert_allclose(
        maximum.to('degC').value, [[20, 20.2, 20.2], [30, 30.4, 30.4], [40, 40.6, 40.6]]
    )
    density, edges = np.histogram(residuals, bins=5, density=True)
    normalization = np.sum(density * np.diff(edges))
    np.testing.assert_allclose(normalization.value_in(''), 1)

    destination = AffineQuantity(np.zeros((3, 1)), 'K' if coordinate == 'degC' else 'degC')
    storage = destination.value
    assert np.mean(readings, axis=1, keepdims=True, out=destination) is destination
    assert destination.value is storage
    np.testing.assert_allclose(destination.to('degC').value[:, 0], [20, 30, 40], atol=1e-12)
    difference_buffer = Quantity(np.full((3, 1), 7.0), 'K')
    with pytest.raises(DimensionError):
        np.mean(readings, axis=1, keepdims=True, out=difference_buffer)
    np.testing.assert_array_equal(difference_buffer.value, 7)


@pytest.mark.parametrize('resistance_unit', [Unit('ohm'), Factor(2) * pi * Unit('ohm')])
@pytest.mark.parametrize('voltage_unit', ['V', 'mV'])
def test_numpy_workflow_circuit_solve(resistance_unit: Unit, voltage_unit: str) -> None:
    resistance = Quantity(np.array([[3.0, -1.0], [-1.0, 2.0]]), 'ohm').to(resistance_unit)
    voltage = Quantity(np.array([5.0, 0.0]), 'V').to(voltage_unit)
    current = np.linalg.solve(resistance, voltage)
    np.testing.assert_allclose(current.value_in('A'), [2, 1], rtol=1e-13)
    fitted, residuals, rank, singular = np.linalg.lstsq(resistance, voltage)
    np.testing.assert_allclose(fitted.value_in('A'), current.value_in('A'), rtol=1e-13)
    assert residuals.unit == voltage.unit**2 and rank == 2
    assert singular.unit == resistance.unit

    time = np.linspace(Quantity(0.0, 'ms'), Quantity(2.0, 's'), 11)
    current_grid = np.broadcast_to(current, (time.size, 2))
    power = np.einsum('ti,ij,tj->t', current_grid, resistance, current_grid)
    np.testing.assert_allclose(power.value_in('W'), 10, rtol=1e-13)
    energy = np.trapezoid(power, time)
    np.testing.assert_allclose(energy.value_in('J'), 20, rtol=1e-13)

    shared = resistance.value.copy()
    aliased_resistance = Quantity(shared, resistance.unit)
    view = shared[0]
    destination = Quantity(view, 'mV')
    assert np.matmul(aliased_resistance, current, out=destination) is destination
    assert destination.value is view and np.shares_memory(destination, aliased_resistance)
    np.testing.assert_allclose(destination.value, [5000, 0], atol=1e-10)

    invalid = Quantity(np.full(2, 7.0), 's')
    with pytest.raises(DimensionError):
        np.matmul(resistance, current, out=invalid)
    np.testing.assert_array_equal(invalid.value, 7)
    integer = Quantity(np.full(2, 7, dtype=int), 'V')
    with pytest.raises(TypeError):
        np.matmul(resistance, current, out=integer)
    np.testing.assert_array_equal(integer.value, 7)


@pytest.mark.parametrize('unit', [Unit('mm'), Factor(2) * pi * Unit('mm')])
def test_numpy_workflow_sampled_spectrum(unit: Unit) -> None:
    count = 128
    time, spacing = np.linspace(
        Quantity(0.0, 'ms'), Quantity(1.0, 's'), count, endpoint=False, retstep=True
    )
    signal = (Quantity(3.0, 'mm') * np.sin(2 * np.pi * Quantity(8.0, 'Hz') * time)).to(unit)
    spectrum = np.fft.rfft(signal)
    frequency = np.fft.rfftfreq(count, d=spacing)
    peak = np.argmax(np.abs(spectrum))
    np.testing.assert_allclose(frequency[peak].value_in('Hz'), 8)
    amplitude = 2 * np.abs(spectrum[peak]) / count
    np.testing.assert_allclose(amplitude.value_in('mm'), 3, rtol=1e-13)
    restored = np.fft.irfft(spectrum, n=count)
    np.testing.assert_allclose(restored.value_in('mm'), signal.value_in('mm'), atol=1e-12)

    full_spectrum = np.fft.fft(signal)
    time_power = np.mean(np.abs(signal) ** 2)
    frequency_power = np.sum(np.abs(full_spectrum) ** 2) / count**2
    np.testing.assert_allclose(time_power.value_in('mm2'), 4.5, rtol=1e-13)
    np.testing.assert_allclose(
        frequency_power.value_in('mm2'), time_power.value_in('mm2'), rtol=1e-13
    )

    shared = np.empty(count + 2)
    shared[:count] = signal.value
    aliased_signal = Quantity(shared[:count], unit)
    view = shared.view(complex)
    destination = Quantity(view, 'cm')
    assert np.fft.rfft(aliased_signal, out=destination) is destination
    assert destination.value is view and np.shares_memory(destination, aliased_signal)
    np.testing.assert_allclose(destination.value, spectrum.value_in('cm'), atol=1e-12)
    invalid = Quantity(np.full(spectrum.shape, 7.0 + 0j), 's')
    with pytest.raises(DimensionError):
        np.fft.rfft(signal, out=invalid)
    np.testing.assert_array_equal(invalid.value, 7)
