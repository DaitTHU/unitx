"""Weighted exponential-decay fit with explicit parameter scales.

Run with unitx[scipy]. The synthetic voltage measurements have independent,
known Gaussian standard deviations; covariance is a local linear estimate.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from unitx import Quantity
from unitx.si import si
from unitx.typing import Array, NumpyValue, Scalar


def decay_jacobian(
    parameters: NDArray[np.float64],
    time: Quantity[Array],
    sigma: Quantity[Scalar],
) -> NDArray[np.float64]:
    """Differentiate weighted residuals with respect to the numerical V/s vector."""
    amplitude_scale, time_scale = 1.0 * si.V, 1.0 * si.s
    amplitude = parameters[0] * amplitude_scale
    lifetime = parameters[1] * time_scale
    envelope = np.exp(-time / lifetime)
    amplitude_column = amplitude_scale * envelope / sigma
    lifetime_column = amplitude * envelope / sigma * (time / lifetime) * (time_scale / lifetime)
    return np.column_stack((amplitude_column.n, lifetime_column.n))


def fit_decay(
    time: Quantity[Array],
    signal: Quantity[Array],
    sigma: Quantity[Scalar],
    *,
    analytic_jacobian: bool = True,
) -> tuple[Quantity[NumpyValue], Quantity[NumpyValue], NDArray[np.float64]]:
    """Return amplitude, decay time and covariance in numerical V/s scales.

    Set `analytic_jacobian=False` to compare SciPy's two-point finite differences.
    """
    amplitude_scale, time_scale = 1.0 * si.V, 1.0 * si.s

    def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        amplitude = parameters[0] * amplitude_scale
        lifetime = parameters[1] * time_scale
        prediction = amplitude * np.exp(-time / lifetime)
        # The solver sees dimensionless residuals, weighted by known errors.
        return np.asarray(((prediction - signal) / sigma).n)

    def jacobian_callback(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        return decay_jacobian(parameters, time, sigma)

    solution = least_squares(
        residual,
        np.array([2.0, 1.0]),
        # SciPy's unannotated default infers str, although it accepts callables.
        jac=jacobian_callback if analytic_jacobian else '2-point',  # pyright: ignore[reportArgumentType]
        bounds=([0.0, 0.01], [np.inf, np.inf]),
        x_scale=1.0,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    if not solution.success:
        error_message = solution.message
        raise RuntimeError(error_message)
    parameters: NDArray[np.float64] = solution.x
    jacobian: NDArray[np.float64] = solution.jac
    # SVD avoids squaring the condition number in J.T @ J. This example has
    # two identifiable parameters and known absolute measurement errors.
    _, singular_values, vh = np.linalg.svd(jacobian, full_matrices=False)
    if singular_values[-1] <= np.finfo(float).eps * max(jacobian.shape) * singular_values[0]:
        error_message = 'The example fit has unidentifiable parameters.'
        raise ValueError(error_message)
    covariance = (vh.T / singular_values**2) @ vh
    return parameters[0] * amplitude_scale, parameters[1] * time_scale, covariance


def main() -> None:
    time = Quantity(np.linspace(0.0, 6.0, 41), si.s)
    truth = 2.5 * si.V * np.exp(-time / (1.7 * si.s))
    rng = np.random.default_rng(20261028)
    signal = truth + Quantity(rng.normal(0.0, 0.02, time.shape), si.V)
    sigma = 0.02 * si.V
    amplitude, lifetime, covariance = fit_decay(time, signal, sigma)
    other_amplitude, other_lifetime, other_covariance = fit_decay(
        time.to(si.ms), signal.to(si.mV), sigma.to(si.mV)
    )
    np.testing.assert_allclose(amplitude.value_in(si.V), 2.5, rtol=0.02)
    np.testing.assert_allclose(lifetime.value_in(si.s), 1.7, rtol=0.03)
    np.testing.assert_allclose(amplitude.value_in(si.V), other_amplitude.value_in(si.V), rtol=1e-7)
    np.testing.assert_allclose(lifetime.value_in(si.s), other_lifetime.value_in(si.s), rtol=1e-7)
    np.testing.assert_allclose(covariance, other_covariance, rtol=2e-6)
    # Each covariance element has its own unit; no homogeneous Quantity matrix.
    parameter_units = (si.V, si.s)
    amplitude_variance = covariance[0, 0] * parameter_units[0] ** 2
    cross_covariance = covariance[0, 1] * parameter_units[0] * parameter_units[1]
    lifetime_error = np.sqrt(covariance[1, 1]) * parameter_units[1]
    assert amplitude_variance.value > 0 and lifetime_error.value > 0
    assert cross_covariance.unit == si.V * si.s
    # Re-express parameter covariance in mV/ms; each row and column gets a scale.
    factors = np.array([1000.0, 1000.0])
    converted = covariance * factors[:, None] * factors[None, :]
    np.testing.assert_allclose(converted[0, 1] / 1e6, cross_covariance.value)
    correlation = covariance / np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    np.testing.assert_allclose(np.diag(correlation), 1)
    print(f'Amplitude: {amplitude}; lifetime: {lifetime}; lifetime error: {lifetime_error}')


if __name__ == '__main__':
    main()
