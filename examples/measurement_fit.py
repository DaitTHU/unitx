"""Synthetic measurements, weighted decay fitting and error bars.

Install unitx[scipy,matplotlib]. Keep nonlinear_fit.py beside this file.
The independent Gaussian errors have known sigma; covariance is a local
linear approximation. The random seed makes this demonstration reproducible.
"""

from math import sqrt

import numpy as np
from matplotlib import pyplot as plt
from nonlinear_fit import fit_decay

from unitx import Quantity, math
from unitx.si import si

time = Quantity(np.linspace(0.0, 6.0, 41), si.s)
truth = 2.5 * si.V * math.exp(-time / (1.7 * si.s))
sigma = 0.02 * si.V
noise = Quantity(np.random.default_rng(20261028).normal(0.0, sigma.value, time.shape), sigma.unit)
signal = truth + noise
amplitude, lifetime, covariance = fit_decay(time, signal, sigma)
other_amplitude, other_lifetime, other_covariance = fit_decay(
    time.to(si.ms), signal.to(si.mV), sigma.to(si.mV)
)
np.testing.assert_allclose(amplitude.value_in(si.V), 2.5, rtol=0.02)
np.testing.assert_allclose(lifetime.value_in(si.s), 1.7, rtol=0.03)
np.testing.assert_allclose(other_amplitude.value_in(amplitude.unit), amplitude.value, rtol=1e-7)
np.testing.assert_allclose(other_lifetime.value_in(lifetime.unit), lifetime.value, rtol=1e-7)
np.testing.assert_allclose(other_covariance, covariance, rtol=2e-6)
lifetime_error = sqrt(covariance[1, 1]) * si.s
prediction = amplitude * math.exp(-time / lifetime)

# errorbar needs numerical inputs. Select one unit for each axis and for
# the uncertainty width; all three conversions remain explicit.
fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
ax.errorbar(
    time.value_in(si.ms),
    signal.value_in(si.mV),
    yerr=sigma.value_in(si.mV),
    fmt='o',
    markersize=3,
    label='Synthetic measurements',
)
ax.plot(time.value_in(si.ms), prediction.value_in(si.mV), label='Weighted fit')
ax.set(xlabel=f'Time / {si.ms}', ylabel=f'Signal / {si.mV}', title='Exponential decay')
ax.legend()
ax.grid(True, alpha=0.25)

if __name__ == '__main__':
    print(f'Amplitude: {amplitude}; lifetime: {lifetime}; standard error: {lifetime_error}')
    plt.show()
