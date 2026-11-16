"""Planck spectra: constants, broadcasting, display units and a Wien check.

Install unitx[matplotlib], then run this file. The finite wavelength grid
approximates the peak; temperatures and wavelengths are positive vacuum inputs.
"""

import numpy as np
from matplotlib import pyplot as plt

from unitx import Quantity, math
from unitx.si import SI, si

wavelength = Quantity(np.linspace(0.1, 3.0, 600)[:, None], si.um)
temperature = Quantity(np.asarray([3000.0, 4000.0, 5000.0]), si.K)
exponent = SI.h * SI.c / (wavelength * SI.kB * temperature)
radiance = 2 * SI.h * SI.c**2 / (wavelength**5 * math.expm1(exponent))
radiance = radiance.to('kW/sr/m2/nm')

peak = wavelength[np.argmax(radiance.value, axis=0), 0]
wien_peak = SI.b / temperature
assert radiance.shape == (600, 3)
assert np.all(np.isfinite(radiance.value)) and np.all(radiance.value >= 0)
assert np.all(peak.isclose(wien_peak, rel_tol=0, abs_tol=0.005 * si.um))
# Equivalent input units must describe the same spectra.
other_wavelength = wavelength.to(si.nm)
other_exponent = SI.h * SI.c / (other_wavelength * SI.kB * temperature)
other_radiance = 2 * SI.h * SI.c**2 / (other_wavelength**5 * math.expm1(other_exponent))
np.testing.assert_allclose(other_radiance.value_in(radiance.unit), radiance.value, rtol=1e-12)

fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
ax.plot(wavelength.value, radiance.value)
ax.set(
    xlabel=f'Wavelength / {wavelength.unit}',
    ylabel=f'Spectral radiance / {radiance.unit}',
    title='Blackbody radiation',
)
ax.legend([f'T = {value}' for value in temperature])
ax.grid(True, alpha=0.25)

if __name__ == '__main__':
    print('Grid peaks:', peak.to(si.nm))
    plt.show()
