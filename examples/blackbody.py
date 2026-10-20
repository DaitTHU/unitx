"""Plot Planck's spectral radiance with NumPy and unitx."""

import numpy as np
from matplotlib import pyplot as plt

from unitx.si import SI, si

lam = np.linspace(0.01, 3, 100)[:, None] * si.um  # wavelength
T = np.asarray([3000, 4000, 5000]) * si.K  # temperature

nu = SI.c / lam  # frequency
radiance = 2 * SI.h * SI.c**2 / (lam**5 * np.expm1(SI.h * nu / (SI.kB * T)))
radiance = radiance.to('kW/sr/m2/nm')  # spectral radiance per unit wavelength

plt.plot(lam.value, radiance.value)
plt.xlabel(f'Wavelength [{lam.unit}]')
plt.ylabel(f'Spectral radiance [{radiance.unit}]')
plt.legend([f'T = {t}' for t in T])
plt.title('Blackbody radiation')
plt.show()
