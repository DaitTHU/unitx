"""Integrate a sampled power pulse, save it with units and restore it.

Install unitx[matplotlib]. The in-memory archive writes no project files;
replace BytesIO with a path to retain it. Finite-grid quadrature is approximate.
"""

import io
import json

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from unitx import Quantity, Unit, math
from unitx.numpy import trapezoid
from unitx.serialization import from_dict, to_dict
from unitx.si import si

time = Quantity(np.linspace(-8.0, 8.0, 2001), si.ms)
width = 1.5 * si.ms
peak_power = 2.0 * si.W
power = peak_power * math.exp(-0.5 * (time / width) ** 2)
energy = trapezoid(power, x=time).to(si.mJ)
expected_energy = peak_power * width * np.sqrt(2 * np.pi)
assert energy.isclose(expected_energy, rel_tol=2e-7)
other_energy = trapezoid(power.to(si.mW), x=time.to(si.s))
assert energy.isclose(other_energy, rel_tol=1e-12)

# The existing scalar schema preserves complete Unit identities and scales.
# NumPy owns numeric shape/dtype/storage; no object arrays or pickle are used.
metadata = json.dumps({'time': to_dict(time.unit), 'power': to_dict(power.unit)})
archive = io.BytesIO()
np.savez(archive, time=time.value, power=power.value, metadata=np.array(metadata))
archive.seek(0)
with np.load(archive, allow_pickle=False) as saved:
    records = json.loads(str(saved['metadata'].item()))
    time_unit = from_dict(records['time'])
    power_unit = from_dict(records['power'])
    assert isinstance(time_unit, Unit) and isinstance(power_unit, Unit)
    # np.load has dynamically typed records. State and validate this archive's
    # numerical contract at the storage boundary, before constructing quantities.
    time_values: NDArray[np.float64] = saved['time']
    power_values: NDArray[np.float64] = saved['power']
    assert isinstance(time_values, np.ndarray) and isinstance(power_values, np.ndarray)
    assert time_values.dtype == np.float64 and power_values.dtype == np.float64
    assert time_values.shape == time.shape and power_values.shape == power.shape
    restored_time = Quantity(time_values, time_unit)
    restored_power = Quantity(power_values, power_unit)
assert restored_time.unit == time.unit and restored_power.unit == power.unit
assert restored_power.value.dtype == power.value.dtype and restored_power.shape == power.shape
np.testing.assert_array_equal(restored_time.value, time.value)
np.testing.assert_array_equal(restored_power.value, power.value)
restored_energy = trapezoid(restored_power, x=restored_time)
assert restored_energy.isclose(energy, rel_tol=1e-12)

fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
ax.plot(restored_time.value_in(si.ms), restored_power.value_in(si.W))
ax.set(xlabel=f'Time / {si.ms}', ylabel=f'Power / {si.W}', title=f'Pulse energy: {energy:.3f}')
ax.grid(True, alpha=0.25)

if __name__ == '__main__':
    print(f'Integrated energy: {energy}; restored: {restored_energy.to(si.mJ)}')
    plt.show()
