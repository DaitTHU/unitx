# Array storage with explicit metadata

Use NumPy for numeric storage and the existing scalar record format for the
unit/coordinate metadata. This recipe supports the three built-in quantity
families with plain numeric ndarrays. It stores a zero-valued scalar of the same
family as a metadata template, preserving complete units, coordinate origins,
physical references and log conventions. It does not turn scalar JSON into an
array codec or promise to preserve subclasses.

The `.npz` values retain shape, dtype and stored readings, including complex
linear values. Memory layout, views and shared storage are not preserved. The
JSON template's zero is only metadata; it is replaced by the loaded numeric
array. No numerical unit conversion occurs.

```python
import io
import json
import numpy as np
from unitx import AffineQuantity, LogQuantity, Quantity
from unitx.serialization import from_dict, to_dict
from unitx.si import si

originals = [
    Quantity(np.array([[1 + 2j, 3 - 4j]], dtype=np.complex64), si.mV),
    AffineQuantity(np.array([20.0, 25.0], dtype=np.float32), si.degC),
    LogQuantity(np.array([-10.0, 0.0]), si.dBm),
]
for original in originals:
    template = type(original)(0, original.unit)
    metadata = json.dumps(to_dict(template), allow_nan=False)
    archive = io.BytesIO()  # Replace with a file path in an application.
    np.savez(archive, values=original.value, metadata=np.array(metadata))
    archive.seek(0)
    with np.load(archive, allow_pickle=False) as saved:
        prototype = from_dict(json.loads(str(saved['metadata'].item())))
        assert isinstance(prototype, (Quantity, AffineQuantity, LogQuantity))
        restored = type(prototype)(saved['values'].copy(), prototype.unit)
    assert type(restored) is type(original)
    assert restored.unit == original.unit
    assert restored.value.dtype == original.value.dtype
    assert restored.shape == original.shape
    np.testing.assert_array_equal(restored.value, original.value)
```

For custom units/coordinates, load the matching definitions first and supply
`registry=your_registry` to `from_dict`. The scalar record checks definition
identity and meaning; storing only `unit.symbol` would lose this guarantee.
See [registry requirements](serialization.md#registry-requirements) for missing
or conflicting definitions. Anonymous coordinates retain their complete mapping
under the same prerequisites for their referenced units. Avoid object arrays;
`allow_pickle=False` is intentional. Applications accepting external archives
should validate their own keys, expected family, dtype, shape and size limits.

Keep the scalar schema version in the JSON. Applications extending this archive
layout should version their own envelope separately, and record the unitx,
NumPy and constant-data versions needed to reproduce calculations. NumPy storage
and [unitx record compatibility](compatibility.md#data-and-reconstruction) are
separate contracts.
