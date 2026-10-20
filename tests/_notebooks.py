"""Read ordinary Python notebook cells without a Jupyter test dependency."""

import json
from pathlib import Path


def notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding='utf-8'))
    return '\n\n'.join(
        ''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code'
    )
