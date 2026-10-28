"""Profile a warmed workload, keeping imports outside the recorded region.

Use --profile PREFIX for Pyinstrument HTML and speedscope sampling reports,
or --cprofile FILE for standard-library deterministic call statistics.
Sampling and deterministic profiles are diagnostic data, not benchmark timings.
"""

import argparse
import cProfile
import runpy
import time
from collections.abc import Callable
from pathlib import Path


def workload(name: str) -> Callable[[], object]:
    if name == 'scalar':
        from unitx import Quantity
        from unitx.si import si

        q = 1.5 * si.m
        return lambda: (q * q + Quantity(10000.0, si.cm**2)).value_in(si.m**2)
    import numpy as np

    from unitx import Quantity
    from unitx.si import si

    if name == 'fit':
        fit = runpy.run_path(
            str(Path(__file__).resolve().parents[1] / 'examples/nonlinear_fit.py')
        )['fit_decay']
        x = Quantity(np.linspace(0.0, 6.0, 41), si.s)
        y = 2.5 * si.V * np.exp(-x / (1.7 * si.s))
        sigma = 0.02 * si.V
        return lambda: fit(x, y, sigma)
    q = Quantity(np.linspace(1.0, 2.0, 41), si.m)
    denominator = 2.0 * si.m
    return lambda: np.exp(-q / denominator).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workload', choices=['scalar', 'array', 'fit'])
    parser.add_argument('--seconds', type=float, default=10.0)
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        '--profile', type=Path, help='Write PREFIX.html and PREFIX.speedscope.json.'
    )
    output.add_argument('--cprofile', type=Path, help='Write deterministic .pstats data.')
    options = parser.parse_args()
    if not 0 < options.seconds < float('inf'):
        parser.error('--seconds must be positive and finite.')
    operation = workload(options.workload)
    operation()
    if options.profile:
        from pyinstrument import Profiler
        from pyinstrument.renderers import HTMLRenderer, SpeedscopeRenderer

        sampling = Profiler(interval=0.001)
        sampling.start()
    elif options.cprofile:
        deterministic = cProfile.Profile()
        deterministic.enable()
    deadline = time.perf_counter() + options.seconds
    while time.perf_counter() < deadline:
        operation()
    if options.profile:
        sampling.stop()
        options.profile.with_suffix('.html').write_text(sampling.output(HTMLRenderer()))
        options.profile.with_suffix('.speedscope.json').write_text(
            sampling.output(SpeedscopeRenderer(processor_options={'filter_threshold': 0}))
        )
    elif options.cprofile:
        deterministic.disable()
        deterministic.dump_stats(options.cprofile)


if __name__ == '__main__':
    main()
