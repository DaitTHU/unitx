"""Read-only numerical coefficients shared by electromagnetic catalogues."""

from dataclasses import dataclass

from ..quantity import Quantity


@dataclass(frozen=True, slots=True)
class _EMCoefficients:
    """Coefficients for the following common electromagnetic equations.

    Coulomb force: `F = k1*q1*q2/r**2`.
    Lorentz force: `F = q*(E + alpha*(v cross B))`.
    Maxwell-Ampere: `curl B = 4*pi*k2*J + (k2/k1)*dE/dt`.
    The wave-speed constraint gives `k2 = k1/(alpha*c**2)`.

    SI uses `k1 = 1/(4*pi*epsilon0)` and `alpha = 1`; Gaussian uses
    `k1 = 1` and `alpha = 1/c`. ESU uses `(1, 1)`, EMU `(c**2, 1)` and
    Heaviside–Lorentz `(1/(4*pi), 1/c)`. Here `alpha` is a Lorentz-force coefficient,
    distinct from the fine-structure constant `SI.alpha`. All fields are
    ordinary Quantities with float magnitudes. They select formula coefficients;
    inputs still need the matching electromagnetic representation.

    Convention reference: S. E. Kruger (2019), section II and table II,
    https://arxiv.org/abs/1909.11264 (its `k3` is named `alpha` here).
    Physical inputs use the BIPM defining speed of light and CODATA 2022 mu0;
    nominal values and floating evaluation do not propagate uncertainty.
    https://physics.nist.gov/cuu/pdf/all.pdf
    """

    k1: Quantity[float]
    """Coulomb-force coefficient in `F = k1*q1*q2/r**2`."""

    alpha: Quantity[float]
    """Magnetic Lorentz-force coefficient in `F = q*(E + alpha*(v cross B))`."""

    c: Quantity[float]
    """Vacuum wave speed: `299792458 m/s`, exact before float evaluation."""

    @property
    def k2(self) -> Quantity[float]:
        """Current coefficient in `curl B`: `k1/(alpha*c**2)`, derived."""
        return Quantity(
            self.k1.value / (self.alpha.value * self.c.value**2),
            self.k1.unit / (self.alpha.unit * self.c.unit**2),
        )
