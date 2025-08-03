r"""
Physical and mathematical constants and related functions used throughout the package.
Most masses and widths are taken by the Review of Particle Physics by the Particle Data Group :cite:`Zyla:2020zbs`.
The base unit is taken to be `GeV`.
"""

import numpy as np


MeV = 1e-3
r"""Megaelectronvolt, numerically defined as :math:`10^{-3}` as a default (so that :code:`GeV` :math:`=1`)."""

eV = 1e-6 * MeV
r"""Electronvolt, numerically defined as :math:`10^{-6}\times` :code:`MeV`."""

keV = 1e-3 * MeV
r"""Kiloelectronvolt, numerically defined as :math:`10^{-3}\times` :code:`MeV`."""

GeV = 1e3 * MeV
r"""Gigaelectronvolt, numerically defined as :math:`10^{3}\times` :code:`MeV`."""

TeV = 1e6 * GeV
r"""Teraelectronvolt, numerically defined as :math:`10^{6}\times` :code:`MeV`."""

alpha_fs = 1 / 137
r"""Electromagnetic fine structure constant :math:`\alpha_{\text{fs}}\approx1/137`."""

e2 = 4 * np.pi * alpha_fs
r"""Electric charge squared, :math:`e^2 = 4\pi\alpha_{\text{fs}}`."""

to_nb = 3.894e5 * GeV**2
r"""Conversion factor from natural units to nanobarns.

**Note**: a "natural unit" is defined by whatever is equal to `1` by default.
If :math:`\text{GeV}=1`, then the constant :code:`to_nb` will convert
:math:`{\text{GeV}}^{-2}` to nanobarns."""

degrees = np.pi / 180
r"""Conversion factor from radians to degrees :math:`=\pi/180`.

----"""


M_PI = 139.57 * MeV
r"""Mass of the charged pion :cite:`Zyla:2020zbs`."""

M_PI0 = 134.98 * MeV
r"""Mass of the neutral pion :cite:`Zyla:2020zbs`."""

M_K = 493.68 * MeV
r"""Mass of the charged kaon :cite:`Zyla:2020zbs`."""

M_K0 = 497.61 * MeV
r"""Mass of the neutral kaon :cite:`Zyla:2020zbs`."""

M_RHO = 762.5 * MeV
r"""Mass of the :math:`\rho` meson, obtained from the pole position :cite:`Garcia-Martin:2011nna`."""

M_RHO_STD = 1.7 * MeV  # largest error
r"""Standard deviation of the :math:`\rho` meson mass, obtained from the pole position :cite:`Garcia-Martin:2011nna`."""

G_RHO = 2 * 73.2 * MeV
r"""Width of the :math:`\rho` meson, obtained from the pole position :cite:`Garcia-Martin:2011nna`."""

G_RHO_STD = 2 * 1.1 * MeV  # largest error
r"""Standard deviation of the :math:`\rho` meson width, obtained from the pole position :cite:`Garcia-Martin:2011nna`."""

M_RHO_BW = 775.26 * MeV
r"""Mass of the :math:`\rho` meson, obtained as a Breit–Wigner parameter :cite:`Zyla:2020zbs`."""

M_RHO_BW_STD = 0.23 * MeV
r"""Standard deviation of the :math:`\rho` meson mass, obtained as a Breit–Wigner parameter :cite:`Zyla:2020zbs`."""

G_RHO_BW = 147.4 * MeV
r"""Width of the :math:`\rho` meson, obtained as a Breit–Wigner parameter :cite:`Zyla:2020zbs`."""

G_RHO_BW_STD = 0.8 * MeV
r"""Standard deviation of the :math:`\rho` meson width, obtained as a Breit–Wigner parameter :cite:`Zyla:2020zbs`."""

M_OMEGA = 782.66 * MeV
r"""Mass of the :math:`\omega` meson :cite:`Zyla:2020zbs`."""

M_OMEGA_STD = 0.13 * MeV  # largest error
r"""Standard deviation of the :math:`\omega` meson mass :cite:`Zyla:2020zbs`."""

G_OMEGA = 8.68 * MeV
r"""Width of the :math:`\omega` meson :cite:`Zyla:2020zbs`."""

G_OMEGA_STD = 0.13 * MeV  # largest error
r"""Standard deviation of the :math:`\omega` meson width :cite:`Zyla:2020zbs`."""

M_PHI = 1019.461 * MeV
r"""Mass of the :math:`\phi` meson :cite:`Zyla:2020zbs`."""

M_PHI_STD = 0.016 * MeV
r"""Standard deviation of the :math:`\phi` meson mass :cite:`Zyla:2020zbs`."""

G_PHI = 4.25 * MeV
r"""Width of the :math:`\phi` meson :cite:`Zyla:2020zbs`."""

G_PHI_STD = 0.013 * MeV
r"""Standard deviation of the :math:`\phi` meson width :cite:`Zyla:2020zbs`."""

M_ETA = 547.862 * MeV
r"""Mass of the :math:`\eta` meson :cite:`Zyla:2020zbs`."""

M_ETA_STD = 0.017 * MeV
r"""Standard deviation of the :math:`\eta` meson mass :cite:`Zyla:2020zbs`."""

G_ETA = 1.31 * keV
r"""Width of the :math:`\eta` meson :cite:`Zyla:2020zbs`."""

G_ETA_STD = 0.05 * keV
r"""Standard deviation of the :math:`\eta` meson width :cite:`Zyla:2020zbs`."""

M_KSTAR = 890 * MeV
r"""Mass of the :math:`K^*` :cite:`Zyla:2020zbs`."""

M_KSTAR_STD = 14 * MeV
r"""Standard deviation of the :math:`K^*` mass :cite:`Zyla:2020zbs`."""

G_KSTAR = 2 * 26 * MeV
r"""Width of the :math:`K^*` :cite:`Zyla:2020zbs`."""

G_KSTAR_STD = 6 * MeV
r"""Standard deviation of the :math:`K^*` width :cite:`Zyla:2020zbs`."""

M_A1_POLE = 1209 * MeV
r"""Mass of the :math:`a_1`, obtained from the pole position :cite:`JPAC:2018zwp`."""

M_A1_POLE_STD = 16 * MeV  # largest error
r"""Standard deviation of the :math:`a_1` mass, obtained from the pole position :cite:`JPAC:2018zwp`."""

G_A1_POLE = 576 * MeV
r"""Width of the :math:`a_1`, obtained from the pole position :cite:`JPAC:2018zwp`."""

G_A1_POLE_STD = 100 * MeV  # largest error
r"""Standard deviation of the :math:`a_1` width, obtained from the pole position :cite:`JPAC:2018zwp`."""


def scientific_notation(num, rounding=2):
    r"""
    Scientific notation.

    Parameters
    ----------
    num : float
        A real number.
    rounding : int
        Number of digits after the period.

    Returns
    -------
    num_not : str
        A LaTeX code for the number representation.

    Examples
    --------
    >>> import numpy as np
    >>> from DispersionRelations.constants import scientific_notation
    >>> print(scientific_notation(np.pi))
    '3.1415926536'
    >>> print(scientific_notation(13.4))
    '1.34 \\times 10^{1}'
    """
    exponent = int(np.floor(np.log10(np.absolute(num))))
    radical = round(num / (10**exponent), rounding)
    if exponent == 0:
        return str(radical)
    return str(radical) + r" \times 10^{" + str(exponent) + "}"


def rounding_PDG(mean, std):
    r"""
    Rounding with PDG rules :cite:`Zyla:2020zbs`.

    Parameters
    ----------
    mean : float
        Mean value of the quantity.
    std : float
        Standard deviation of the quantity.

    Returns
    -------
    mean_out, std_out : (float, float)
        Mean value and standard deviation, cited according to the PDG prescription.

    Examples
    --------
    >>> from DispersionRelations.constants import rounding_PDG
    >>> print(rounding_PDG(0.827, 0.119))
    (0.83, 0.12)
    >>> print(rounding_PDG(0.827, 0.367))
    (0.8, 0.4)
    """
    before_zero = int(np.floor(np.log10(np.absolute(std)))) + 1
    three_highest_digits = int(std / 10 ** (before_zero - 3))
    to_keep = 2  # default
    if 100 <= three_highest_digits <= 354:
        to_keep = 2
    if 355 <= three_highest_digits <= 949:
        to_keep = 1
    if 950 <= three_highest_digits <= 999:
        to_keep = 1  # because 1000 has 3 digits now
        three_highest_digits = 1000

    std_out = round(
        three_highest_digits * 10 ** (before_zero - 3), to_keep - before_zero
    )
    mean_out = round(mean, to_keep - before_zero)

    if mean_out % 1 == 0 and std_out % 1 == 0:
        mean_out = int(mean_out)
        std_out = int(std_out)

    return mean_out, std_out


def sR(M_R, G_R):
    r"""
    Resonance pole location calculated from its mass and width.

    Parameters
    ----------
    M_R : float
        Mass of the particle.
    G_R : float
        Width of the particle.

    Returns
    -------
    sR : complex
        The complex pole location.

    Notes
    -----
    The pole location is calculated using :math:`s_R(M_R, \Gamma_R) = (M_R - i \Gamma_R / 2)^2`.
    """
    return (M_R - 1j * G_R / 2) ** 2
