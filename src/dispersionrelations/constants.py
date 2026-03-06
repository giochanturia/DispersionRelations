r"""
Physical and mathematical constants and related functions used throughout the package.
Most masses and widths are taken by the Review of Particle Physics by the Particle Data Group :cite:`Zyla:2020zbs`.
The base unit is taken to be `GeV`.
"""

import numpy as np
from uncertainties import ufloat


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

E_almost_zero = 1e-5 * GeV
r"""A small energy, used in order to avoid division by zero."""

s_almost_zero = E_almost_zero**2

alpha_fs = 1 / 137.036
r"""Electromagnetic fine structure constant :math:`\alpha_{\text{fs}}\approx1/137`."""

e2 = 4 * np.pi * alpha_fs
r"""Electric charge squared, :math:`e^2 = 4\pi\alpha_{\text{fs}}`."""

to_nb = 1 / (2.56819 * 1e-6) * GeV**2
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

M_K = 493.677 * MeV
r"""Mass of the charged kaon :cite:`Zyla:2020zbs`."""

M_K0 = 497.611 * MeV
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

G_KSTAR_STD = 2 * 6 * MeV
r"""Standard deviation of the :math:`K^*` width :cite:`Zyla:2020zbs`."""

M_A1_POLE = 1209 * MeV
r"""Mass of the :math:`a_1`, obtained from the pole position :cite:`JPAC:2018zwp`."""

M_A1_POLE_STD = 16 * MeV  # largest error
r"""Standard deviation of the :math:`a_1` mass, obtained from the pole position :cite:`JPAC:2018zwp`."""

G_A1_POLE = 576 * MeV
r"""Width of the :math:`a_1`, obtained from the pole position :cite:`JPAC:2018zwp`."""

G_A1_POLE_STD = 100 * MeV  # largest error
r"""Standard deviation of the :math:`a_1` width, obtained from the pole position :cite:`JPAC:2018zwp`."""

BR_OMEGA_TO_3PI = 89.2 / 100
r"""Branching ratio of :math:`\omega \to 3\pi` decay :cite:`Zyla:2020zbs`."""

BR_OMEGA_TO_3PI_STD = 0.7 / 100
r"""Standard deviation of the branching ratio of :math:`\omega \to 3\pi` decay :cite:`Zyla:2020zbs`."""

__G_OMEGA = ufloat(G_OMEGA, G_OMEGA_STD)
__BR_OMEGA_TO_3PI = ufloat(BR_OMEGA_TO_3PI, BR_OMEGA_TO_3PI_STD)
__G_OMEGA_TO_3PI = __G_OMEGA * __BR_OMEGA_TO_3PI

G_OMEGA_TO_3PI = float(__G_OMEGA_TO_3PI.nominal_value)
r"""Partial width of :math:`\omega \to 3\pi` decay."""

G_OMEGA_TO_3PI_STD = float(__G_OMEGA_TO_3PI.std_dev)
r"""Standard deviation of the partial width of :math:`\omega \to 3\pi` decay."""

BR_OMEGA_TO_PI0GAMMA = 8.33 / 100
r"""Branching ratio of :math:`\omega \to \pi^0 \gamma` decay :cite:`Zyla:2020zbs`."""

BR_OMEGA_TO_PI0GAMMA_STD = 0.25 / 100
r"""Standard deviation of the branching ratio of :math:`\omega \to \pi^0 \gamma` decay :cite:`Zyla:2020zbs`.""" 

__BR_OMEGA_TO_PI0GAMMA = ufloat(BR_OMEGA_TO_PI0GAMMA, BR_OMEGA_TO_PI0GAMMA_STD)
__G_OMEGA_TO_PI0GAMMA = __G_OMEGA * __BR_OMEGA_TO_PI0GAMMA

G_OMEGA_TO_PI0GAMMA = float(__G_OMEGA_TO_PI0GAMMA.nominal_value)
r"""Partial width of :math:`\omega \to \pi^0 \gamma` decay."""

G_OMEGA_TO_PI0GAMMA_STD = float(__G_OMEGA_TO_PI0GAMMA.std_dev)
r"""Standard deviation of the partial width of :math:`\omega \to \pi^0 \gamma` decay."""

G_RHO_TO_ETAGAMMA = 44.2 * keV
r"""Partial width of :math:`\rho \to \eta \gamma` decay :cite:`Schafer:2023qtl`."""

G_RHO_TO_ETAGAMMA_STD = 3.1 * keV
r"""Standard deviation of the partial width of :math:`\rho \to \eta \gamma` decay :cite:`Schafer:2023qtl`."""

BR_KSTAR_TO_KGAMMA_CHARGED = 0.098 / 100
r"""Branching ratio of the charged :math:`K^* \to K \gamma` decay :cite:`Dax:2020dzg`."""

BR_KSTAR_TO_KGAMMA_CHARGED_STD = 0.009 / 100
r"""Standard deviation of the branching ratio of the charged :math:`K^* \to K \gamma` decay :cite:`Dax:2020dzg`."""

BR_KSTAR_TO_KGAMMA_NEUTRAL = 0.246 / 100
r"""Branching ratio of the neutral :math:`K^* \to K \gamma` decay :cite:`Dax:2020dzg`."""

BR_KSTAR_TO_KGAMMA_NEUTRAL_STD = 0.021 / 100
r"""Standard deviation of the branching ratio of the neutral :math:`K^* \to K \gamma` decay :cite:`Dax:2020dzg`."""

__G_KSTAR = ufloat(G_KSTAR, G_KSTAR_STD)
__BR_KSTAR_TO_KGAMMA_CHARGED = ufloat(BR_KSTAR_TO_KGAMMA_CHARGED, BR_KSTAR_TO_KGAMMA_CHARGED_STD)
__BR_KSTAR_TO_KGAMMA_NEUTRAL = ufloat(BR_KSTAR_TO_KGAMMA_NEUTRAL, BR_KSTAR_TO_KGAMMA_NEUTRAL_STD)
__G_KSTAR_TO_KGAMMA_CHARGED = __G_KSTAR * __BR_KSTAR_TO_KGAMMA_CHARGED
__G_KSTAR_TO_KGAMMA_NEUTRAL = __G_KSTAR * __BR_KSTAR_TO_KGAMMA_NEUTRAL

G_KSTAR_TO_KGAMMA_CHARGED = float(__G_KSTAR_TO_KGAMMA_CHARGED.nominal_value)
r"""Partial width of the charged :math:`K^* \to K \gamma` decay."""

G_KSTAR_TO_KGAMMA_CHARGED_STD = float(__G_KSTAR_TO_KGAMMA_CHARGED.std_dev)
r"""Standard deviation of the partial width of the charged :math:`K^* \to K \gamma` decay."""

G_KSTAR_TO_KGAMMA_NEUTRAL = float(__G_KSTAR_TO_KGAMMA_NEUTRAL.nominal_value)
r"""Partial width of the neutral :math:`K^* \to K \gamma` decay."""

G_KSTAR_TO_KGAMMA_NEUTRAL_STD = float(__G_KSTAR_TO_KGAMMA_NEUTRAL.std_dev)
r"""Standard deviation of the partial width of the neutral :math:`K^* \to K \gamma` decay."""