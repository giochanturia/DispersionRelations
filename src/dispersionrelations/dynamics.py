# ┌──────────────────────────────────────────────────────────┐
# │                         Typing                           │
# └──────────────────────────────────────────────────────────┘
from __future__ import annotations  # for type hints in python==3.9

from typing import Callable, Literal, Protocol
from typing_extensions import override
from numpy.typing import NDArray

# ┌──────────────────────────────────────────────────────────┐
# │                         Logging                          │
# └──────────────────────────────────────────────────────────┘
import logging

logger = logging.getLogger(__name__)

# ┌──────────────────────────────────────────────────────────┐
# │                    External imports                      │
# └──────────────────────────────────────────────────────────┘
from abc import ABC, abstractmethod
import warnings
import numpy as np
from scipy.interpolate import interp1d

# ┌──────────────────────────────────────────────────────────┐
# │                    Internal imports                      │
# └──────────────────────────────────────────────────────────┘
from dispersionrelations.complex import Complex2RS
from dispersionrelations.domains import DomainedFunctionCollection, DomainedFunction
from dispersionrelations.interpolation import ComplexInterpolator
from dispersionrelations.constants import GeV, alpha_fs, E_almost_zero
from dispersionrelations.kinematics import phase_space_twobody, kallen
from dispersionrelations.integrals import (
    # split_intervals_evenly,
    # split_intervals_logarithmically,
    # integrate_intervals_partially_gl,
    DispersionIntegralRHC,
    integrate_gl,
    disp_int_to_complex_2RS,
)

# ┌──────────────────────────────────────────────────────────┐
# │                        DYNAMICS                          │
# └──────────────────────────────────────────────────────────┘


def s_pole(
    M_R: float,
    G_R: float,
) -> complex:
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
    s_pole : complex
        The complex pole location.

    Notes
    -----
    The pole location is calculated using :math:`s_R(M_R, \Gamma_R) = (M_R - i \Gamma_R / 2)^2`.
    """
    return (M_R - 1j * G_R / 2) ** 2


# ┌──────────────────────────────────────────────────────────┐
# │                 DYNAMICS: VERTEX FUNCTIONS               │
# └──────────────────────────────────────────────────────────┘


class VertexFunction(Protocol):
    def __call__(
        self,
        s: NDArray | complex,
        m1_squared: NDArray | complex,
        m2_squared: NDArray | complex,
    ) -> NDArray | complex:
        pass


def vertex_squared_vpp(
    s: NDArray | complex,
    m1_squared: NDArray | complex,
    m2_squared: NDArray | complex,
) -> NDArray | complex:
    r"""A vector :math:`\to` pseudoscalar–pseudoscalar vertex squared.

    Parameters
    ----------
    s : array_like
        Kinematic variable (can be complex).

    m1_squared : complex
        Mass of the pseudoscalar with momentum :math:`p_1`.

    m2_squared : complex
        Mass of the pseudoscalar with momentum :math:`p_2`.

    Returns
    -------
    b : array_like
        The same shape as input `s`.

    Notes
    -----
    The vertex :math:`V(q)\to P(p_1)P(p_2)` is given by

    .. math::
        \hat{\beta}_{VPP}^{\mu}(q,p_1,p_2) \propto (p_1 - p_2)^{\mu}.

    Spin-averaged vertex function

    .. math::
        \beta_{VPP}^2(q^2, p_1^2, p_2^2) = \frac{\lambda(q^2, p_1^2, p_2^2)}{q^2} .

    The function returns :math:`\beta_{VPP}^2(s, m_{P1}^2, m_{P2}^2)/3`,
    where :math:`3=(2l+1)` corresponds to the P-wave :math:`(l=1)`.
    """
    # if m1_squared == m2_squared:
    #     return 1 / 3 * (s - 4 * m1_squared)
    return 1 / 3 * kallen(s, m1_squared, m2_squared) / s


def vertex_vpp(
    s: NDArray | complex,
    m1_squared: NDArray | complex,
    m2_squared: NDArray | complex,
) -> NDArray | complex:
    r"""A vector :math:`\to` pseudoscalar–pseudoscalar vertex.

    Parameters
    ----------
    s : array_like
        Kinematic variable (can be complex).

    m1_squared : complex
        Mass of the pseudoscalar with momentum :math:`p_1`.

    m2_squared : complex
        Mass of the pseudoscalar with momentum :math:`p_2` (if not passed, a case with equal masses will be returned).

    Returns
    -------
    b : array_like
        The same shape as input `s`.

    Notes
    -----
    This returns the square root of :func:`vertex_squared_vpp`, i.e. :math:`\frac{1}{\sqrt{3}}\beta_{VPP}(s, m_{P1}^2, m_{P2}^2)`.
    For most applications we need the squared version.
    """
    return np.sqrt(vertex_squared_vpp(s, m1_squared, m2_squared) + 0j)


def vertex_squared_vvp(
    s: NDArray | complex,
    m1_squared: NDArray | complex,
    m2_squared: NDArray | complex,
) -> NDArray | complex:
    r"""A vector :math:`\to` vector–pseudoscalar vertex squared.

    Parameters
    ----------
    s : array_like
        Kinematic variable (can be complex).

    m1_squared : float
        Mass of the vector with momentum :math:`k`.

    m2_squared : float
        Mass of the pseudoscalar with momentum :math:`p`.

    Returns
    -------
    b : array_like
        The same shape as input `s`.

    Notes
    -----
    The vertex :math:`V(q)\to V(k)P(p)` is given by

    .. math::
        \hat{\beta}_{VVP}^{(\lambda)\mu}(q, k, p) = \epsilon^{\mu\nu\alpha\beta} n^{(\lambda)}_\nu p_\alpha q_\beta .

    Spin-averaged vertex function

    .. math::
        \beta_{VVP}^2(q^2, k^2, p^2) = \frac{\lambda(q^2, k^2, p^2)}{2} .

    The function returns :math:`\beta_{VVP}^2(s, m_V^2, m_P^2)/3`,
    where :math:`3=(2l+1)` corresponds to the P-wave :math:`(l=1)`.
    """
    return 1 / 6 * kallen(s, m1_squared, m2_squared)


def vertex_vvp(
    s: NDArray | complex,
    m1_squared: NDArray | complex,
    m2_squared: NDArray | complex,
) -> NDArray | complex:
    r"""A vector :math:`\to` vector–pseudoscalar vertex.

    Parameters
    ----------
    s : array_like
        Kinematic variable (can be complex).

    m1_squared : float
        Mass of the vector with momentum :math:`k`.

    m2_squared : float
        Mass of the pseudoscalar with momentum :math:`p`.

    Returns
    -------
    b : array_like
        The same shape as input `s`.

    Notes
    -----
    This returns the square root of :func:`vertex_squared_vvp`, i.e. :math:`\frac{1}{\sqrt{3}}\beta_{VVP}(s, m_V^2, m_P^2)`.
    For most applications we need the squared version.
    """
    return np.sqrt(vertex_squared_vvp(s, m1_squared, m2_squared) + 0j)


def vertex_squared_avp(
    s: NDArray | complex,
    m1_squared: NDArray | complex,
    m2_squared: NDArray | complex,
) -> NDArray | complex:
    r"""An axial vector :math:`\to` vector–pseudoscalar vertex squared.

    Parameters
    ----------
    s : array_like
        Kinematic variable (can be complex).

    m1_squared : float
        Mass of the vector with momentum :math:`k`.

    m2_squared : float
        Mass of the pseudoscalar with momentum :math:`p`.

    Returns
    -------
    b : array_like
        The same shape as input `s`.

    Notes
    -----
    The vertex :math:`A(q)\to V(k)P(p)` is given by

    .. math::
        \hat{\beta}_{AVP}^{(\lambda)\mu}(q, k, p) = (q \cdot k) \, n^{(\lambda)\mu} - (n^{(\lambda)} \cdot q) \, k^\mu .

    Spin-averaged vertex function

    .. math::
        \beta_{AVP}^2(q^2, k^2, p^2) = 3q^2k^2 + \frac{\lambda(q^2, k^2, p^2)}{2} .

    The function returns :math:`\beta_{AVP}^2(s, m_V^2, m_P^2)/3`.
    """
    return s * m1_squared + 1 / 6 * kallen(s, m1_squared, m2_squared)


def vertex_avp(
    s: NDArray | complex,
    m1_squared: NDArray | complex,
    m2_squared: NDArray | complex,
) -> NDArray | complex:
    r"""An axial vector :math:`\to` vector–pseudoscalar vertex.

    Parameters
    ----------
    s : array_like
        Kinematic variable (can be complex).

    m1_squared : float
        Mass of the vector with momentum :math:`k`.

    m2_squared : float
        Mass of the pseudoscalar with momentum :math:`p`.

    Returns
    -------
    b : array_like
        The same shape as input `s`.

    Notes
    -----
    This returns the square root of :func:`vertex_squared_avp`, i.e. :math:`\frac{1}{\sqrt{3}}\beta_{AVP}(s, m_V^2, m_P^2)`.
    For most applications we need the squared version.
    """
    return np.sqrt(vertex_squared_avp(s, m1_squared, m2_squared) + 0j)


def vertex_squared_vap(
    s: NDArray | complex,
    m1_squared: NDArray | complex,
    m2_squared: NDArray | complex,
) -> NDArray | complex:
    r"""A vector :math:`\to` axial vector–pseudoscalar vertex squared.

    Parameters
    ----------
    s : array_like
        Kinematic variable (can be complex).

    mV : float
        Mass of the axial vector with momentum :math:`k`.

    m2_squared : float
        Mass of the pseudoscalar with momentum :math:`p`.

    Returns
    -------
    b : array_like
        The same shape as input `s`.

    Notes
    -----
    The vertex :math:`V(q)\to A(k)P(p)` is given by

    .. math::
        \hat{\beta}_{VAP}^{(\lambda)\mu}(q, k, p) = (q \cdot k) \, n^{(\lambda)\mu} - (n^{(\lambda)} \cdot q) \, k^\mu .

    Spin-averaged vertex function

    .. math::
        \beta_{VAP}^2(q^2, k^2, p^2) = 3q^2k^2 + \frac{\lambda(q^2, k^2, p^2)}{2} .

    The function returns :math:`\beta_{VAP}^2(s, m_A^2, m_P^2)/3`.
    """
    return s * m1_squared + 1 / 6 * kallen(s, m1_squared, m2_squared)


def vertex_vap(
    s: NDArray | complex,
    m1_squared: NDArray | complex,
    m2_squared: NDArray | complex,
) -> NDArray | complex:
    r"""A vector :math:`\to` axial vector–pseudoscalar vertex.

    Parameters
    ----------
    s : array_like
        Kinematic variable (can be complex).

    mV : float
        Mass of the axial vector with momentum :math:`k`.

    m2_squared : float
        Mass of the pseudoscalar with momentum :math:`p`.

    Returns
    -------
    b : array_like
        The same shape as input `s`.

    Notes
    -----
    This returns the square root of :func:`vertex_squared_vap`, i.e. :math:`\frac{1}{\sqrt{3}}\beta_{VAP}(s, m_A^2, m_P^2)`.
    For most applications we need the squared version.
    """
    return np.sqrt(vertex_squared_vap(s, m1_squared, m2_squared) + 0j)


def vertex_squared_vap_t1(
    s: NDArray | complex,
    m1_squared: NDArray | complex,
    m2_squared: NDArray | complex,
) -> NDArray | complex:
    r"""A vector :math:`\to` axial vector–pseudoscalar vertex squared.

    Parameters
    ----------
    s : array_like
        Kinematic variable (can be complex).

    mV : float
        Mass of the axial vector with momentum :math:`k`.

    m2_squared : float
        Mass of the pseudoscalar with momentum :math:`p`.

    Returns
    -------
    b : array_like
        The same shape as input `s`.
    """
    return s / 3


def vertex_squared_vap_t2(
    s: NDArray | complex,
    m1_squared: NDArray | complex,
    m2_squared: NDArray | complex,
) -> NDArray | complex:
    r"""A vector :math:`\to` axial vector–pseudoscalar vertex squared.

    Parameters
    ----------
    s : array_like
        Kinematic variable (can be complex).

    mV : float
        Mass of the axial vector with momentum :math:`k`.

    m2_squared : float
        Mass of the pseudoscalar with momentum :math:`p`.

    Returns
    -------
    b : array_like
        The same shape as input `s`.
    """
    return 1 / 6 + s * 0


# ┌──────────────────────────────────────────────────────────┐
# │                  DYNAMICS: TAMING FACTORS                │
# └──────────────────────────────────────────────────────────┘


class TamingFactor(Protocol):
    def __call__(
        self,
        s: NDArray | complex,
    ) -> NDArray | complex:
        pass


def taming_blatt_weisskopf(
    s: NDArray | complex,
    sthr: float,
    sB: float,
    l: int = 1,
    normalize_at_sB: bool = False,
) -> NDArray | complex:
    r"""
    Blatt–Weisskopf taming factors.

    Parameters
    ----------
    s : array_like
        Kinematic variable (can be complex).
    sthr : float
        Threshold of the relevant channel.
    sB : float
        Range parameter.
    l : int
        Partial wave.
    normalize_at_sB : boolean
        If set to `True`, :math:`B(s_B) = 1`.

    Returns
    -------
    B : array_like
        The same shape as input `s`.

    Raises
    ------
    NotImplementedError
        The function is only implemented for the partial waves :math:`l\in\{0,1,2,3,4\}`.
        Any other input triggers an error.

    Notes
    -----
    The first few Blatt–Weisskopf taming factors are given by :cite:`Chung:1995dx`

    .. math::

        \begin{aligned}
        B_0(s) &= 1, \\
        B_1(s) &= 1 / (x + 1), \\
        B_2(s) &= 1 / ((x-3)^2 + 9x), \\
        \end{aligned}

    where :math:`x=(s-s_{thr}) / (s_B - s_{thr})`.
    """
    if l == 0:
        # return 1
        return np.ones_like(s)
    elif l == 1:
        x = (s - sthr) / (sB - sthr)
        numerator = 2 * x if normalize_at_sB else 1
        return numerator / (1 + x)
    elif l == 2:
        x = (s - sthr) / (sB - sthr)
        numerator = 13 * x**2 if normalize_at_sB else 1
        return numerator / ((x - 3) ** 2 + 9 * x)
    # elif l == 3:  # THIS CASE SEEMS TO BE WRONG IN THE PAPER THIS IS FROM
    #     x = (s - sthr) / (sB - sthr)
    #     numerator = 277 * x**3 if normalize_at_sB else 1
    #     return numerator / (x * (x - 15) ** 2 + 9 * (2 * x - 5))
    # elif l == 4:
    #     x = (s - sthr) / (sB - sthr)
    #     numerator = 12746 * x**4 if normalize_at_sB else 1
    #     return numerator / ((x**2 - 45 * x + 105) ** 2 + 25 * x * (2 * x - 21) ** 2)
    else:
        raise NotImplementedError(
            f"Blatt-Weisskopf factor for l={l} has not been implemented."
        )


def taming_spacelike_pole(
    s: NDArray | complex,
    sB: float = (2 * GeV) ** 2,
    n: int = 1,
) -> NDArray | complex:
    r"""
    A simple decaying function, used as a taming factor.

    Parameters
    ----------
    s : array_like
        Kinematic variable (can be complex).
    sB : float
        Range parameter.
    n : int
        Degree of decay.

    Returns
    -------
    B : array_like
        The same shape as input `s`.

    Notes
    -----
    The function is defined as

    .. math:: B_n(s) = \left( \frac{s_B}{s_B + s} \right)^n .

    This creates a space-like pole of degree `n` at :math:`s = -s_B`.
    """
    return (sB / (sB + s)) ** n


def breit_wigner(
    s: NDArray | complex,
    mass: float,
    width: float,
) -> NDArray | complex:
    r"""
    Breit–Wigner distribution.

    Parameters
    ----------
    s : array_like
        Four-momentum squared of the propagating particle.
    mass : float
        Mass of the propagating particle.
    width : float
        Width of the propagating particle.

    Returns
    -------
    G : array_like
        The same shape as input `s`.

    Notes
    -----
    The distribution is defined as :cite:`Breit:1936zzb`

    .. math:: G(s, M, \Gamma) = \frac{1}{s - M^2 + i M \Gamma},

    where :math:`M` and :math:`\Gamma` are the mass and the
    width of the propagating particle, correspondingly.
    """
    return 1 / (s - mass**2 + 1j * mass * width)


def breit_wigner_energy_dependent(
    s: NDArray | complex,
    mass: float,
    width_s: Callable,
    width_0: float,
) -> NDArray | complex:
    r"""
    Breit–Wigner distribution.

    Parameters
    ----------
    s : array_like
        Four-momentum squared of the propagating particle.
    mass : float
        Mass of the propagating particle.
    width_s : callable
        Energy-dependent width of the propagating particle, must be a function of `s`, `mass`, and `width_0`.
    width_0 : float
        A parameter that enters the definition of the energy-dependent width.

    Returns
    -------
    G : array_like
        The same shape as input `s`.

    Notes
    -----
    The distribution is defined as

    .. math:: G(s, M, \Gamma_s, \Gamma_0) = \frac{1}{s - M^2 + i M \Gamma_s(s, M, \Gamma_0)},

    where :math:`M` and :math:`\Gamma_s` are the mass and the
    width of the propagating particle, correspondingly. The index :math:`s` indicates that
    the width need not be constant and can depend on energy.

    See Also
    --------
    dispersionrelations.kinematics.breit_wigner
    """
    return 1 / (s - mass**2 + 1j * mass * width_s(s, mass, width_0))


def vector_pseudoscalar_normalisation_squared(
    radiative_width: float,
    mV: float,
    mP: float,
) -> float:
    r"""Computes the vector–pseudoscalar form factor normalisation squared from the radiative width of the vector particle.

    Parameters
    ----------
    radiative_width : float
        Radiative width of the vector particle.
    mV : float
        Mass of the vector particle.
    mP : float
        Mass of the pseudoscalar particle.

    Returns
    -------
    c_squared : float
        Normalisation squared :math:`|f(0)|^2`.

    Notes
    -----

    The radiative width :math:`\Gamma` is connected to the form factor normalisation :math:`|f(0)|` via

    .. math:: \Gamma_{V \rightarrow P\gamma} = \frac{\alpha(M_V^2 - M_P^2)^3}{24 M_V^3} |f_{VP}(0)|^2.

    """
    return radiative_width / ((alpha_fs / 24) * ((mV**2 - mP**2) / mV) ** 3)


def vector_pseudoscalar_normalisation_squared_error(
    radiative_width: float,
    mV: float,
    mP: float,
    radiative_width_std: float,
    mV_std: float,
    mP_std: float,
) -> float:
    r"""Computes the error of the vector–pseudoscalar form factor normalisation squared from the radiative width of the vector particle.

    Parameters
    ----------
    radiative_width, radiative_width_std : float
        Radiative width of the vector particle (and the corresponding error).
    mV, mV_std : float
        Mass of the vector particle (and the corresponding error).
    mP, mP_std : float
        Mass of the pseudoscalar particle (and the corresponding error).
    Returns
    -------
    c_squared_std : float
        Error of the normalisation squared :math:`|f(0)|^2`.

    Notes
    -----

    The radiative width :math:`\Gamma` is connected to the form factor normalisation :math:`|f(0)|` via

    .. math:: \Gamma_{V \rightarrow P\gamma} = \frac{\alpha(M_V^2 - M_P^2)^3}{24 M_V^3} |f_{VP}(0)|^2.

    """
    jacobian_width = 1 / ((alpha_fs / 24) * ((mV**2 - mP**2) / mV) ** 3)
    jacobian_mV = (
        3
        * radiative_width
        * mV**2
        * (mP**2 + mV**2)
        / (alpha_fs * (mV**2 - mP**2) ** 4)
    )
    jacobian_mP = 6 * radiative_width * mP * mV**3 / (alpha_fs * (mV**2 - mP**2) ** 4)
    return np.sqrt(
        (jacobian_width * radiative_width_std) ** 2
        + (jacobian_mV * mV_std) ** 2
        + (jacobian_mP * mP_std) ** 2
    )


class OmnesFunction(DispersionIntegralRHC):
    r"""
    Omnès–Muskhelishvili solution :cite:`Omnes:1958hv`.

    Parameters
    ----------
    integrand : Callable
        Integrand along the RHC.
    threshold : float
        Lower boundary of the integral.
    infinity : float, optional
        Upper boundary of the integral in the units of :code:`threshold`.
    integration_split_points : array_like, optional
        Integration split points in the units of :code:`threshold`.
    integration_order : int, optional
        Gauss–Legendre quadrature order.
    subtraction_point : float, optional
        Point :math:`s_0` of subtraction.
    subtraction_constants : array_like, optional
        Array of :math:`f_i` subtraction constants. Defines subtraction level :math:`n`.

    Notes
    -----
    The function is defined as

    .. math:: \Omega(s) = \exp\left(\sum_{i=0}^{n-1}f_i (s-s_0)^i + \frac{(s-s_0)^n}{\pi}\int_{s_{thr}}^{\infty}\frac{\delta(x) \, dx}{(x-s_0)^n(x-s-i\epsilon)}\right),

    where :math:`s_0` is the subtraction point, :math:`f_i` are the subtraction constants for :math:`\log\Omega`,
    :math:`n` is the subtraction level, and
    :math:`\delta` is the input phase, passed as `integrand`.

    See Also
    --------
    dispersionrelations.integrals.DispersionIntegralRHC
    """

    @override
    def __call__(
        self,
        s: NDArray | complex,
    ) -> NDArray | complex:
        r"""
        Parameters
        ----------
        s : array_like
            Four-momentum squared of the propagating particle.

        Returns
        -------
        Ω : array_like
            The same shape as input `s`.
        """
        return np.exp(super().__call__(s))


# TODO refactor and drop the inheritance from DispersionIntegralRHC
class Channel(DispersionIntegralRHC, ABC):
    r"""
    Dispersive representation of a channel.

    Parameters
    ----------
    loop_integrand : callable
        Imaginary part of the self-energy.
    threshold : float
        Lower boundary of the integral.
    infinity : float, optional
        Upper boundary of the integral in the units of :code:`threshold`.
    integration_split_points : array_like, optional
        Integration split points in the units of :code:`threshold`.
    integration_order : int, optional
        Gauss–Legendre quadrature order.
    subtraction_point : float, optional
        Point :math:`s_0` of subtraction.
    subtraction_constants : array_like, optional
        Array of :math:`f_i` subtraction constants. Defines subtraction level :math:`n`.

    Notes
    -----
    The integral is defined as

    .. math:: \Pi(s) = \sum_{i=0}^{n-1}f_i (s-s_0)^i + \frac{(s-s_0)^n}{\pi}\int_{s_{thr}}^{\infty}\frac{\mathrm{Im}(\Pi(x)) \, dx}{(x-s_0)^n(x-s-i\epsilon)},

    where :math:`s_0` is the subtraction point, :math:`f_i` are the subtraction constants,
    :math:`n` is the subtraction level, and
    :math:`\mathrm{Im}(\Pi(x))` is the `integrand`.

    See also
    --------

    dispersionrelations.dynamics.StableTwoBodyChannel

    """

    def __init__(
        self,
        loop_integrand: callable,
        threshold: float,
        infinity: float = 1e6,
        integration_split_points: list[float] = [
            2,
            3,
            5,
            8,
            10,
            50,
            1e2,
            1e3,
            1e4,
            1e5,
        ],
        integration_order: int = 100,
        subtraction_point: float = 0,
        subtraction_constants: list[float] = [0],
    ):
        super().__init__(
            loop_integrand,
            threshold,
            infinity,
            integration_split_points,
            integration_order,
            subtraction_point,
            subtraction_constants,
        )

    def loop_discontinuity(
        self,
        s: NDArray,
    ) -> NDArray:
        r"""
        Discontinuity of the two-body channel self-energy.

        Parameters
        ----------
        s : array_like
            Four-momentum squared of the two-body state.

        Returns
        -------
        DiscΠ : array_like
            The same shape as input `s`.

        Notes
        -----

        We assume the schwarz_reflection reflection principle and therefore,

        .. math::
            \mathrm{Disc}(\Pi(s)) = 2i \, \mathrm{Im}(\Pi(s)) \, .

        """
        result_up = np.where(np.imag(s) >= 0, 2j * self.integrand(s), 0)
        result_down = np.where(
            np.imag(s) < 0, np.conj(2j * self.integrand(np.conj(s))), 0
        )
        return result_up + result_down

    def loop(
        self,
        s: NDArray,
        sheet: Literal[1, 2] = 1,
    ) -> NDArray:
        r"""
        Invokes the :code:`__call__` method of the parent class,
        with an added functionality of switching to the second sheet.
        Computed the self-energy function of the two-body channel.

        Parameters
        ----------
        s : array_like
            Four-momentum squared of the two-body state.
        sheet : int, optional
            Riemann sheet number (can only be 1 or 2, other inputs return the first sheet).

        Returns
        -------
        Π : array_like
            The same shape as input `s`.
        """
        result = self.__call__(s)
        if sheet == 2:
            result -= self.loop_discontinuity(s)
        return result

    @abstractmethod
    def phase_space_function(self, s: NDArray) -> NDArray:
        return None

    @abstractmethod
    def vertex_function_squared(self, s: NDArray) -> NDArray:
        return None

    @abstractmethod
    def taming_factor_squared(self, s: NDArray) -> NDArray:
        return None

    def vertex_function(self, s: NDArray) -> NDArray:
        return np.sqrt(self.vertex_function_squared(s) + 0j)

    def taming_factor(self, s: NDArray) -> NDArray:
        return np.sqrt(self.taming_factor_squared(s) + 0j)


# TODO this class will be redundant after interpolated complex functions are finalized
class PrecomputedChannel(Channel):
    def __init__(self, s_values, integrand_values, threshold, **kwargs):
        self.last_s = s_values[-1]
        self.last_integrand = integrand_values[-1]
        self.interpolated_integrand = interp1d(s_values, integrand_values)
        warnings.warn(
            "You are loading this channel from precomputed values. Full discontinuity will be absorbed in the phase space function."
        )
        super().__init__(self.extrapolated_integrand, threshold, **kwargs)

    def extrapolated_integrand(self, s):
        s = np.asarray(s)
        res = np.zeros_like(s)
        mask1 = s < self.threshold
        mask2 = s > self.last_s
        mask3 = ~(mask1 | mask2)
        res[mask1] = 0
        res[mask2] = self.last_integrand
        res[mask3] = self.interpolated_integrand(s[mask3])
        return res

    def phase_space_function(self, s):
        return self.integrand(s)

    def vertex_function_squared(self, s):
        return 0 * s + 1

    def taming_factor_squared(self, s):
        return 0 * s + 1


# TODO this class will be redundant after interpolated complex functions are finalized
class InterpolatedChannel(Channel):
    def __init__(
        self,
        channel: Channel,
        loop_1: ComplexInterpolator,
        loop_2: ComplexInterpolator,
    ):
        self._channel = channel
        self.threshold = self._channel.threshold
        self.integrand = self._channel.integrand
        self._loop_1 = loop_1
        self._loop_2 = loop_2

    def __getattr__(self, name):
        return getattr(self._channel, name)

    def loop_from_interpolation(self, s, sheet=1):
        if sheet == 1:
            return self._loop_1(s)
        if sheet == 2:
            return self._loop_2(s)
        raise ValueError(f"Invalid sheet number: {sheet}. Only 1 and 2 are allowed.")

    @override
    def loop(self, s, sheet=1, fallback_to_channel_loop=True):
        result = self.loop_from_interpolation(s, sheet)
        if fallback_to_channel_loop and np.any(np.isnan(result)):
            warnings.warn(
                f"Interpolation returned NaN for {np.sum(np.isnan(result))} values. Falling back to the original channel loop for those values."
            )
            mask = np.isnan(result)
            result[mask] = self._channel.loop(s[mask], sheet=sheet)
        return result

    @override
    def loop_discontinuity(self, s):
        return self.loop(s, sheet=1) - self.loop(s, sheet=2)

    def phase_space_function(self, s):
        return self._channel.phase_space_function(s)

    def vertex_function_squared(self, s):
        return self._channel.vertex_function_squared(s)

    def taming_factor_squared(self, s):
        return self._channel.taming_factor_squared(s)


class StableTwoBodyChannel(Channel):
    r"""
    Constructs a two-body channel with stable particles with masses :code:`m1` and :code:`m2`.

    Parameters
    ----------
    m1 : float
        Mass of the first particle.
    m2 : float
        Mass of the second particle.
    vertex_function_squared : callable
        Vertex function squared. Depends on the interaction of the particles.
    taming_factor : callable
        Taming factor, to make sure that the integrand does not grow indefinitely.
    **kwargs
        Other keyword arguments, to be passed to the constructor of the parent class.

    Notes
    -----

    The integrand of the loop is constructed via

    .. math::
        \mathrm{Disc}(\Pi(s)) = 2i \, \rho(s, m_1^2, m_2^2) \beta^2(s, m_1^2, m_2^2) B^2(s) \, ,

    where :math:`\rho` is the phase space, :math:`\beta` is the vertex function,
    and :math:`B` is the taming factor.
    """

    def __init__(
        self,
        m1,
        m2,
        vertex_function_squared=vertex_squared_vpp,
        taming_factor_squared=lambda s: taming_spacelike_pole(s) ** 2,
        **kwargs,
    ):
        # Store parameters:
        self.m1 = m1
        self.m2 = m2
        self.threshold = (m1 + m2) ** 2
        # Implement abstract methods:
        self.__psf__ = np.vectorize(lambda s: phase_space_twobody(s, m1, m2))
        self.__vfs__ = np.vectorize(lambda s: vertex_function_squared(s, m1, m2))
        self.__tfs__ = np.vectorize(lambda s: taming_factor_squared(s))
        # Construct the integrand and call the parent constructor:
        loop_integrand = lambda s: (
            self.phase_space_function(s)
            * self.vertex_function_squared(s)
            * self.taming_factor_squared(s)
        )
        super().__init__(loop_integrand, self.threshold, **kwargs)

    def phase_space_function(self, s):
        return self.__psf__(s)

    def vertex_function_squared(self, s):
        return self.__vfs__(s)

    def taming_factor_squared(self, s):
        return self.__tfs__(s)


class UnstableParticle:
    r"""
    Unstable particle.

    Parameters
    ----------
    decay_channel : TwoBodyChannel
        A channel that the particle decays into.
        The channel does not need to be two-body,
        as long as it has a :code:`threshold` and
        functions :code:`integrand` and :code:`loop`.
    pole_location : complex
        Location of the resonance pole (see :func:`dispersionrelations.constants.s_pole`).
    pole_sheet : int
        Riemann sheet of the decay channel, on which the resonance lives.
        For above-threshold resonances, the typical sheet is `2`.
    mass_and_coupling : (float, float)
        Bare mass and the coupling constant, parameters of the propagator.
        If passed, :code:`pole_location` is ignored.
    interpolation_points : list
        Intervals to use for interpolating the spectral function.
    interpolation_density : int
        Number of points per interpolation interval.
    """

    def __init__(
        self,
        decay_channel: Channel,
        pole_location: complex,
        pole_sheet: int = 2,
        mass_and_coupling: tuple[float, float] | None = None,
    ):
        self.decay_channel = decay_channel
        self.sp = pole_location
        self.pole_sheet = pole_sheet
        self.mass_and_coupling = mass_and_coupling
        self.__calculate_m_g__()

    def __calculate_m_g__(self):
        if self.mass_and_coupling is not None:
            self.m, self.g = self.mass_and_coupling
            return

        # The code below is the only part where self.decay_channel.loop should be known for complex numbers.
        # If mass_and_coupling is provided, real-valued loop is sufficient.
        loop_at_the_pole = self.decay_channel.loop(self.sp, sheet=self.pole_sheet)
        self.g = np.sqrt(-np.imag(self.sp) / np.imag(loop_at_the_pole))
        self.m = np.sqrt(np.real(self.sp) + self.g**2 * np.real(loop_at_the_pole))

    def propagator(
        self,
        s: NDArray,
        sheet: Literal[1, 2] = 1,
    ) -> NDArray:
        r"""
        Propagator of the unstable particle.

        Parameters
        ----------

        s : array_like
            Four-momentum squared of the propagating particle.
        sheet : int, optional
            Riemann sheet number of the decay channel self-energy
            (can only be 1 or 2, other inputs return the first sheet).

        Returns
        -------
        G : array_like
            The same shape as input `s`.

        Notes
        -----

        The propagator is defined as

        .. math:: G(s) = \frac{1}{s - m^2 + g^2 \Pi(s)} \, ,

        where :math:`\Pi(s)` is the self-energy of the decay channel.
        """
        return 1 / (s - self.m**2 + self.g**2 * self.decay_channel.loop(s, sheet))

    def spectral_function_CP(
        self,
        s: NDArray,
    ) -> NDArray:
        r"""
        Spectral function of the unstable particle for :math:`s\in\mathbb{C}`.

        Parameters
        ----------

        s : array_like
            Four-momentum squared of the propagating particle.

        Returns
        -------
        σ : array_like
            The same shape as input `s`.

        Notes
        -----

        The spectral function is defined as

        .. math:: \sigma(s) = \frac{1}{\pi} G^{I}(s) G^{II}(s) \, g^2 \mathrm{Im}(\Pi(s)) \, ,

        which, for real :math:`s` reduces to

        .. math:: \sigma(s\in\mathbb{R}) = -\frac{1}{\pi} \mathrm{Im}(G(s)) \, .
        """
        return (
            1
            / (2j * np.pi)
            * self.propagator(s, sheet=1)
            * self.propagator(s, sheet=2)
            * self.g**2
            * self.decay_channel.loop_discontinuity(s)
        )

    def spectral_function_RE(
        self,
        s: NDArray,
    ) -> NDArray:
        r"""
        Spectral function of the unstable particle for :math:`s\in\mathbb{R}`.

        Parameters
        ----------

        s : array_like
            Four-momentum squared of the propagating particle.

        Returns
        -------
        σ : array_like
            The same shape as input `s`.

        Notes
        -----

        The spectral function is defined as

        .. math:: \sigma(s) = -\frac{1}{\pi} \mathrm{Im}(G(s)) \, .
        """
        return (
            1
            / np.pi
            * np.abs(self.propagator(s)) ** 2
            * self.g**2
            * self.decay_channel.integrand(s)
        )


# TODO this class will be redundant after interpolated complex functions are finalized
class UnstableParticleInterpolated(UnstableParticle):
    r"""
    Unstable particle.

    Parameters
    ----------
    decay_channel : TwoBodyChannel
        A channel that the particle decays into.
        The channel does not need to be two-body,
        as long as it has a :code:`threshold` and
        functions :code:`integrand` and :code:`loop`.
    pole_location : complex
        Location of the resonance pole (see :func:`dispersionrelations.constants.s_pole`).
    pole_sheet : int
        Riemann sheet of the decay channel, on which the resonance lives.
        For above-threshold resonances, the typical sheet is `2`.
    mass_and_coupling : (float, float)
        Bare mass and the coupling constant, parameters of the propagator.
        If passed, :code:`pole_location` is ignored.
    interpolation_points : list
        Intervals to use for interpolating the spectral function.
    interpolation_density : int
        Number of points per interpolation interval.
    """

    def __init__(
        self,
        decay_channel: Channel,
        pole_location: complex,
        pole_sheet: int = 2,
        mass_and_coupling: tuple | None = None,
        interpolation_points=[1, 2, 5, 1e1, 5e1, 1e2, 1e3, 1e5, 1e8],
        interpolation_density=1000,
    ):
        super().__init__(decay_channel, pole_location, pole_sheet, mass_and_coupling)
        self.interpolation_points = interpolation_points
        self.interpolation_density = interpolation_density
        self.__interpolate_spectral_function__()

    def __interpolate_spectral_function__(self):
        s_interpolation = np.concatenate(
            [
                self.decay_channel.threshold
                * np.linspace(
                    1.0001 * self.interpolation_points[i],
                    self.interpolation_points[i + 1],
                    self.interpolation_density,
                )
                for i in range(len(self.interpolation_points) - 1)
            ]
        )
        sf_interpolation = np.real(self.spectral_function_CP(s_interpolation))
        self.__spectral_function_interpolated__ = interp1d(
            [self.decay_channel.threshold, *s_interpolation], [0, *sf_interpolation]
        )

    @override
    def spectral_function_RE(
        self,
        s: NDArray,
    ) -> NDArray:
        r"""
        Spectral function of the unstable particle for :math:`s\in\mathbb{R}`.

        Parameters
        ----------

        s : array_like
            Four-momentum squared of the propagating particle.

        Returns
        -------
        σ : array_like
            The same shape as input `s`.

        Notes
        -----

        The spectral function is defined as

        .. math:: \sigma(s) = -\frac{1}{\pi} \mathrm{Im}(G(s)) \, .

        An interpolation function is used when possible.
        """
        below_threshold = np.real(s) < self.decay_channel.threshold
        above_interpolation_maximum = (
            np.real(s) > self.interpolation_points[-1] * self.decay_channel.threshold
        )
        interpolatable = ~below_threshold & ~above_interpolation_maximum
        result = np.zeros_like(s, dtype=float)
        if np.sum(interpolatable) > 0:
            result[interpolatable] = self.__spectral_function_interpolated__(
                s[interpolatable]
            )
        return result


# TODO interpolation should not be the responsibility of this class
class SemiStableTwoBodyCut:
    r"""
    Two-body channel with one unstable particle and one spectator.

    Parameters
    ----------
    unstable_particle : UnstableParticle
        Unstable particle.
    spectator_mass : float
        Mass of the spectator particle.
    vertex_function : callable
        Vertex function. Depends on the interaction of the particles.
    phase_space : callable
        Phase space function.
    taming_factor : callable
        Taming factor, to make sure that the integrand does not grow indefinitely.
    gl_order : int
        Gauss–Legendre quadrature order.
    split_n : int
        Number of intervals per numerical integration.
    interpolation_points : list
        Intervals to use for interpolating the spectral function.
    interpolation_density: int
        Number of points per interpolation interval.
    """

    def __init__(
        self,
        unstable_particle: UnstableParticle,
        spectator_mass: float,
        vertex_function_squared: callable,
        gl_order: int = 100,
        split_n: int = 3,
        integration_split_points: list[float] = [2, 3, 5, 10, 50, 100],
        interpolation_points: list = [
            1,
            2,
            3,
            4,
            5,
            1e1,
            3e1,
            5e1,
            1e2,
            1e3,
            1e4,
            1e5,
            1e6,
            1e7,
            1e8,
            1e9,
        ],
        interpolation_density: int = 1000,
    ):
        self.unstable_particle = unstable_particle
        self.spectator_mass = spectator_mass
        self.phase_space_factor = phase_space_twobody
        self.vertex_function_squared = vertex_function_squared
        self.gl_order = gl_order
        self.split_n = split_n
        self.integration_split_points = (
            self.unstable_particle.decay_channel.threshold
            * np.array(integration_split_points)
        )
        self.interpolation_points = interpolation_points
        self.interpolation_density = interpolation_density
        self.__interpolate_ImPI_integral__()

    def ImPI_integrand_RE(
        self,
        s: NDArray,
        x: NDArray,
    ) -> NDArray:
        r"""
        Integrand for the :math:`\mathrm{Im}(\Pi(s))` spectral integral for :math:`x\in\mathbb{R}`.

        Parameters
        ----------
        s : complex
            Four-momentum squared of the two-body system.
        x : array_like
            Four-momentum squared of the unstable particle.

        Returns
        -------
        f : array_like
            The same shape as input `x`.

        Notes
        -----
        The integrand is defined via

        .. math:: \mathrm{Im}(\Pi(s)) = \int_{s_\text{thr}}^{(\sqrt{s}-m)^2} \sigma(x) \rho(s, x, m^2) \beta^2(s, x, m^2) B^2(s) \, ,

        where :math:`\sigma` is the spectral function of the unstable particle
        and :math:`s_\text{thr}` is the threshold of the corresponding decay channel.
        """
        return (
            self.unstable_particle.spectral_function_RE(x)
            * self.phase_space_factor(s, np.sqrt(x + 0j), self.spectator_mass)
            * self.vertex_function_squared(s, np.sqrt(x + 0j), self.spectator_mass)
        )

    def ImPI_integrand_CP(
        self,
        s: NDArray,
        x: NDArray,
    ) -> NDArray:
        r"""
        Integrand for the :math:`\mathrm{Im}(\Pi(s))` spectral integral for :math:`x\in\mathbb{C}`.

        Parameters
        ----------
        s : complex
            Four-momentum squared of the two-body system.
        x : array_like
            Four-momentum squared of the unstable particle.

        Returns
        -------
        f : array_like
            The same shape as input `x`.

        Notes
        -----
        The integrand is defined via

        .. math:: \mathrm{Im}(\Pi(s)) = \int_{s_\text{thr}}^{(\sqrt{s}-m)^2} \sigma(x) \rho(s, x, m^2) \beta^2(s, x, m^2) B^2(s) \, ,

        where :math:`\sigma` is the spectral function of the unstable particle
        and :math:`s_\text{thr}` is the threshold of the corresponding decay channel.
        """
        return (
            self.unstable_particle.spectral_function_CP(x)
            * self.phase_space_factor(s, np.sqrt(x + 0j), self.spectator_mass)
            * self.vertex_function_squared(s, np.sqrt(x + 0j), self.spectator_mass)
        )

    def ImPI_integral(
        self,
        s: NDArray,
    ) -> NDArray:
        r"""
        The :math:`\mathrm{Im}(\Pi(s))` spectral integral.

        Parameters
        ----------
        s : complex
            Four-momentum squared of the two-body system.

        Returns
        -------
        ImΠ : complex
            A single complex number.

        Notes
        -----
        The integral is defined via

        .. math:: \mathrm{Im}(\Pi(s)) = \int_{s_\text{thr}}^{(\sqrt{s}-m)^2} \sigma(x) \rho(s, x, m^2) \beta^2(s, x, m^2) B^2(s) \, ,

        where :math:`\sigma` is the spectral function of the unstable particle
        and :math:`s_\text{thr}` is the threshold of the corresponding decay channel.

        For complex values of :math:`s`, the integral contour is taken to be
        rectangular (see e.g. :cite:`JPAC:2018zwp` for more details).
        """
        lower_bound = self.unstable_particle.decay_channel.threshold
        upper_bound = (np.sqrt(s) - self.spectator_mass) ** 2
        middle_bound = np.real(upper_bound)

        integrand = lambda x: self.ImPI_integrand_RE(s, x)
        if np.imag(upper_bound) != 0:
            integrand = lambda x: self.ImPI_integrand_CP(s, x)

        integration_points_1 = np.concatenate(
            (
                [lower_bound],
                self.integration_split_points[
                    self.integration_split_points < middle_bound
                ],
                [middle_bound],
            )
        )

        integral_1 = 0 * s + 0j
        for i in range(len(integration_points_1) - 1):
            integral_1 += integrate_gl(
                integrand,
                integration_points_1[i],
                integration_points_1[i + 1],
                order=self.gl_order,
                split_n=self.split_n,
            )
        # integral_1 = integrate_gl(
        #     integrand,
        #     lower_bound,
        #     middle_bound,
        #     order=self.gl_order,
        #     split_n=self.split_n,
        # )

        if np.imag(upper_bound) == 0:
            return integral_1

        integral_2 = integrate_gl(
            integrand,
            middle_bound,
            upper_bound,
            order=self.gl_order,
            split_n=self.split_n,
        )

        return integral_1 + integral_2

    @property
    def ImPI_integral_vectorized(
        self,
    ) -> None:
        r"""Vectorized version of :func:`dispersionrelations.dynamics.SemiStableTwoBodyCut.ImPI_integral`"""
        return np.vectorize(self.ImPI_integral)

    def __interpolate_ImPI_integral__(
        self,
    ) -> None:
        s_interpolation = np.concatenate(
            [
                self.unstable_particle.decay_channel.threshold
                * np.linspace(
                    1.0001 * self.interpolation_points[i],
                    self.interpolation_points[i + 1],
                    self.interpolation_density,
                )
                for i in range(len(self.interpolation_points) - 1)
            ]
        )
        ImPI_interpolation = np.array(
            [np.real(self.ImPI_integral(s)) for s in s_interpolation]
        )
        self.__ImPI_integral_interpolated__ = interp1d(
            [self.unstable_particle.decay_channel.threshold, *s_interpolation],
            [0, *ImPI_interpolation],
        )

    def ImPI_integral_RE(
        self,
        s: NDArray,
    ) -> NDArray:
        r"""
        The :math:`\mathrm{Im}(\Pi(s))` spectral integral for :math:`s\in\mathbb{R}`.

        Parameters
        ----------
        s : array_like
            Four-momentum squared of the two-body system.

        Returns
        -------
        ImΠ : array_like
            The same shape as input `s`.

        Notes
        -----
        An interpolation function is used when possible.
        """
        below_threshold = np.real(s) < self.unstable_particle.decay_channel.threshold
        above_interpolation_maximum = (
            np.real(s)
            > self.interpolation_points[-1]
            * self.unstable_particle.decay_channel.threshold
        )
        interpolatable = ~below_threshold & ~above_interpolation_maximum
        result = np.zeros_like(s, dtype=float)
        if np.sum(interpolatable) > 0:
            result[interpolatable] = self.__ImPI_integral_interpolated__(
                s[interpolatable]
            )
        if np.sum(above_interpolation_maximum) > 0:
            result[above_interpolation_maximum] = self.ImPI_integral_vectorized(
                s[above_interpolation_maximum]
            )
        return result

    def ImPI_integral_CP(
        self,
        s: NDArray,
    ) -> NDArray:
        r"""
        The :math:`\mathrm{Im}(\Pi(s))` spectral integral for :math:`s\in\mathbb{C}`.

        Parameters
        ----------
        s : array_like
            Four-momentum squared of the two-body system.

        Returns
        -------
        ImΠ : array_like
            The same shape as input `s`.
        """
        return self.ImPI_integral_vectorized(s)

    def __call__(
        self,
        s: NDArray,
    ) -> NDArray:
        r"""
        The :math:`\mathrm{Im}(\Pi(s))` spectral integral for a generic :math:`s`.

        Parameters
        ----------
        s : array_like
            Four-momentum squared of the two-body system.

        Returns
        -------
        ImΠ : array_like
            The same shape as input `s`.

        Notes
        -----
        An interpolation function is used when possible for real values of :math:`s`.
        """
        s = np.array(s)
        is_complex = np.imag(s) != 0
        result = np.zeros(s.shape, dtype=np.complex128)
        if np.sum(is_complex) > 0:
            result[is_complex] = self.ImPI_integral_CP(s[is_complex])
        if np.sum(~is_complex) > 0:
            result[~is_complex] = self.ImPI_integral_RE(np.real(s[~is_complex]))
        return result


class SemiStableTwoBodyChannel(Channel):
    def __init__(
        self,
        twobodycut: SemiStableTwoBodyCut,
        taming_factor_squared=lambda s: taming_spacelike_pole(s) ** 2,
    ):
        # Store parameters:
        threshold = (
            np.sqrt(twobodycut.unstable_particle.decay_channel.threshold)
            + twobodycut.spectator_mass
        ) ** 2
        # Implement abstract methods:
        self.__psf__ = twobodycut.__call__
        self.__vfs__ = np.vectorize(lambda s: 1)
        self.__tfs__ = np.vectorize(taming_factor_squared)
        # Construct the integrand and call the parent constructor:
        loop_integrand = lambda s: (
            self.phase_space_function(s)
            * self.vertex_function_squared(s)
            * self.taming_factor_squared(s)
        )
        super().__init__(loop_integrand, threshold)

    def phase_space_function(self, s):
        return self.__psf__(s)

    def vertex_function_squared(self, s):
        return self.__vfs__(s)

    def taming_factor_squared(self, s):
        return self.__tfs__(s)


class TwoPotentialModel:
    def __init__(
        self,
        channels: dict[str, Channel],
        bg_par: dict,
        res_par: dict,
        channel_sheets: dict = {},
        s_array: np.ndarray | list = np.linspace(E_almost_zero, 3 * GeV, 200) ** 2,
    ):
        # Initialize channels
        self.channels = channels
        self.channel_sheets = {key: 1 for key in channels.keys()}
        for key in channel_sheets.keys():
            if key in channels.keys():
                self.channel_sheets[key] = channel_sheets[key]
            else:
                warnings.warn(
                    f"Channel '{key}' not found in the provided channels dictionary. Ignoring the sheet specification."
                )
        # Initialize parameters
        self.bg_par = bg_par
        self.res_par = res_par
        # Initialize auxilary variables
        self.nC = len(channels)
        self.nR = len(res_par["m"])
        self.IdC = np.identity(self.nC)
        self.IdR = np.identity(self.nR)

        # Initialize kinematic variables
        self.__set_s_array__(s_array)

        # Default parameters, in case none are provided
        self.bg_par = {
            "f0": np.zeros((self.nC, self.nC)),
            "fR": np.array([0.0]),
        }
        self.res_par = {
            "g": np.zeros((self.nR, self.nC)),
            "m": np.arange(self.nR) + 1 * GeV,
            "c": np.ones(self.nC),
            "a": np.ones(self.nR),
        }

        # Store model parameters from passed dictionaries
        self.store_parameters(bg_par, res_par)

        # Computation steps
        self.precompute()
        self.compute_bg()
        self.compute_res()

    def __set_s_array__(self, s_array):
        # Index _rs stands for "reshaped"
        self.s_array = np.array(s_array)
        self.s_rs = self.s_array[..., np.newaxis, np.newaxis]
        self.E_array = np.sign(s_array) * np.sqrt(np.absolute(self.s_array))
        self.E_rs = self.E_array[..., np.newaxis, np.newaxis]

    def store_parameters(self, new_bg_par=None, new_res_par=None):
        if new_bg_par is not None:
            for key in new_bg_par:
                if key in self.bg_par.keys():
                    self.bg_par[key] = new_bg_par[key]
                else:
                    warnings.warn(
                        f"Background parameter '{key}' not recognized. Ignoring."
                    )

            self.f0 = np.array(self.bg_par["f0"])
            self.fR = np.array(self.bg_par["fR"])

        if new_res_par is not None:
            for key in new_res_par:
                if key in self.res_par.keys():
                    self.res_par[key] = new_res_par[key]
                else:
                    warnings.warn(
                        f"Resonance parameter '{key}' not recognized. Ignoring."
                    )

            self.g = np.array(self.res_par["g"])
            self.m = np.array(self.res_par["m"])
            self.c = np.array(self.res_par["c"])
            self.a = np.array(self.res_par["a"])

    def __diag_in_channel_space__(self, arr):
        arr = np.asarray(arr)
        if arr.shape != (*self.s_array.shape, self.nC):
            raise ValueError(
                f"Input array must have shape {(*self.s_array.shape, self.nC)}"
            )

        channel_indexing = np.arange(self.nC)
        result = np.zeros((*self.s_array.shape, self.nC, self.nC)) + 0j
        result[..., channel_indexing, channel_indexing] = arr
        return result

    def __diag_in_resonance_space__(self, arr):
        arr = np.asarray(arr)
        if arr.shape != (*self.s_array.shape, self.nR):
            raise ValueError(
                f"Input array must have shape {(*self.s_array.shape, self.nR)}"
            )

        resonance_indexing = np.arange(self.nR)
        result = np.zeros((*self.s_array.shape, self.nR, self.nR)) + 0j
        result[..., resonance_indexing, resonance_indexing] = arr
        return result

    def precompute(self):
        loops_arr = np.array(
            [
                self.channels[key].loop(self.s_array, sheet=self.channel_sheets[key])
                for key in self.channels.keys()
            ]
        )
        self.loops = self.__diag_in_channel_space__(np.moveaxis(loops_arr, 0, -1))

    def compute_bg(self, new_bg_par=None):
        if new_bg_par is not None:
            self.store_parameters(new_bg_par=new_bg_par)

        self.f_s = (
            np.ones((*self.s_array.shape, self.nC, self.nC)) + self.fR * self.s_rs
        )
        self.VB = self.f0 / self.f_s
        self.TB = np.linalg.solve(self.IdC - self.VB @ self.loops, self.VB)
        self.vertex = self.IdC + self.TB @ self.loops
        self.loops_with_bg = self.loops @ self.vertex

    def compute_res(self, new_res_par=None):
        if new_res_par is not None:
            self.store_parameters(new_res_par=new_res_par)

        self.GR_inv = self.__diag_in_resonance_space__(
            self.s_array[..., np.newaxis] - self.m[np.newaxis, ...] ** 2
        )
        self.GR_renormalized_inv = self.GR_inv + self.g @ self.loops_with_bg @ self.g.T
        self.tR = -self.g.T @ np.linalg.solve(
            self.GR_renormalized_inv, self.g
        )  # TODO: add a warning for failed inversion
        self.TR = self.vertex @ self.tR @ (np.swapaxes(self.vertex, -2, -1))
        self.F = np.einsum(
            "...ij,...j->...i",
            -self.vertex @ self.g.T,
            np.linalg.solve(
                self.GR_renormalized_inv, self.a
            ),  # TODO: add a warning for failed inversion
        )

    def compute_everything(self, bg_par=None, res_par=None):
        self.compute_bg(bg_par)
        self.compute_res(res_par)

    def new_s_array(self, s_array):
        self.__set_s_array__(s_array)
        self.precompute()
        self.compute_bg()
        self.compute_res()

    def __amplitude__(self, amp):
        vertex_factors = self.__diag_in_channel_space__(
            np.moveaxis(
                np.array(
                    [
                        channel.vertex_function(self.s_array)
                        * channel.taming_factor(self.s_array)
                        for channel in self.channels.values()
                    ]
                ),
                0,
                -1,
            )
        )
        return vertex_factors @ amp @ vertex_factors

    def T_matrix_bg(self):
        return self.__amplitude__(self.TB)

    def T_matrix_res(self):
        return self.__amplitude__(self.TR)

    def T_matrix(self):
        return self.__amplitude__(self.TB + self.TR)

    def S_matrix(self):
        phase_space_factors = self.__diag_in_channel_space__(
            np.moveaxis(
                np.array(
                    [
                        channel.phase_space_function(self.s_array)
                        for channel in self.channels.values()
                    ]
                ),
                0,
                -1,
            )
        )
        return self.IdC + 2j * phase_space_factors * self.T_matrix()

    def form_factor(self):
        return self.F
