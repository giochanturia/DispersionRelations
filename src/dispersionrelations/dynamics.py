import numpy as np
from scipy.interpolate import interp1d
from dispersionrelations.constants import *
from dispersionrelations.utils import *
from dispersionrelations.kinematics import *
from dispersionrelations.integrals import *


def vertex_VPP__2(s, mP, mP_2=None):
    r"""A vector :math:`\to` pseudoscalar–pseudoscalar vertex squared.

    Parameters
    ----------
    s : array_like
        Kinematic variable (can be complex).

    mP : float
        Mass of the pseudoscalar with momentum :math:`p_1`.

    mP_2 : float
        Mass of the pseudoscalar with momentum :math:`p_2` (if not passed, a case with equal masses will be returned).

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
    if mP_2 is None:
        return 1 / 3 * (s - 4 * mP**2)
    return 1 / 3 * (s - (mP + mP_2) ** 2) * (s - (mP - mP_2) ** 2) / s


def vertex_VPP(s, mP, mP_2=None):
    r"""A vector :math:`\to` pseudoscalar–pseudoscalar vertex.

    Parameters
    ----------
    s : array_like
        Kinematic variable (can be complex).

    mP : float
        Mass of the pseudoscalar with momentum :math:`p_1`.

    mP_2 : float
        Mass of the pseudoscalar with momentum :math:`p_2` (if not passed, a case with equal masses will be returned).

    Returns
    -------
    b : array_like
        The same shape as input `s`.

    Notes
    -----
    This returns the square root of :func:`vertex_VPP__2`, i.e. :math:`\frac{1}{\sqrt{3}}\beta_{VPP}(s, m_{P1}^2, m_{P2}^2)`.
    For most applications we need the squared version.
    """
    return np.sqrt(vertex_VPP__2(s, mP, mP_2) + 0j)


def vertex_VVP__2(s, mV, mP):
    r"""A vector :math:`\to` vector–pseudoscalar vertex squared.

    Parameters
    ----------
    s : array_like
        Kinematic variable (can be complex).

    mV : float
        Mass of the vector with momentum :math:`k`.

    mP : float
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
    return 1 / 6 * (s - (mV - mP) ** 2) * (s - (mV + mP) ** 2)


def vertex_VVP(s, mV, mP):
    r"""A vector :math:`\to` vector–pseudoscalar vertex.

    Parameters
    ----------
    s : array_like
        Kinematic variable (can be complex).

    mV : float
        Mass of the vector with momentum :math:`k`.

    mP : float
        Mass of the pseudoscalar with momentum :math:`p`.

    Returns
    -------
    b : array_like
        The same shape as input `s`.

    Notes
    -----
    This returns the square root of :func:`vertex_VVP__2`, i.e. :math:`\frac{1}{\sqrt{3}}\beta_{VVP}(s, m_V^2, m_P^2)`.
    For most applications we need the squared version.
    """
    return np.sqrt(vertex_VVP__2(s, mV, mP) + 0j)


def vertex_AVP__2(s, mV, mP):
    r"""An axial vector :math:`\to` vector–pseudoscalar vertex squared.

    Parameters
    ----------
    s : array_like
        Kinematic variable (can be complex).

    mV : float
        Mass of the vector with momentum :math:`k`.

    mP : float
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
    return 1 / 6 * (2 * s * mV**2 + (s + mV**2 - mP**2) ** 2)


def vertex_AVP(s, mV, mP):
    r"""An axial vector :math:`\to` vector–pseudoscalar vertex.

    Parameters
    ----------
    s : array_like
        Kinematic variable (can be complex).

    mV : float
        Mass of the vector with momentum :math:`k`.

    mP : float
        Mass of the pseudoscalar with momentum :math:`p`.

    Returns
    -------
    b : array_like
        The same shape as input `s`.

    Notes
    -----
    This returns the square root of :func:`vertex_AVP__2`, i.e. :math:`\frac{1}{\sqrt{3}}\beta_{AVP}(s, m_V^2, m_P^2)`.
    For most applications we need the squared version.
    """
    return np.sqrt(vertex_AVP__2(s, mV, mP) + 0j)


def vertex_VAP__2(s, mA, mP):
    r"""A vector :math:`\to` axial vector–pseudoscalar vertex squared.

    Parameters
    ----------
    s : array_like
        Kinematic variable (can be complex).

    mV : float
        Mass of the axial vector with momentum :math:`k`.

    mP : float
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
    return 1 / 6 * (2 * s * mA**2 + (s + mA**2 - mP**2) ** 2)


def vertex_VAP(s, mA, mP):
    r"""A vector :math:`\to` axial vector–pseudoscalar vertex.

    Parameters
    ----------
    s : array_like
        Kinematic variable (can be complex).

    mV : float
        Mass of the axial vector with momentum :math:`k`.

    mP : float
        Mass of the pseudoscalar with momentum :math:`p`.

    Returns
    -------
    b : array_like
        The same shape as input `s`.

    Notes
    -----
    This returns the square root of :func:`vertex_VAP__2`, i.e. :math:`\frac{1}{\sqrt{3}}\beta_{VAP}(s, m_A^2, m_P^2)`.
    For most applications we need the squared version.
    """
    return np.sqrt(vertex_VAP__2(s, mA, mP) + 0j)


def taming_BlattWeisskopf(s, sthr, sB, l=1, normalize_at_sB=False):
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

        \begin{align}
        B_0(s) &= 1, \\
        B_1(s) &= 1 / (x + 1), \\
        B_2(s) &= 1 / ((x-3)^2 + 9x), \\
        \end{align}

    where :math:`x=(s-s_{thr}) / (s_B - s_{thr})`.
    """
    if l == 0:
        return 1
    elif l == 1:
        x = (s - sthr) / (sB - sthr)
        numerator = 2 * x if normalize_at_sB else 1
        return numerator / (1 + x)
    elif l == 2:
        x = (s - sthr) / (sB - sthr)
        numerator = 13 * x**2 if normalize_at_sB else 1
        return numerator / ((x - 3) ** 2 + 9 * x)
    elif l == 3:
        x = (s - sthr) / (sB - sthr)
        numerator = 277 * x**3 if normalize_at_sB else 1
        return numerator / (x * (x - 15) ** 2 + 9 * (2 * x - 5))
    elif l == 4:
        x = (s - sthr) / (sB - sthr)
        numerator = 12746 * x**4 if normalize_at_sB else 1
        return numerator / ((x**2 - 45 * x + 105) ** 2 + 25 * x * (2 * x - 21) ** 2)
    else:
        raise NotImplementedError(
            f"Blatt-Weisskopf factor for l={l} has not been implemented."
        )


def taming_spacelike_pole(s, sB=(2 * GeV) ** 2, n=1):
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

    This creates a space-like pole (of degree `n`) at :math:`s = -s_B`.
    """
    return (sB / (sB + s)) ** n


def BreitWigner(s, mass, width):
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


def BreitWignerED(s, mass, width_s, width_0):
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
    dispersionrelations.kinematics.BreitWigner
    """
    return 1 / (s - mass**2 + 1j * mass * width_s(s, mass, width_0))


def radiative_width_to_normalisation(width, mV, mP):
    r"""Computes the vector–pseudoscalar form factor normalisation squared from the radiative width of the vector particle.

    Parameters
    ----------
    width : float
        Radiative width of the vector particle.
    mV : float
        Mass of the vector particle.
    mP : float
        Mass of the pseudoscalar particle.

    Returns
    -------
    c : float
        Normalisation squared :math:`|f(0)|^2`.

    Notes
    -----

    The radiative width :math:`\Gamma` is connected to the form factor normalisation :math:`|f(0)|` via

    .. math:: \Gamma_{V \rightarrow P\gamma} = \frac{\alpha(M_V^2 - M_P^2)^3}{24 M_V^3} |f_{VP}(0)|^2.
    
    """
    return width / ((alpha_fs / 24) * ((mV**2 - mP**2) / mV) ** 3)


class TwoBodyChannel(DispersionIntegralRHC):
    r"""
    Dispersive representation of a two-body channel.

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
        loop_integrand,
        threshold,
        infinity=1e6,
        integration_split_points=[2, 3, 5, 8, 10, 50, 1e2, 1e3, 1e4, 1e5],
        integration_order=100,
        subtraction_point=0,
        subtraction_constants=[0],
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

    def loop_discontinuity(self, s):
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
        
        We assume the Schwarz reflection principle and therefore,

        .. math::
            \mathrm{Disc}(\Pi(s)) = 2i \, \mathrm{Im}(\Pi(s)) \, .

        """
        result_up = np.where(np.imag(s) >= 0, 2j * self.integrand(s), 0)
        result_down = np.where(
            np.imag(s) < 0, np.conj(2j * self.integrand(np.conj(s))), 0
        )
        return result_up + result_down

    def loop(self, s, sheet=1):
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


class StableTwoBodyChannel(TwoBodyChannel):
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
    def __init__(self, m1, m2, vertex_function_squared=vertex_VPP__2, taming_factor=taming_spacelike_pole, **kwargs):
        self.m1 = m1
        self.m2 = m2
        self.threshold = (m1 + m2)**2
        self.phase_space = lambda s: phase_space_twobody(s, m1, m2)
        self.vertex_function_squared = lambda s: vertex_function_squared(s, m1, m2)
        self.taming_factor = taming_factor
        self.loop_integrand = lambda s: self.phase_space(s) * self.vertex_function_squared(s) * self.taming_factor(s)**2
        super().__init__(self.loop_integrand, self.threshold, **kwargs)


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
        Location of the resonance pole (see :func:`dispersionrelations.constants.sR`).
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
        decay_channel,
        pole_location,
        pole_sheet=2,
        mass_and_coupling=None,
        interpolation_points=[1, 2, 5, 1e1, 5e1, 1e2, 1e3, 1e5, 1e8],
        interpolation_density=1000,
    ):
        self.decay_channel = decay_channel
        self.sp = pole_location
        self.pole_sheet = pole_sheet
        self.mass_and_coupling = mass_and_coupling
        self.__calculate_m_g__()
        self.interpolation_points = interpolation_points
        self.interpolation_density = interpolation_density
        self.__interpolate_spectral_function__()

    def __calculate_m_g__(self):
        if self.mass_and_coupling is not None:
            self.m, self.g = self.mass_and_coupling
            return

        # The code below is the only part where self.decay_channel.loop should be known for complex numbers.
        # If mass_and_coupling is provided, real-valued loop is sufficient.
        loop_at_the_pole = self.decay_channel.loop(self.sp, sheet=self.pole_sheet)
        self.g = np.sqrt(-np.imag(self.sp) / np.imag(loop_at_the_pole))
        self.m = np.sqrt(np.real(self.sp) + self.g**2 * np.real(loop_at_the_pole))

    def propagator(self, s, sheet=1):
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

    def spectral_function_CP(self, s):
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
            / np.pi
            * self.propagator(s, sheet=1)
            * self.propagator(s, sheet=2)
            * self.g**2
            * self.decay_channel.integrand(s)
        )

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

    def spectral_function_RE(self, s):
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
        unstable_particle,
        spectator_mass,
        vertex_function,
        phase_space=phase_space_twobody,
        taming_factor=taming_spacelike_pole,
        gl_order=100,
        split_n=3,
        interpolation_points=[1, 2, 3, 4, 5, 1e1, 3e1, 5e1, 1e2, 1e3, 1e5, 1e8, 1e9],
        interpolation_density=1000,
    ):
        self.unstable_particle = unstable_particle
        self.spectator_mass = spectator_mass
        self.phase_space = phase_space
        self.vertex_function = vertex_function
        self.taming_factor = taming_factor
        self.gl_order = gl_order
        self.split_n = split_n
        self.interpolation_points = interpolation_points
        self.interpolation_density = interpolation_density
        self.__interpolate_ImPI_integral__()

    def ImPI_integrand_RE(self, s, x):
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
            * self.phase_space(s, np.sqrt(x + 0j), self.spectator_mass)
            * self.vertex_function(s, np.sqrt(x + 0j), self.spectator_mass) ** 2
            * self.taming_factor(s) ** 2
        )

    def ImPI_integrand_CP(self, s, x):
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
            * self.phase_space(s, np.sqrt(x + 0j), self.spectator_mass)
            * self.vertex_function(s, np.sqrt(x + 0j), self.spectator_mass) ** 2
            * self.taming_factor(s) ** 2
        )

    def ImPI_integral(self, s):
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

        integral_1 = integrate_gl(
            integrand,
            lower_bound,
            middle_bound,
            order=self.gl_order,
            split_n=self.split_n,
        )

        if np.imag(upper_bound) == 0:
            return integral_1

        integral_2 = integrate_gl(
            integrand,
            lower_bound,
            middle_bound,
            order=self.gl_order,
            split_n=self.split_n,
        )

        return integral_1 + integral_2

    @property
    def ImPI_integral_vectorized(self):
        r"""Vectorized version of :func:`dispersionrelations.dynamics.SemiStableTwoBodyCut.ImPI_integral`"""
        return np.vectorize(self.ImPI_integral)

    def __interpolate_ImPI_integral__(self):
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

    def ImPI_integral_RE(self, s):
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
            result[~above_interpolation_maximum] = self.ImPI_integral_vectorized(
                s[~above_interpolation_maximum]
            )
        return result

    def ImPI_integral_CP(self, s):
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

    def __call__(self, s):
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

