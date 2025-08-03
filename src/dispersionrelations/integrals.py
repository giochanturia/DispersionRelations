import numpy as np
from dispersionrelations.utils import *

z_gl_100, w_gl_100 = np.polynomial.legendre.leggauss(100)
z_gl_500, w_gl_500 = np.polynomial.legendre.leggauss(500)
z_gl_1000, w_gl_1000 = np.polynomial.legendre.leggauss(1000)
z_gl_2000, w_gl_2000 = np.polynomial.legendre.leggauss(2000)

gl_storage = {
    100: (z_gl_100, w_gl_100),
    500: (z_gl_500, w_gl_500),
    1000: (z_gl_1000, w_gl_1000),
    2000: (z_gl_2000, w_gl_2000),
}


def integrate_gl(function, a, b, order=100, split_n=3):
    r"""
    Gauss–Legendre integration.
    
    Parameters
    ----------
    function : callable
        A complex function of a single variable.
    a : float, complex
        Lower boundary of the integral.
    b : float, complex
        Upper boundary of the integral.
    order : int
        Quadrature order.
    split_n : int
        Number of subintervals of the same length :math:`[a,x], [x,y], ... [z,b]`.

    Returns
    -------
    I : float, complex
        Integral :math:`I = \int_a^b f(x) \mathrm{d}x`.
    """
    if order in gl_storage.keys():
        (gl_roots, gl_weights) = gl_storage[order]
    else:
        (gl_roots, gl_weights) = np.polynomial.legendre.leggauss(order)
    jacobian = (b - a) / 2 / split_n
    s_prime = np.array(
        [[(x + 1 + 2 * i) * jacobian + a for x in gl_roots] for i in range(split_n)]
    )
    return np.sum(function(s_prime) * gl_weights * jacobian)


def residue(function, z0, radius=1e-2, order=100, split_n=3):
    r"""
    Residue of a complex function.
    
    Parameters
    ----------
    function : callable
        Complex function of a single variable.
    z0 : complex
        Complex point to compute the residue at.
    radius : float
        Radius of the circular integral around :math:`z_0`.
    order : int
        Quadrature order.
    split_n : int
        Number of sections of the circle.

    Returns
    -------
    r : complex
        Residue, computed via :math:`r = \frac{1}{2\pi i}\oint_{\mathcal{C}_{z_0}} f(z) \mathrm{d}z`.

    Examples
    --------
    >>> from dispersionrelations import integrals
    >>> print(integrals.residue(lambda z: 13/(z-2-2j), z0=2+2j))
    np.complex128(13.000000000000007+4.3193350746692004e-15j)
    """
    function_θ = (
        lambda θ: 1
        / (2 * np.pi)
        * function(z0 + radius * np.exp(1j * θ))
        * radius
        * np.exp(1j * θ)
    )
    return integrate_gl(function_θ, 0, 2*np.pi, order, split_n)

class SimplePoleMonomial:
    r"""
    Cauchy integral of a shifted monomial

    Parameters
    ----------
    n : float
        Power of the shifted monomial (can be an integer or a half-integer).
        
    Raises
    ------
    NotImplementedError
        Any input :code:`n` other than an integer or a half-integer triggers an error.
    """
    def __init__(self, n):
        self.n = n
        if n % 1 == 0:
            if n == 0:
                self.indefinite_integral = self.__indefinite_integral_00__
            elif n > 0:
                self.indefinite_integral = self.__indefinite_integral_ZP__
            elif n < 0:
                self.indefinite_integral = self.__indefinite_integral_ZM__
            else:
                raise NotImplementedError(
                    f"Integral assignment failed for n={self.n}. Only integers and half-integers are supported."
                )
        elif n % 1 == 1 / 2:
            if n > 0:
                self.indefinite_integral = self.__indefinite_integral_ZP_HALF__
            elif n < 0:
                self.indefinite_integral = self.__indefinite_integral_ZM_HALF__
            else:
                raise NotImplementedError(
                    f"Integral assignment failed for n={self.n}. Only integers and half-integers are supported."
                )
        else:
            raise NotImplementedError(
                f"Integral assignment failed for n={self.n}. Only integers and half-integers are supported."
            )

    def __definite_integral__(self, s, s0, a, b):
        return self.indefinite_integral(s, s0, b) - self.indefinite_integral(s, s0, a)

    def __call__(self, s, s0, a, b):
        r"""
        Definite integral

        .. math::
            \mathcal{I}_{n}(s,s_0;a,b) = \int_a^b \frac{(x-s_0)^n \, \mathrm{d}x}{x-s}.

        Parameters
        ----------
        s : float
            Location of the Cauchy singularity.
        s0 : float
            Shift of the monomial in the integrand.
        a : float
            Lower integration boundary.
        b : float
            Upper integration boundary.

        Notes
        -----
        Several special cases are defined separately,

        Case 1. :math:`n = 0`,

        .. math:: \mathcal{I}_{0}(s, s_0; a, b) = \log(x-s) \bigg|_a^b \, .

        Case 2. :math:`n` is a positive integer,

        .. math::
            \mathcal{I}_{n \in \mathbb{Z}^+}(s, s_0; a, b) =
            (s-s_0)^n \left(
            \mathcal{I}_{0}(s, s_0; a, b)
            + \sum_{k=1}^{n} \frac{1}{k}\frac{(x-s_0)^{k}}{(s-s_0)^{k}}
            \bigg|_a^b
            \right) \, .

        Case 3. :math:`n = -\bar{n}` is a negative integer,

        .. math::
            \mathcal{I}_{-\bar{n}}(s, s_0; a, b)
            = (s-s_0)^{-\bar{n}} \left(
            \log\left(\frac{x-s}{x-s_0}\right)
            + \sum_{k=1}^{\bar{n}-1} \frac{1}{k} \frac{(s-s_0)^{k}}{(x-s_0)^{k}}
            \right)\bigg|_{a}^{b} \, .

        Case 4. :math:`n=\frac{m}{2}` is a positive half-integer,

        .. math::
            \mathcal{I}_{m/2}(s, s_0; a, b)
            = \sqrt{s-s_0}^{m} \left(
            \log\left(\frac{\sqrt{s-s_0}-\sqrt{x-s_0}}{\sqrt{s-s_0}+\sqrt{x-s_0}}\right)
            + \sum_{k=1}^{(m+1)/2} \frac{2}{2k-1} \frac{\sqrt{x-s_0}^{2k-1}}{\sqrt{s-s_0}^{2k-1}}
            \right)\bigg|_a^b \, .

        Case 5. :math:`n=-\frac{\bar{m}}{2}` is a negative half-integer,

        .. math::
            \mathcal{I}_{-\bar{m}/2}(s, s_0; a, b)
            \frac{1}{\sqrt{s-s_0}^{\bar{m}}}
            \left(
            \log\left(\frac{\sqrt{s-s_0}-\sqrt{x-s_0}}{\sqrt{s-s_0}+\sqrt{x-s_0}}\right)
            + \sum_{k=1}^{(\bar{m}-1)/2} \frac{2}{2k-1} \frac{\sqrt{s-s_0}^{2k-1}}{\sqrt{x-s_0}^{2k-1}}
            \right)\bigg|_a^b \, .

        """
        return self.__definite_integral__(s, s0, a, b)

    def __indefinite_integral_00__(self, s, s0, x):
        """Special case: n=0."""
        return logC(x - s)

    def __indefinite_integral_ZP__(self, s, s0, x):
        """n is a positive integer."""
        s_shifted = s - s0
        x_shifted = x - s0
        w = s_shifted / x_shifted

        result = np.log(x - s)
        for k in range(1, self.n + 1):
            result += 1 / k * (1 / w) ** k

        return s_shifted**self.n * result

    def __indefinite_integral_ZM__(self, s, s0, x):
        """n is a negative integer."""
        s_shifted = s - s0
        x_shifted = x - s0
        w = s_shifted / x_shifted

        # Special case:
        if self.n == -1:
            return 1 / s_shifted * logC(1 - w)

        # The rest:
        n_abs = -self.n
        result = logC(1 - w)
        for k in range(1, n_abs):
            result += 1 / k * w**k

        return s_shifted**self.n * result

    def __indefinite_integral_ZP_HALF__(self, s, s0, x):
        """n is a positive half-integer."""
        s_shifted = s - s0
        x_shifted = x - s0
        w = s_shifted / x_shifted
        w_sqrt = np.sqrt(w + 0j)

        m = 2 * self.n
        result = logC((w_sqrt - 1) / (w_sqrt + 1))
        for k in range(1, (m + 1) // 2 + 1):
            result += (2 / (2 * k - 1)) * (1 / w_sqrt) ** (2 * k - 1)

        return s_shifted**self.n * result

    def __indefinite_integral_ZM_HALF__(self, s, s0, x):
        """n is a negative half-integer."""
        s_shifted = s - s0
        x_shifted = x - s0
        w = s_shifted / x_shifted
        w_sqrt = np.sqrt(w + 0j)

        m = 2 * self.n
        result = logC((w_sqrt - 1) / (w_sqrt + 1))
        for k in range(1, (m - 1) // 2 + 1):
            result += (2 / (2 * k - 1)) * w_sqrt ** (2 * k - 1)

        return s_shifted**self.n * result


class DispersionIntegralRHC:
    r"""
    Dispersion integral along the right-hand cut.

    Parameters
    ----------
    integrand : callable
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
    The integral is defined as

    .. math:: F(s) = \sum_{i=0}^{n-1}f_i (s-s_0)^i + \frac{(s-s_0)^n}{\pi}\int_{s_{thr}}^{\infty}\frac{f(x) \, dx}{(x-s_0)^n(x-s-i\epsilon)},

    where :math:`s_0` is the subtraction point, :math:`f_i` are the subtraction constants,
    :math:`n` is the subtraction level, and
    :math:`f` is the `integrand`.
    """
    def __init__(
        self,
        integrand,
        threshold,
        infinity=1e6,
        integration_split_points=[2, 5, 10, 50, 100],
        integration_order=100,
        subtraction_point=0,
        subtraction_constants=[0],
    ):
        self.integrand = integrand
        self.threshold = threshold
        self.infinity = infinity
        self.interval_split = threshold * np.array(
            [1, *integration_split_points, infinity]
        )

        self.order = integration_order

        self.s0 = subtraction_point
        self.f0_arr = subtraction_constants
        self.n_subtr = len(subtraction_constants)

        self.analytic_remainder = SimplePoleMonomial(-self.n_subtr)

        self.__precompute__()
        # self.compute = np.vectorize(self.__compute__)

    def __precompute__(self):
        (self.gl_roots, self.gl_weights) = np.polynomial.legendre.leggauss(self.order)

        intervals = []
        scalings = []
        s_prime = []

        for i in range(len(self.interval_split) - 1):
            a = self.interval_split[i]
            b = self.interval_split[i + 1]
            intervals.append([a, b])
            scalings.append((b - a) / 2)
            s_prime.append(
                np.array([((x + 1) / 2) * (b - a) + a for x in self.gl_roots])
            )

        self.intervals = np.array(intervals)
        self.scalings = np.array(scalings)
        self.scaled_weights = self.gl_weights[:, None] * self.scalings[None, :]
        self.s_prime = np.array(s_prime).T
        self.f_at_s_prime = self.integrand(self.s_prime)

    def __compute_real__(self, s):
        s = np.array(s)
        dim_s = len(s.shape)
        s_axes = tuple(np.arange(dim_s))
        gl_axes = tuple(dim_s + np.arange(2))

        scaled_weights = np.expand_dims(self.scaled_weights, s_axes)
        __s__ = np.expand_dims(s, gl_axes)
        __s_prime__ = np.expand_dims(self.s_prime, s_axes)
        __f_at_s_prime__ = np.expand_dims(self.f_at_s_prime, s_axes)

        f_at_s = self.integrand(s)
        __f_at_s__ = np.expand_dims(f_at_s, gl_axes)

        integrand = (
            (__s__ - self.s0) ** self.n_subtr
            * (__f_at_s_prime__ - __f_at_s__)
            / ((__s_prime__ - self.s0) ** self.n_subtr * (__s_prime__ - __s__))
        )

        numerical_integral = np.zeros(s.shape, dtype=np.complex128)
        analytic_integral = np.zeros(s.shape, dtype=np.complex128)

        numerical_integral += np.sum(integrand * scaled_weights, gl_axes)
        analytic_integral += (
            f_at_s
            * (s - self.s0) ** self.n_subtr
            * np.where(
                np.imag(s) == 0,
                self.analytic_remainder(s, self.s0, self.threshold, np.inf),
                0,
            )
        )

        polynomial_head = 0.0
        for i in range(self.n_subtr):
            polynomial_head += self.f0_arr[i] * (s - self.s0) ** i

        return polynomial_head + (1 / np.pi) * (numerical_integral + analytic_integral)
    
    def __compute_complex__(self, s):
        s = np.array(s)
        dim_s = len(s.shape)
        s_axes = tuple(np.arange(dim_s))
        gl_axes = tuple(dim_s + np.arange(2))

        scaled_weights = np.expand_dims(self.scaled_weights, s_axes)
        __s__ = np.expand_dims(s, gl_axes)
        __s_prime__ = np.expand_dims(self.s_prime, s_axes)
        __f_at_s_prime__ = np.expand_dims(self.f_at_s_prime, s_axes)

        integrand = (
            (__s__ - self.s0) ** self.n_subtr
            * (__f_at_s_prime__)
            / ((__s_prime__ - self.s0) ** self.n_subtr * (__s_prime__ - __s__))
        )

        numerical_integral = np.zeros(s.shape, dtype=np.complex128)
        
        numerical_integral += np.sum(integrand * scaled_weights, gl_axes)

        polynomial_head = 0.0
        for i in range(self.n_subtr):
            polynomial_head += self.f0_arr[i] * (s - self.s0) ** i

        return polynomial_head + (1 / np.pi) * numerical_integral

    def __call__(self, s):
        r"""
        Parameters
        ----------
        s : array_like
            Four-momentum squared of the propagating particle.

        Returns
        -------
        F : array_like
            The same shape as input `s`.
        """
        s = np.array(s)
        is_complex = np.imag(s) != 0
        result = np.zeros(s.shape, dtype=np.complex128)
        if np.sum(is_complex) > 0:
            result[is_complex] = self.__compute_complex__(s[is_complex])
        if np.sum(~is_complex) > 0:
            result[~is_complex] = self.__compute_real__(np.real(s[~is_complex]))
        return result


class OmnesFunction(DispersionIntegralRHC):
    r"""
    Omnès–Muskhelishvili solution :cite:`Omnes:1958hv`.

    Parameters
    ----------
    integrand : callable
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
    def __call__(self, s):
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
