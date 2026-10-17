# ┌──────────────────────────────────────────────────────────┐
# │                         Typing                           │
# └──────────────────────────────────────────────────────────┘
from __future__ import annotations  # for type hints in python==3.9
from typing import Callable, NewType
from typing_extensions import override
from numpy.typing import NDArray

# ┌──────────────────────────────────────────────────────────┐
# │                         Logging                          │
# └──────────────────────────────────────────────────────────┘
import logging

from dispersionrelations.domains import Domainify

logger = logging.getLogger(__name__)

# ┌──────────────────────────────────────────────────────────┐
# │                    External imports                      │
# └──────────────────────────────────────────────────────────┘
import warnings
import numpy as np

# ┌──────────────────────────────────────────────────────────┐
# │                    Internal imports                      │
# └──────────────────────────────────────────────────────────┘
from dispersionrelations.complex import log_c, Complex2RS, log_lhc, schwarz_reflection

# ┌──────────────────────────────────────────────────────────┐
# │                       INTEGRALS                          │
# └──────────────────────────────────────────────────────────┘

# ┌──────────────────────────────────────────────────────────┐
# │                      Partitions                          │
# └──────────────────────────────────────────────────────────┘

Partition = NewType("Partition", NDArray[np.float64])


def even_partition(
    split_n: int,
) -> Partition:
    if (split_n < 1) or (split_n % 1 != 0):
        raise ValueError(
            f"Number of intervals (split_n) must be at least 1 and an integer, instead got {split_n}."
        )
    partition = np.linspace(0, 1, split_n + 1)
    return partition


def logarithmic_partition(
    split_n: int,
    base: float = 2,
) -> Partition:
    if (split_n < 1) or (split_n % 1 != 0):
        raise ValueError(
            f"Number of intervals (split_n) must be at least 1 and an integer, instead got {split_n}."
        )
    partition = np.linspace(0, 1, split_n + 1)
    partition[1:-1] = 1 / np.power(base, np.arange(split_n - 1, 0, -1))
    return partition


def distance_based_partition(
    split_n: int,
    distance_parallel: NDArray | float,
    distance_transverse: NDArray | float,
    min_distance_transverse: NDArray | float = 1e-3,
) -> Partition:
    if (split_n < 1) or (split_n % 1 != 0):
        raise ValueError(
            f"Number of intervals (split_n) must be at least 1 and an integer, instead got {split_n}."
        )
    center = np.asarray(distance_parallel)
    height = np.maximum(
        np.abs(np.asarray(distance_transverse)),
        min_distance_transverse,
    )
    norm = 1 / height * (np.atan((1 - center) / height) - np.atan(-center / height))
    partition = np.empty((split_n + 1, *norm.shape))
    partition[0, ...] = 0
    partition[-1, ...] = 1
    partition_in_between = np.linspace(0, 1, split_n + 1)[1:-1]
    if partition.ndim > 1:
        partition_in_between = np.expand_dims(
            partition_in_between, axis=tuple(range(1, partition.ndim))
        )
    partition[1:-1] = center + height * np.tan(
        (height * norm) * partition_in_between + np.atan(-center / height)
    )
    return np.moveaxis(partition, -1, 0)


def pole_based_partition(
    split_n: int,
    startpoint: NDArray | complex,
    endpoint: NDArray | complex,
    pole_position: complex,
    min_distance: float = 1e-3,
) -> Partition:
    scale = np.abs(endpoint - startpoint)
    start_to_end = (endpoint - startpoint) / scale
    start_to_pole = (pole_position - startpoint) / scale
    angle_pole_start_end = np.angle(start_to_pole / start_to_end)
    center_scalar = np.abs(start_to_pole) * np.cos(angle_pole_start_end)
    height_scalar = np.abs(np.abs(start_to_pole) * np.sin(angle_pole_start_end))
    return distance_based_partition(
        split_n, center_scalar, height_scalar, min_distance_transverse=min_distance
    )


# ┌──────────────────────────────────────────────────────────┐
# │                      Intervals                           │
# └──────────────────────────────────────────────────────────┘

Interval = NewType("Interval", NDArray[np.complex128])


def remove_duplicate_neighbours(array, axis=-1):
    "For more complicated dimensions, only those elements are retained, that are not considered duplicates in other slices."
    array = np.asarray(array)

    if np.any(np.take(array, 0, axis=axis) == np.take(array, -1, axis=axis)):
        raise ValueError(
            f"First and last index of chosen axis should not be equal, instead got {np.take(array, 0, axis = axis)} and {np.take(array, -1, axis = axis)}."
        )

    axis = axis % array.ndim
    duplicate = np.roll(array, -1, axis=axis)

    mask = array != duplicate
    if array.ndim > 1:
        axes = tuple(ax for ax in range(array.ndim) if ax != axis)
        mask_big = 1 * mask
        mask = np.prod(mask, axis=axes, dtype=bool)

    no_dupl = array[..., mask]

    return no_dupl


def split_interval(
    interval: Interval,
    partition: Partition,
) -> Interval:
    interval = np.asarray(interval)
    lower = interval[..., :-1]
    upper = interval[..., 1:]
    sizes = upper - lower
    partition = np.clip(partition, 0, 1)
    interval_split = lower[..., np.newaxis] + sizes[..., np.newaxis] * partition
    interval_split_shape = interval_split.shape
    interval_concat_shape = tuple(
        [
            *interval_split_shape[:-2],
            interval_split_shape[-2] * interval_split_shape[-1],
        ]
    )
    interval_concat = np.reshape(interval_split, interval_concat_shape)
    interval_unique = remove_duplicate_neighbours(interval_concat, axis=-1)
    return interval_unique


# def split_interval_evenly(
#     interval: Interval,
#     split_n: int,
# ) -> Interval:
#     return split_interval(interval, even_partition(split_n))


# def split_interval_logarithmically(
#     interval: Interval,
#     split_n: int,
#     base: int = 2,
# ) -> Interval:
#     return split_interval(interval, logarithmic_partition(split_n, base))


# ┌──────────────────────────────────────────────────────────┐
# │                      GL Routines                         │
# └──────────────────────────────────────────────────────────┘

gl_storage = {}


def gl_roots_and_weights(
    order: int = 100,
) -> tuple[NDArray, NDArray]:
    if order not in gl_storage.keys():
        gl_storage[order] = np.polynomial.legendre.leggauss(order)
    return gl_storage[order]


def interval_to_gl(
    interval,
    gl_order: int = 100,
):
    interval = np.asarray(interval)
    lower = interval[..., :-1]
    upper = interval[..., 1:]
    gl_roots, gl_weights = gl_roots_and_weights(gl_order)
    jacobian = (upper - lower) / 2
    points = lower[..., np.newaxis] + jacobian[..., np.newaxis] * (gl_roots + 1)
    weights = jacobian[..., np.newaxis] * gl_weights
    return points, weights


def integrate_interval_gl(
    function_x: Callable,
    x_interval: Interval,
    gl_order: int = 100,
    partition: Partition | None = None,
):
    if partition is not None:
        x_interval = split_interval(x_interval, partition)
    else:
        x_interval = np.asarray(x_interval)
    x_gl, w_gl = interval_to_gl(x_interval, gl_order)
    dim_x = len(x_interval.shape) - 1
    gl_axes = tuple(dim_x + np.arange(2))
    return np.sum(function_x(x_gl) * w_gl, gl_axes)


def integrate_interval_second_variable_gl(
    function_x_y: Callable,
    x_values: NDArray,
    y_interval: Interval,
    order: int = 100,
    y_bounds_depend_on_x: bool = False,
) -> NDArray:
    x_values = np.asarray(x_values)
    y_interval = np.asarray(y_interval)
    y_gl, w_gl = interval_to_gl(y_interval, order)
    dim_x = len(x_values.shape)
    dim_y = len(y_interval.shape) - 1
    if y_bounds_depend_on_x:
        x_axes = tuple(np.arange(dim_x))
        y_axes = tuple(np.arange(dim_y))
        gl_axes = tuple(dim_y + np.arange(2))
        x_expanded = np.expand_dims(x_values, gl_axes)
        y_expanded = y_gl
        w_expanded = w_gl
        f_expanded = function_x_y(x_expanded, y_expanded)
        f_integrated = np.sum(f_expanded * w_expanded, gl_axes)
        return f_integrated
    x_axes = tuple(np.arange(dim_x))
    y_axes = tuple(dim_x + np.arange(dim_y))
    gl_axes = tuple(dim_x + dim_y + np.arange(2))
    x_expanded = np.expand_dims(x_values, y_axes + gl_axes)
    y_expanded = np.expand_dims(y_gl, x_axes)
    w_expanded = np.expand_dims(w_gl, x_axes)
    f_expanded = function_x_y(x_expanded, y_expanded)
    f_integrated = np.sum(f_expanded * w_expanded, gl_axes)
    return f_integrated


def integrate_gl(
    function_x: Callable,
    x_lower: NDArray | complex,
    x_upper: NDArray | complex,
    gl_order: int = 100,
    partition: Partition = even_partition(split_n=3),
) -> NDArray | complex:
    x_interval = np.stack((x_lower, x_upper), axis=-1)
    return integrate_interval_gl(function_x, x_interval, gl_order, partition)


# def integrate_gl(
#     function: Callable[[complex], complex],
#     a: float | complex,
#     b: float | complex,
#     order: int = 100,
#     partition: Partition = even_partition(split_n=3),
# ) -> complex:
#     r"""
#     Gauss–Legendre integration.

#     Parameters
#     ----------
#     function : Callable
#         A complex function of a single variable.
#     a : float, complex
#         Lower boundary of the integral.
#     b : float, complex
#         Upper boundary of the integral.
#     order : int
#         Quadrature order.
#     split_n : int
#         Number of subintervals of the same length :math:`[a,x], [x,y], ... [z,b]`.

#     Returns
#     -------
#     I : float, complex
#         Integral :math:`I = \int_a^b f(x) \mathrm{d}x`.
#     """
#     interval = partition()
#     return integrate_interval_gl(function, interval, order)


def residue(
    function: Callable,
    z0: complex,
    radius: float = 1e-2,
    order: int = 100,
    partition: Partition = even_partition(split_n=3),
) -> complex:
    r"""
    Residue of a complex function.

    Parameters
    ----------
    function : Callable
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
    function_angle = (
        lambda angle: 1
        / (2 * np.pi)
        * function(z0 + radius * np.exp(1j * angle))
        * radius
        * np.exp(1j * angle)
    )
    return integrate_gl(function_angle, 0, 2 * np.pi, order, partition)


# ┌──────────────────────────────────────────────────────────┐
# │             Cauchy and dispersive integrals              │
# └──────────────────────────────────────────────────────────┘


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

    def __init__(
        self,
        n: float,
    ):
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

    def __definite_integral__(
        self,
        s: float,
        s0: float,
        a: float,
        b: float,
    ) -> complex:
        return self.indefinite_integral(s, s0, b) - self.indefinite_integral(s, s0, a)

    def __call__(
        self,
        s: float,
        s0: float,
        a: float,
        b: float,
    ) -> complex:
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

    def logfunc(self, x):
        return np.log(-(-x + 0j))  # through trial and error, needs to be double-checked
        # return log_lhc(x) + 1j * np.pi

    def __indefinite_integral_00__(
        self,
        s: float,
        s0: float,
        x: float,
    ) -> complex:
        """Special case: n=0."""
        return self.logfunc(x - s)

    def __indefinite_integral_ZP__(
        self,
        s: float,
        s0: float,
        x: float,
    ) -> complex:
        """n is a positive integer."""
        s_shifted = s - s0
        x_shifted = x - s0

        result = self.logfunc(x - s + 0j) * s_shifted**self.n
        for k in range(1, self.n + 1):
            result += 1.0 / k * x_shifted**k * (s_shifted) ** (self.n - k)

        return result

    def __indefinite_integral_ZM__(
        self,
        s: float,
        s0: float,
        x: float,
    ) -> complex:
        """n is a negative integer."""
        s_shifted = s - s0 + 0j
        x_shifted = x - s0 + 0j
        w = s_shifted / x_shifted

        # Special case:
        if self.n == -1:
            return 1 / s_shifted * self.logfunc(1 - w + 0j)
        # The rest:
        n_abs = -self.n
        result = self.logfunc(1 - w) * s_shifted**self.n
        for k in range(1, n_abs):
            result += 1.0 / k * s_shifted ** (k + self.n) / x_shifted**k

        return result

    def __indefinite_integral_ZP_HALF__(
        self,
        s: float,
        s0: float,
        x: float,
    ) -> complex:
        """n is a positive half-integer."""
        s_shifted = s - s0 + 0j
        x_shifted = x - s0 + 0j

        m = int(2 * self.n)
        result = (
            np.log(
                (np.sqrt(s_shifted) - np.sqrt(x_shifted))
                / (np.sqrt(s_shifted) + np.sqrt(x_shifted))
            )
            * s_shifted**self.n
        )

        for k in range(1, (m + 1) // 2 + 1):
            result += (
                (2 / (2 * k - 1))
                * x_shifted ** ((2.0 * k - 1) / 2.0)
                * s_shifted ** (self.n - (2 * k - 1) / 2)
            )

        return result

    def __indefinite_integral_ZM_HALF__(
        self,
        s: float,
        s0: float,
        x: float,
    ) -> complex:
        """n is a negative half-integer."""
        s_shifted = s - s0 + 0j
        x_shifted = x - s0 + 0j

        m = int(2 * self.n)
        result = (
            np.log(
                (np.sqrt(s_shifted) - np.sqrt(x_shifted))
                / (np.sqrt(s_shifted) + np.sqrt(x_shifted))
            )
            * s_shifted**self.n
        )
        for k in range(1, (m - 1) // 2 + 1):

            result += (
                (2 / (2 * k - 1))
                * s_shifted ** ((2.0 * k - 1.0) + self.n)
                / x_shifted ** ((2.0 * k - 1.0) / 2.0)
            )

        return result


class DispersionIntegralRHC:
    r"""
    Dispersion integral along the right-hand cut.

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
        Point :math:`s_0` of subtraction. Needs to be below the threshold.
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
        integrand: Callable,
        threshold: float,
        infinity: float = 1e6,
        integration_split_points: list[float] = [2, 5, 10, 50, 100],
        integration_order: int = 100,
        subtraction_point: float = 0,
        subtraction_constants: list[float] = [0],
    ):
        if subtraction_point >= threshold:
            raise ValueError("Please make sure that subtraction point < threshold")
        self.integrand = integrand
        self.threshold = threshold  # TODO what if threshold is zero?!
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

    def __precompute__(
        self,
    ) -> None:
        self.gl_roots, self.gl_weights = gl_roots_and_weights(self.order)

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

    def __compute_real__(
        self,
        s: NDArray | float,
    ) -> NDArray | complex:
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

    # def __compute_complex__(
    #     self,
    #     s: NDArray | complex,
    # ) -> NDArray | complex:
    #     s = np.array(s)
    #     dim_s = len(s.shape)
    #     s_axes = tuple(np.arange(dim_s))
    #     gl_axes = tuple(dim_s + np.arange(2))

    #     scaled_weights = np.expand_dims(self.scaled_weights, s_axes)
    #     __s__ = np.expand_dims(s, gl_axes)
    #     __s_prime__ = np.expand_dims(self.s_prime, s_axes)
    #     __f_at_s_prime__ = np.expand_dims(self.f_at_s_prime, s_axes)

    #     integrand = (
    #         (__s__ - self.s0) ** self.n_subtr
    #         * (__f_at_s_prime__)
    #         / ((__s_prime__ - self.s0) ** self.n_subtr * (__s_prime__ - __s__))
    #     )

    #     numerical_integral = np.zeros(s.shape, dtype=np.complex128)

    #     numerical_integral += np.sum(integrand * scaled_weights, gl_axes)

    #     polynomial_head = 0.0
    #     for i in range(self.n_subtr):
    #         polynomial_head += self.f0_arr[i] * (s - self.s0) ** i

    #     return polynomial_head + (1 / np.pi) * numerical_integral

    def __compute_complex__(
        self,
        s: NDArray | complex,
    ) -> NDArray | complex:
        s = np.array(s)
        dim_s = len(s.shape)
        s_axes = tuple(np.arange(dim_s))
        gl_axes = tuple(dim_s + np.arange(2))

        scaled_weights = np.expand_dims(self.scaled_weights, s_axes)
        __s__ = np.expand_dims(s, gl_axes)
        __s_prime__ = np.expand_dims(self.s_prime, s_axes)
        # __f_at_s__ = self.integrand(np.real(s))
        f_at_s = np.zeros_like(s)
        mask_s_above_cut = (np.real(s) >= self.threshold)
        f_at_s[mask_s_above_cut] = self.integrand(np.real(s[mask_s_above_cut]))
        __f_at_s__ = np.expand_dims(f_at_s, gl_axes)
        __f_at_s_prime__ = np.expand_dims(self.f_at_s_prime, s_axes)
        __f_at_s_prime_sub__ = __f_at_s_prime__ - __f_at_s__

        integrand = (
            (__s__ - self.s0) ** self.n_subtr
            * (__f_at_s_prime_sub__)
            / ((__s_prime__ - self.s0) ** self.n_subtr * (__s_prime__ - __s__))
        )

        numerical_integral = np.zeros(s.shape, dtype=np.complex128)
        analytic_integral = np.zeros(s.shape, dtype=np.complex128)

        numerical_integral += np.sum(integrand * scaled_weights, gl_axes)

        # __f_at_s__ = np.zeros_like(s)
        # mask_s_above_cut = np.real(s) >= self.threshold
        # __f_at_s__[mask_s_above_cut] = self.integrand(np.real(s[mask_s_above_cut]))

        # print("shapes:")
        # print(s[mask_s_above_cut].shape)
        # print(self.s0)

        analytic_integral[mask_s_above_cut] += (
            f_at_s[mask_s_above_cut]
            * (s[mask_s_above_cut] - self.s0) ** self.n_subtr
            # * np.where(
                # np.imag(s[mask_s_above_cut]) == 0,
                * self.analytic_remainder(s[mask_s_above_cut], self.s0, self.threshold, np.inf)#,
                # 0,
            # )
        )

        polynomial_head = 0.0
        for i in range(self.n_subtr):
            polynomial_head += self.f0_arr[i] * (s - self.s0) ** i

        return polynomial_head + (1 / np.pi) * (numerical_integral + analytic_integral)


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


def disp_int_to_complex_2RS(di: DispersionIntegralRHC, cf: Complex2RS):
    f_1_di = lambda z: di.__call__(z)
    f_disc_di = schwarz_reflection(lambda z: 2j * di.integrand(z))
    f_2_di = lambda z: f_1_di(z) - f_disc_di(z)
    #
    cf.sheet_1.add_function(Domainify(f_1_di))
    cf.sheet_2.add_function(Domainify(f_2_di))
    cf.discontinuity.add_function(Domainify(f_disc_di))
