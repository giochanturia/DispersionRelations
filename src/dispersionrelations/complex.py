# ┌──────────────────────────────────────────────────────────┐
# │                         Typing                           │
# └──────────────────────────────────────────────────────────┘
from __future__ import annotations  # for type hints in python==3.9
from typing import Callable, Literal
from numpy.typing import NDArray

# ┌──────────────────────────────────────────────────────────┐
# │                         Logging                          │
# └──────────────────────────────────────────────────────────┘
import logging

logger = logging.getLogger(__name__)

# ┌──────────────────────────────────────────────────────────┐
# │                    External imports                      │
# └──────────────────────────────────────────────────────────┘
import numpy as np
from functools import wraps

# ┌──────────────────────────────────────────────────────────┐
# │                    Internal imports                      │
# └──────────────────────────────────────────────────────────┘
from dispersionrelations.domains import DomainedFunctionCollection

# ┌──────────────────────────────────────────────────────────┐
# │                   COMPLEX FUNCTIONS                      │
# └──────────────────────────────────────────────────────────┘


def schwarz_reflection(f: Callable, *, upper: bool = True):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(z: NDArray[np.complexfloating] | complex):
            z = np.asarray(z, dtype=np.complex128)
            scalar_input = z.ndim == 0

            if scalar_input:
                z_s = z.item()
                in_domain = np.imag(z_s) >= 0 if upper else np.imag(z_s) <= 0
                return func(z_s) if in_domain else complex(np.conj(func(np.conj(z_s))))

            z = np.atleast_1d(z)
            mask_valid = np.imag(z) >= 0 if upper else np.imag(z) <= 0
            mask_reflect = ~mask_valid

            result = np.empty(z.shape, dtype=np.complex128)
            result[mask_valid] = func(z[mask_valid])
            result[mask_reflect] = np.conj(func(np.conj(z[mask_reflect])))
            return result

        return wrapper

    return decorator(f)


def extract_phase(
    f: NDArray | complex,
    jump: float = 1.5,
) -> NDArray | complex:
    r"""
    Extraction of continuous phase (angle).

    Parameters
    ----------
    f : array_like
        A sequence of complex numbers.
    jump : float
        The amount (in radians) by which the phase may jump along discontinuous points (depends on resolution).

    Returns
    -------
    t : array_like
        The same shape as input `f`.


    Examples
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from dispersionrelations.complex import extract_phase
    >>> E_1 = np.linspace(0, 1.1, 1000)
    >>> s_1 = E_1 ** 2
    >>> f_1_r = np.exp(-(s_1)**2)
    >>> f_1_θ = 2*np.pi * 3 * np.sin(2*np.pi * s_1)
    >>> f_1 = f_1_r * np.exp(1j * f_1_θ)
    >>> plt.plot(E_1, np.angle(f_1))
    >>> plt.plot(E_1, extract_phase(f_1))

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from dispersionrelations.complex import extract_phase
        E_1 = np.linspace(0, 1.1, 1000)
        s_1 = E_1 ** 2
        f_1_r = np.exp(-(s_1)**2)
        f_1_θ = 2*np.pi * 3 * np.sin(2*np.pi * s_1)
        f_1 = f_1_r * np.exp(1j * f_1_θ)
        plt.plot(E_1, np.angle(f_1), label="np.angle")
        plt.plot(E_1, extract_phase(f_1), label="extract_phase")
        plt.legend()
    """
    f_angle = np.angle(f)
    f_down = np.cumsum((np.diff(f_angle) < -jump) * 2 * np.pi)
    f_up = np.cumsum((np.diff(f_angle) > jump) * 2 * np.pi)
    f_corr = np.concatenate(([0], f_down - f_up))
    return f_angle + f_corr


# ┌──────────────────────────────────────────────────────────┐
# │                          SQRT                            │
# └──────────────────────────────────────────────────────────┘


def sqrt_custom_branch_cut(
    z: NDArray | complex,
    t_bc: float,
    sheet: Literal[1, 2] = 1,
) -> NDArray | complex:
    r"""
    Square root with a custom branch cut in the real part.

    Parameters
    ----------
    z : array_like
        A complex number or sequence of complex numbers.
    t_bc : float
        The angle of the branch cut in radians.
    sheet : int
        The sheet of the square root function. Must be 1 or 2.

    Returns
    -------
    w : array_like
        The same shape as input `z`.

    Raises
    ------
    ValueError
        If `sheet` is not 1 or 2.

    Examples
    --------

    Complex plot of :code:`sqrt_custom_branch_cut(z, t_bc=np.pi/7)`:

    .. plot::

        import numpy as np
        from dispersionrelations.complex import sqrt_custom_branch_cut
        from dispersionrelations.plotting import *
        re_im = prepare_re_im(-10, 10, 200, -10, 10, 201)
        label = r"sqrt"
        sqrt_plot_input = prepare_complex_plot_input(lambda z: sqrt_custom_branch_cut(z, np.pi/7), *re_im, label)
        _ = complex_plot_contours(sqrt_plot_input)
    """
    if sheet not in (1, 2):
        raise ValueError("Sheet for the square root must be 1 or 2.")
    sheet_selector = dict({1: 1, 2: -1})[sheet]
    r = np.abs(z)
    t = np.angle(z)
    t = np.where(t >= t_bc, t, t + 2 * np.pi) - t_bc
    return np.sqrt(r) * np.exp(1j * t / 2) * sheet_selector


def sqrt_rhc(
    z: NDArray | complex,
    sheet: Literal[1, 2] = 1,
) -> NDArray | complex:
    r"""
    Square root with the right-hand-cut :math:`z\in[0,\infty)` in the real part.

    Parameters
    ----------
    z : array_like
        A complex number or sequence of complex numbers.
    sheet : int
        The sheet of the square root function. Must be 1 or 2.

    Returns
    -------
    w : array_like
        The same shape as input `z`.

    Raises
    ------
    ValueError
        If `sheet` is not 1 or 2.

    Examples
    --------

    Complex plot of :code:`sqrt_rhc`:

    .. plot::

        import numpy as np
        from dispersionrelations.complex import sqrt_rhc
        from dispersionrelations.plotting import *
        re_im = prepare_re_im(-10, 10, 200, -10, 10, 201)
        label = r"sqrt_rhc"
        sqrtRHC_plot_input = prepare_complex_plot_input(sqrt_rhc, *re_im, label)
        _ = complex_plot_contours(sqrtRHC_plot_input)
    """
    return sqrt_custom_branch_cut(z, t_bc=0, sheet=sheet)


def sqrt_lhc(
    z: NDArray | complex,
    sheet: Literal[1, 2] = 1,
) -> NDArray | complex:
    r"""
    Square root with the left-hand-cut :math:`z\in(-\infty,0]` in the real part.

    Parameters
    ----------
    z : array_like
        A complex number or sequence of complex numbers.
    sheet : int
        The sheet of the square root function. Must be 1 or 2.

    Returns
    -------
    w : array_like
        The same shape as input `z`.

    Raises
    ------
    ValueError
        If `sheet` is not 1 or 2.

    Examples
    --------

    Complex plot of :code:`sqrt_lhc`:

    .. plot::

        import numpy as np
        from dispersionrelations.complex import sqrt_lhc
        from dispersionrelations.plotting import *
        re_im = prepare_re_im(-10, 10, 200, -10, 10, 201)
        label = r"sqrt_lhc"
        sqrtLHC_plot_input = prepare_complex_plot_input(sqrt_lhc, *re_im, label)
        _ = complex_plot_contours(sqrtLHC_plot_input)

    One can compare this to the numpy implementation :code:`np.sqrt`, where the cut is in the imaginary part:

    .. plot::

        import numpy as np
        from dispersionrelations.plotting import *
        re_im = prepare_re_im(-10, 10, 200, -10, 10, 201)
        label = r"np.sqrt"
        sqrt_plot_input = prepare_complex_plot_input(np.sqrt, *re_im, label)
        _ = complex_plot_contours(sqrt_plot_input)
    """
    return sqrt_custom_branch_cut(z, t_bc=np.pi, sheet=sheet)


# ┌──────────────────────────────────────────────────────────┐
# │                           LOG                            │
# └──────────────────────────────────────────────────────────┘


def log_custom_branch_cut(
    z: NDArray | complex,
    t_bc: float,
) -> NDArray | complex:
    r"""
    Logarithm with a custom branch cut.

    Parameters
    ----------
    z : array_like
        A complex number or sequence of complex numbers.
    t_bc : float
        The angle of the branch cut in radians.

    Returns
    -------
    w : array_like
        The same shape as input `z`.

    Examples
    --------

    Complex plot of :code:`log_custom_branch_cut(z, t_bc=-np.pi/2)`:

    .. plot::

        import numpy as np
        from dispersionrelations.complex import log_custom_branch_cut
        from dispersionrelations.plotting import *
        re_im = prepare_re_im(-10, 10, 200, -10, 10, 201)
        label = r"log"
        per_column_log = {**per_column_default}
        per_column_log[0]["ncontours"] = 24
        per_column_log[1]["ncontours"] = 48
        per_column_log[2]["ncontours"] = 48
        log_plot_input = prepare_complex_plot_input(lambda z: log_custom_branch_cut(z, -np.pi/2), *re_im, label)
        _ = complex_plot_contours(log_plot_input, per_column_log)
    """
    r = np.absolute(z)
    t = np.angle(z)
    t = np.where(t >= t_bc, t, t + 2 * np.pi) - t_bc
    return np.log(r) + 1j * t


def log_rhc(
    z: NDArray | complex,
) -> NDArray | complex:
    r"""
    Logarithm with the right-hand-cut :math:`z\in[0,\infty)`.

    Parameters
    ----------
    z : array_like
        A complex number or sequence of complex numbers.

    Returns
    -------
    w : array_like
        The same shape as input `z`.

    Examples
    --------

    Complex plot of :code:`log_rhc`:

    .. plot::

        import numpy as np
        from dispersionrelations.complex import log_rhc
        from dispersionrelations.plotting import *
        re_im = prepare_re_im(-10, 10, 200, -10, 10, 201)
        label = r"log"
        per_column_log = {**per_column_default}
        per_column_log[0]["ncontours"] = 24
        per_column_log[1]["ncontours"] = 48
        per_column_log[2]["ncontours"] = 48
        log_plot_input = prepare_complex_plot_input(lambda z: log_rhc(z), *re_im, label)
        _ = complex_plot_contours(log_plot_input, per_column_log)
    """
    return log_custom_branch_cut(z, t_bc=0)


def log_lhc(
    z: NDArray | complex,
) -> NDArray | complex:
    r"""
    Logarithm with the left-hand-cut :math:`z\in(-\infty,0]`.

    Parameters
    ----------
    z : array_like
        A complex number or sequence of complex numbers.

    Returns
    -------
    w : array_like
        The same shape as input `z`.

    Examples
    --------

    Complex plot of :code:`log_lhc`:

    .. plot::

        import numpy as np
        from dispersionrelations.complex import log_lhc
        from dispersionrelations.plotting import *
        re_im = prepare_re_im(-10, 10, 200, -10, 10, 201)
        label = r"log"
        per_column_log = {**per_column_default}
        per_column_log[0]["ncontours"] = 24
        per_column_log[1]["ncontours"] = 48
        per_column_log[2]["ncontours"] = 48
        log_plot_input = prepare_complex_plot_input(lambda z: log_lhc(z), *re_im, label)
        _ = complex_plot_contours(log_plot_input, per_column_log)
    """
    return log_custom_branch_cut(z, t_bc=np.pi)


def log_c(
    z: NDArray | complex,
) -> NDArray | complex:
    r"""
    Logarithm convention used for Cauchy integrals (right-hand-cut).

    Parameters
    ----------
    z : array_like
        A complex number or sequence of complex numbers.

    Returns
    -------
    w : array_like
        The same shape as input `z`.

    Examples
    --------

    Complex plot of :code:`log_c`:

    .. plot::

        import numpy as np
        from dispersionrelations.complex import log_c
        from dispersionrelations.plotting import *
        re_im = prepare_re_im(-10, 10, 200, -10, 10, 201)
        label = r"log"
        per_column_log = {**per_column_default}
        per_column_log[0]["ncontours"] = 24
        per_column_log[1]["ncontours"] = 48
        per_column_log[2]["ncontours"] = 48
        log_plot_input = prepare_complex_plot_input(lambda z: log_c(z), *re_im, label)
        _ = complex_plot_contours(log_plot_input, per_column_log)
    """
    return np.log(np.abs(z)) - 1j * np.where(
        np.angle(z) < 0, np.angle(z) + 2 * np.pi, np.angle(z)
    )


class Complex2RS(Callable):
    def __init__(self):
        self.sheet_1 = DomainedFunctionCollection([])
        self.sheet_2 = DomainedFunctionCollection([])
        self.discontinuity = DomainedFunctionCollection([])

    def __call__(
        self,
        z: NDArray | complex,
        sheet: Literal[1, 2] = 1,
    ) -> NDArray | complex:
        if sheet == 1:
            return self.sheet_1(z)
        elif sheet == 2:
            return self.sheet_2(z)
        else:
            raise ValueError("Sheet must be 1 or 2.")
