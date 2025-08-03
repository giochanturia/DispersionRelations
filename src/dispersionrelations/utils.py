import numpy as np


def sqrt_custom_branch_cut(z, t_bc, sheet=1):
    r"""
    Square root with a custom branch cut in the real part.

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

    Complex plot of :code:`sqrt_custom_branch_cut(z, t_bc=np.pi/7)`:

    .. plot::

        import numpy as np
        from dispersionrelations.utils import sqrt_custom_branch_cut
        from dispersionrelations.plotting import *
        re_im = prepare_re_im(-10, 10, 200, -10, 10, 201)
        label = r"sqrt"
        sqrt_plot_input = prepare_complex_plot_input(lambda z: sqrt_custom_branch_cut(z, np.pi/7), *re_im, label)
        _ = complex_plot_contours(sqrt_plot_input)
    """
    r = np.abs(z)
    if sheet==2:
        r *= -1
    t = np.angle(z)
    t = np.where(t >= t_bc, t, t + 2 * np.pi) - t_bc
    return np.sqrt(r) * np.exp(1j * t / 2)


def sqrtRHC(z):
    r"""
    Square root with the right-hand-cut :math:`z\in[0,\infty)` in the real part.

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

    Complex plot of :code:`sqrtRHC`:

    .. plot::

        import numpy as np
        from dispersionrelations.utils import sqrtRHC
        from dispersionrelations.plotting import *
        re_im = prepare_re_im(-10, 10, 200, -10, 10, 201)
        label = r"sqrtRHC"
        sqrtRHC_plot_input = prepare_complex_plot_input(sqrtRHC, *re_im, label)
        _ = complex_plot_contours(sqrtRHC_plot_input)
    """
    return sqrt_custom_branch_cut(z, t_bc=0)


def sqrtLHC(z):
    r"""
    Square root with the left-hand-cut :math:`z\in(-\infty,0]` in the real part.

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

    Complex plot of :code:`sqrtLHC`:

    .. plot::

        import numpy as np
        from dispersionrelations.utils import sqrtLHC
        from dispersionrelations.plotting import *
        re_im = prepare_re_im(-10, 10, 200, -10, 10, 201)
        label = r"sqrtLHC"
        sqrtLHC_plot_input = prepare_complex_plot_input(sqrtLHC, *re_im, label)
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
    return sqrt_custom_branch_cut(z, t_bc=np.pi)


def log_custom_branch_cut(z, t_bc):
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
        from dispersionrelations.utils import log_custom_branch_cut
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


def logRHC(z):
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

    Complex plot of :code:`logRHC`:

    .. plot::

        import numpy as np
        from dispersionrelations.utils import logRHC
        from dispersionrelations.plotting import *
        re_im = prepare_re_im(-10, 10, 200, -10, 10, 201)
        label = r"log"
        per_column_log = {**per_column_default}
        per_column_log[0]["ncontours"] = 24
        per_column_log[1]["ncontours"] = 48
        per_column_log[2]["ncontours"] = 48
        log_plot_input = prepare_complex_plot_input(lambda z: logRHC(z), *re_im, label)
        _ = complex_plot_contours(log_plot_input, per_column_log)
    """
    return log_custom_branch_cut(z, t_bc=0)


def logLHC(z):
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

    Complex plot of :code:`logLHC`:

    .. plot::

        import numpy as np
        from dispersionrelations.utils import logLHC
        from dispersionrelations.plotting import *
        re_im = prepare_re_im(-10, 10, 200, -10, 10, 201)
        label = r"log"
        per_column_log = {**per_column_default}
        per_column_log[0]["ncontours"] = 24
        per_column_log[1]["ncontours"] = 48
        per_column_log[2]["ncontours"] = 48
        log_plot_input = prepare_complex_plot_input(lambda z: logLHC(z), *re_im, label)
        _ = complex_plot_contours(log_plot_input, per_column_log)
    """
    return log_custom_branch_cut(z, t_bc=np.pi)


def logC(z):
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

    Complex plot of :code:`logC`:

    .. plot::

        import numpy as np
        from dispersionrelations.utils import logC
        from dispersionrelations.plotting import *
        re_im = prepare_re_im(-10, 10, 200, -10, 10, 201)
        label = r"log"
        per_column_log = {**per_column_default}
        per_column_log[0]["ncontours"] = 24
        per_column_log[1]["ncontours"] = 48
        per_column_log[2]["ncontours"] = 48
        log_plot_input = prepare_complex_plot_input(lambda z: logC(z), *re_im, label)
        _ = complex_plot_contours(log_plot_input, per_column_log)
    """
    return np.log(np.abs(z)) - 1j * np.where(
        np.angle(z) < 0, np.angle(z) + 2 * np.pi, np.angle(z)
    )


def extract_phase(f, jump=1.5):
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
    >>> from dispersionrelations.utils import extract_phase
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
        from dispersionrelations.utils import extract_phase
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


def conformal_variable(s, sE, sL):
    r"""
    Conformal variable, as defined in e.g. :cite:`Heuser:2024biq`.

    Parameters
    ----------
    s : array_like
        A complex number or sequence of complex numbers.
    sE : float
        Some conveniently chosen expansion point.
    sL : float
        The location of the closest branch point of the LHC.

    Returns
    -------
    ω : array_like
        The same shape as input `s`.

    Notes
    -----

    The conformal variable is defined as

    .. math:: \omega(s, s_E, s_L) = \frac{\sqrt{s - s_L} - \sqrt{s_E - s_L}}{\sqrt{s - s_L} + \sqrt{s_E - s_L}}.
    """
    sqrt_A = np.sqrt(s - sL + 0j)
    sqrt_B = np.sqrt(sE - sL + 0j)
    return (sqrt_A - sqrt_B) / (sqrt_A + sqrt_B)


def chi2(fit_data, observed_data, covariance_matrix, weights=None):
    r"""
    :math:`\chi^2`, computed using modelled and observed data.

    Parameters
    ----------
    fit_data : array_like
        Data obtained from a model, :math:`f_i \in \{f_1, f_2, \dots, f_n\}`.
    observed_data : array_like
        Data obtained from an experiment, :math:`o_i \in \{o_1, o_2, \dots, o_n\}`.
    covariance_matrix : array_like
        Covariance matrix of the experimental data, :math:`C \in \mathbb{R}^{n\times n}`.
    weights : array_like
        Weight matrix, used to manually attenuate parts of the data, :math:`W \in \mathbb{R}^{n\times n}`.

    Returns
    -------
    s : float
        The total :math:`\chi^2`.

    Notes
    -----
    The :math:`\chi^2` is calculated using

    .. math:: \chi^2 = \sum_i \sum_j (f_i-o_i)(C^{-1}_{ij}W_{ij})(f_j-o_j).

    See also
    --------
    dispersionrelations.utils.chi2_vector : for individual contributions.
    """
    if weights is not None:
        covariance_matrix = covariance_matrix / weights
    difference = fit_data - observed_data
    return difference @ np.linalg.inv(covariance_matrix) @ difference


def chi2_with_inverse(fit_data, observed_data, inverse_covariance_matrix, weights=None):
    r"""
    :math:`\chi^2`, computed using modelled and observed data.
    This version of the function accepts the inverse of the covariance matrix
    as a parameter and therefore gains speed by avoiding matrix inversion.

    Parameters
    ----------
    fit_data : array_like
        Data obtained from a model, :math:`f_i \in \{f_1, f_2, \dots, f_n\}`.
    observed_data : array_like
        Data obtained from an experiment, :math:`o_i \in \{o_1, o_2, \dots, o_n\}`.
    inverse_covariance_matrix : array_like
        Inverted covariance matrix of the experimental data, :math:`I \in \mathbb{R}^{n\times n}`.
    weights : array_like
        Weight matrix, used to manually attenuate parts of the data, :math:`W \in \mathbb{R}^{n\times n}`.

    Returns
    -------
    s : float
        The total :math:`\chi^2`.

    Notes
    -----
    The :math:`\chi^2` is calculated using

    .. math:: \chi^2 = \sum_i \sum_j (f_i-o_i)(I_{ij}W_{ij})(f_j-o_j).

    See also
    --------
    dispersionrelations.utils.chi2_vector_with_inverse : for individual contributions.
    """
    if weights is not None:
        inverse_covariance_matrix = inverse_covariance_matrix * weights
    difference = fit_data - observed_data
    return difference @ inverse_covariance_matrix @ difference


def chi2_vector(fit_data, observed_data, covariance_matrix, weights=None):
    r"""
    :math:`\chi^2` vector.

    Parameters
    ----------
    fit_data : array_like
        Data obtained from a model, :math:`f_i \in \{f_1, f_2, \dots, f_n\}`.
    observed_data : array_like
        Data obtained from an experiment, :math:`o_i \in \{o_1, o_2, \dots, o_n\}`.
    covariance_matrix : array_like
        Covariance matrix of the experimental data, :math:`C \in \mathbb{R}^{n\times n}`.
    weights : array_like
        Weight matrix, used to manually attenuate parts of the data, :math:`W \in \mathbb{R}^{n\times n}`.

    Returns
    -------
    v : array_like
        A vector with shape (n,) of individual :math:`\chi^2` contributions.

    Notes
    -----
    The :math:`\chi^2` vector is built using

    .. math:: v_i = \sum_j (f_i-o_i)(C^{-1}_{ij}W_{ij})(f_j-o_j).

    Warning
    -------
    When the covariance matrix is non-diagonal, the individual components of this
    vector do not have a statistical meaning, since the observations are correlated.
    Use with caution!

    Examples
    --------
    >>> import numpy as np
    >>> from dispersionrelations.utils import chi2_vector
    >>> np.random.seed(137)
    >>> n = 10
    >>> x_f = np.linspace(1, 2, n)
    >>> x_o = x_f + 0.1 * np.random.rand(n)
    >>> x_e = 0.1 * (np.random.rand(n) + 1) / 2
    >>> x_C = np.diag(x_e)**2
    >>> print(chi2_vector(x_f, x_o, x_C))
    [0.89280398 0.01007333 0.62584069 0.67718574 0.32026806 0.99981995 0.91740446 0.01387227 2.33635045 0.18841755]

    .. plot::

        import numpy as np
        from dispersionrelations.utils import chi2_vector
        import matplotlib.pyplot as plt

        np.random.seed(137)
        n = 10
        x_f = np.linspace(1, 2, n)
        x_o = x_f + 0.1 * np.random.rand(n)
        x_e = 0.1 * (np.random.rand(n) + 1) / 2
        x_C = np.diag(x_e)**2
        x_chi2_vector = chi2_vector(x_f, x_o, x_C)
        plt.errorbar(np.arange(n), x_o, yerr=x_e, fmt="o", markersize=0, capsize=2, label="observed_data", zorder=2)
        plt.plot(np.arange(n), x_f, label="fit_data", zorder=2)
        plt.fill_between(np.arange(n), 0.1 * x_chi2_vector, 0, color='tab:red', label=r"$\chi^2/10$", zorder=3)
        plt.legend()

    See also
    --------
    dispersionrelations.utils.chi2
    """
    if weights is not None:
        covariance_matrix = covariance_matrix / weights
    difference = fit_data - observed_data
    return difference * (np.linalg.inv(covariance_matrix) @ difference)


def chi2_vector_with_inverse(
    fit_data, observed_data, inverse_covariance_matrix, weights=None
):
    r"""
    :math:`\chi^2`, computed using modelled and observed data.
    This version of the function accepts the inverse of the covariance matrix
    as a parameter and therefore gains speed by avoiding matrix inversion.

    Parameters
    ----------
    fit_data : array_like
        Data obtained from a model, :math:`f_i \in \{f_1, f_2, \dots, f_n\}`.
    observed_data : array_like
        Data obtained from an experiment, :math:`o_i \in \{o_1, o_2, \dots, o_n\}`.
    inverse_covariance_matrix : array_like
        Inverted covariance matrix of the experimental data, :math:`I \in \mathbb{R}^{n\times n}`.
    weights : array_like
        Weight matrix, used to manually attenuate parts of the data, :math:`W \in \mathbb{R}^{n\times n}`.

    Returns
    -------
    v : array_like
        A vector with shape (n,) of individual :math:`\chi^2` contributions.

    Notes
    -----
    The :math:`\chi^2` vector is built using

    .. math:: v_i = \sum_j (f_i-o_i)(I_{ij}W_{ij})(f_j-o_j).

    See also
    --------
    dispersionrelations.utils.chi2_with_inverse
    """
    if weights is not None:
        inverse_covariance_matrix = inverse_covariance_matrix * weights
    difference = fit_data - observed_data
    return difference * (inverse_covariance_matrix @ difference)


def chi2_reduced(
    fit_data, observed_data, covariance_matrix, weights=None, number_of_parameters=0
):
    r"""
    A reduced :math:`\chi^2`, computed using modelled and observed data.

    Parameters
    ----------
    fit_data : array_like
        Data obtained from a model, :math:`f_i \in \{f_1, f_2, \dots, f_n\}`.
    observed_data : array_like
        Data obtained from an experiment, :math:`o_i \in \{o_1, o_2, \dots, o_n\}`.
    covariance_matrix : array_like
        Covariance matrix of the experimental data, :math:`C \in \mathbb{R}^{n\times n}`.
    weights : array_like
        Weight matrix, used to manually attenuate parts of the data, :math:`W \in \mathbb{R}^{n\times n}`.
    number_of_parameters : int
        Number of parameters in the model, used for calculating the degrees of freedom.

    Returns
    -------
    s : float
        The reduced :math:`\chi^2`.

    Notes
    -----
    The reduced :math:`\chi^2` is calculated using

    .. math:: \bar{\chi}^2 = \frac{\chi^2}{\text{d.o.f.}} = \frac{1}{n - n_\text{par}} \sum_i \sum_j (f_i-o_i)(C^{-1}_{ij}W_{ij})(f_j-o_j).

    See also
    --------
    dispersionrelations.utils.chi2
    """
    degrees_of_freedom = len(observed_data) - number_of_parameters
    return (
        chi2(fit_data, observed_data, covariance_matrix, weights) / degrees_of_freedom
    )


def chi2_reduced_with_inverse(
    fit_data,
    observed_data,
    inverse_covariance_matrix,
    weights=None,
    number_of_parameters=0,
):
    r"""
    A reduced :math:`\chi^2`, computed using modelled and observed data.
    This version of the function accepts the inverse of the covariance matrix
    as a parameter and therefore gains speed by avoiding matrix inversion.

    Parameters
    ----------
    fit_data : array_like
        Data obtained from a model, :math:`f_i \in \{f_1, f_2, \dots, f_n\}`.
    observed_data : array_like
        Data obtained from an experiment, :math:`o_i \in \{o_1, o_2, \dots, o_n\}`.
    inverse_covariance_matrix : array_like
        Inverted covariance matrix of the experimental data, :math:`I \in \mathbb{R}^{n\times n}`.
    weights : array_like
        Weight matrix, used to manually attenuate parts of the data, :math:`W \in \mathbb{R}^{n\times n}`.
    number_of_parameters : int
        Number of parameters in the model, used for calculating the degrees of freedom.

    Returns
    -------
    s : float
        The reduced :math:`\chi^2`.

    Notes
    -----
    The reduced :math:`\chi^2` is calculated using

    .. math:: \bar{\chi}^2 = \frac{\chi^2}{\text{d.o.f.}} = \frac{1}{n - n_\text{par}} \sum_i \sum_j (f_i-o_i)(I_{ij}W_{ij})(f_j-o_j).

    See also
    --------
    dispersionrelations.utils.chi2
    """
    degrees_of_freedom = len(observed_data) - number_of_parameters
    return (
        chi2_with_inverse(fit_data, observed_data, inverse_covariance_matrix, weights)
        / degrees_of_freedom
    )
