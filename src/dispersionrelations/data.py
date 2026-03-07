import numpy as np
import scipy as sp
import warnings
import os

from dispersionrelations.utils import cite, uncite


def check_pd(mat):
    r"""
    Check if a matrix is positive definite.

    Parameters
    ----------
    mat : array-like, shape (N, N)
        The matrix to be checked.

    Returns
    -------
    bool
        True if the matrix is positive definite, False otherwise.
    """
    try:
        np.linalg.cholesky(mat)
        return True
    except np.linalg.LinAlgError:
        return False


def corr_to_cov(corr, std):
    r"""
    Convert a correlation matrix to a covariance matrix.

    Parameters
    ----------
    corr : array-like, shape (N, N)
        The correlation matrix to be converted.
    std : array-like, shape (N,)
        The standard deviations corresponding to the variables.

    Returns
    -------
    cov : array-like, shape (N, N)
        The resulting covariance matrix.

    Raises
    ------
    ValueError
        If corr is not a square matrix or if corr and std have incompatible dimensions.
    """
    corr = np.asarray(corr)
    std = np.asarray(std)
    if corr.shape[0] != corr.shape[1]:
        raise ValueError(f"corr must be a square matrix, but got shape {corr.shape}")
    if corr.shape[0] != std.shape[0]:
        raise ValueError(
            f"corr and std must have compatible dimensions, but got {corr.shape} and {std.shape}"
        )
    if std.ndim != 1:
        raise ValueError(
            f"std must be a one-dimensional array, but got shape {std.shape}"
        )
    outer = np.outer(std, std)
    cov = corr * outer
    return cov


def cov_to_corr(cov):
    r"""
    Convert a covariance matrix to a correlation matrix.

    Parameters
    ----------
    cov : array-like, shape (N, N)
        The covariance matrix to be converted.

    Returns
    -------
    corr : array-like, shape (N, N)
        The resulting correlation matrix.

    Raises
    ------
    ValueError
        If cov is not a square matrix.
    """
    cov = np.asarray(cov)
    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be a square matrix, but got shape {cov.shape}")
    corr = np.zeros(cov.shape)
    std = np.sqrt(np.diag(cov))
    outer = np.outer(std, std)
    nonzero = np.where(outer != 0)
    corr[nonzero] = cov[nonzero] / outer[nonzero]
    corr[np.isnan(corr)] = 0
    return corr


def cov_to_std(cov):
    r"""
    Convert a covariance matrix to standard deviations (square root of the diagonal elements).

    Parameters
    ----------
    cov : array-like, shape (N, N)
        The covariance matrix to be converted.

    Returns
    -------
    std : array-like, shape (N,)
        The resulting standard deviations.

    Raises
    ------
    ValueError
        If cov is not a square matrix.
    """
    cov = np.asarray(cov)
    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be a square matrix, but got shape {cov.shape}")
    return np.sqrt(np.diag(cov))


def std_to_cov(std, correlation=0.0):
    r"""
    Convert standard deviations to a covariance matrix, assuming a given correlation structure.

    Parameters
    ----------
    std : array-like, shape (N,)
        The standard deviations to be converted.
    correlation : float, optional
        The correlation coefficient to be used in the conversion. Default is 0.0 (no correlation).

    Returns
    -------
    cov : array-like, shape (N, N)
        The resulting covariance matrix.

    Raises
    ------
    ValueError
        If std is not a one-dimensional array.
    """
    std = np.asarray(std)
    if std.ndim != 1:
        raise ValueError(
            f"std must be a one-dimensional array, but got shape {std.shape}"
        )
    corr = (1.0 - correlation) * np.eye(len(std)) + correlation * np.ones(
        (len(std), len(std))
    )
    cov = np.outer(std, std) * corr
    return cov


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
    dispersionrelations.data.chi2_vector : for individual contributions.
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
    dispersionrelations.data.chi2_vector_with_inverse : for individual contributions.
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
    >>> from dispersionrelations.data import chi2_vector
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
        from dispersionrelations.data import chi2_vector
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
    dispersionrelations.data.chi2
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
    dispersionrelations.data.chi2_with_inverse
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
    dispersionrelations.data.chi2
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
    dispersionrelations.data.chi2
    """
    degrees_of_freedom = len(observed_data) - number_of_parameters
    return (
        chi2_with_inverse(fit_data, observed_data, inverse_covariance_matrix, weights)
        / degrees_of_freedom
    )


class Datablock:
    r"""
    A class to represent a dataset with x and y values, along with statistical and systematic covariance matrices.

    Parameters
    ----------
    x : array-like, shape (N,)
        The x values of the data points.
    y : array-like, shape (N,)
        The y values of the data points.
    cov_stat : array-like, shape (N, N), optional
        The initial statistical covariance matrix. Default is None, which initializes it to a zero matrix.
    cov_syst : array-like, shape (N, N), optional
        The initial systematic covariance matrix. Default is None, which initializes it to a zero matrix.
    metadata : dict, optional
        A dictionary to store additional metadata about the dataset. Default is an empty dictionary.
        Common keys include 'ref', 'header', 'footer', utilised by the :class:`Datawriter` class.

    Attributes
    ----------
    x : array-like, shape (N,)
        The x values of the data points.
    y : array-like, shape (N,)
        The y values of the data points.
    N : int
        The number of data points.
    cov_stat : array-like, shape (N, N)
        The statistical covariance matrix.
    cov_syst : array-like, shape (N, N)
        The systematic covariance matrix.
    metadata : dict
        A dictionary to store additional metadata about the dataset.
    """

    def __init__(self, x, y, xlabel, ylabel, cov_stat=None, cov_syst=None, metadata={}):
        self.x = x
        self.y = y
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.N = len(x)
        if len(y) != self.N:
            raise ValueError(
                f"x and y must have the same length, but got {len(x)} and {len(y)}"
            )
        self.cov_stat = np.zeros((self.N, self.N))
        self.cov_syst = np.zeros((self.N, self.N))

        if cov_stat is not None:
            self.add_cov_stat(cov_stat)
            self.check_cov_pd()
        if cov_syst is not None:
            self.add_cov_syst(cov_syst)
            self.check_cov_pd()

        self.metadata = metadata

    def check_cov_pd(self):
        r"""
        Check if the total covariance matrix (statistical + systematic) is positive definite.
        Display a warning if it is not.

        Returns
        -------
        None
        """
        if check_pd(self.cov_stat + self.cov_syst) == False:
            warnings.warn("The total covariance matrix is not positive definite")

    def get_std_stat(self):
        r"""
        Get the statistical standard deviations (square root of the diagonal elements of the statistical covariance matrix).

        Returns
        -------
        std_stat : array-like, shape (N,)
            The statistical standard deviations.
        """
        return cov_to_std(self.cov_stat)

    def get_std_syst(self):
        r"""
        Get the systematic standard deviations (square root of the diagonal elements of the systematic covariance matrix).

        Returns
        -------
        std_syst : array-like, shape (N,)
            The systematic standard deviations.
        """
        return cov_to_std(self.cov_syst)

    def get_std_total(self, add_in_quadrature=True):
        r"""
        Get the total standard deviations (square root of the diagonal elements of the total covariance matrix).

        Parameters
        ----------
        add_in_quadrature : bool, optional
            If True, compute the total standard deviations by adding the statistical and systematic standard deviations in quadrature.
            If False, compute the total standard deviations from the total covariance matrix. Default is True.

        Returns
        -------
        std_total : array-like, shape (N,)
            The total standard deviations.
        """
        std_stat = self.get_std_stat()
        std_syst = self.get_std_syst()
        if add_in_quadrature:
            return np.sqrt(std_stat**2 + std_syst**2)
        else:
            return std_stat + std_syst

    def add_cov_stat(self, cov_stat):
        r"""
        Add a contribution to the statistical covariance matrix.

        Parameters
        ----------
        cov_stat : array-like, shape (N, N)
            The statistical covariance matrix to be added.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If cov_stat is not of shape (N, N).
        """
        if cov_stat.shape != (self.N, self.N):
            raise ValueError(
                f"cov_stat must be of shape (N, N), but got {cov_stat.shape}"
            )
        self.cov_stat += cov_stat

    def add_cov_syst(self, cov_syst):
        r"""
        Add a contribution to the systematic covariance matrix.

        Parameters
        ----------
        cov_syst : array-like, shape (N, N)
            The systematic covariance matrix to be added.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If cov_syst is not of shape (N, N).
        """
        if cov_syst.shape != (self.N, self.N):
            raise ValueError(
                f"cov_syst must be of shape (N, N), but got {cov_syst.shape}"
            )
        self.cov_syst += cov_syst

    def add_std_stat(self, std_stat, correlation=0.0):
        r"""
        Add a contribution to the statistical covariance matrix from standard deviations.

        Parameters
        ----------
        std_stat : array-like, shape (N,) or scalar
            The standard deviations to be converted to covariance matrix.
        correlation : float, optional
            The correlation coefficient to be used in the conversion. Default is 0.0 (no correlation).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If std_stat is not a scalar or of shape (N,).
        """
        std_stat = np.asarray(std_stat)
        if std_stat.shape != (self.N,) and std_stat.shape != (1,):
            raise ValueError(
                f"std_stat must be a scalar or of shape (N,), but got {std_stat.shape}"
            )
        cov_stat = std_to_cov(std_stat, correlation=correlation)
        self.add_cov_stat(cov_stat)

    def add_rel_stat(self, rel_stat, correlation=0.0):
        r"""
        Add a contribution to the statistical covariance matrix from relative uncertainties.

        Parameters
        ----------
        rel_stat : array-like, shape (N,) or scalar
            The relative uncertainties to be converted to standard deviations.
        correlation : float, optional
            The correlation coefficient to be used in the conversion. Default is 0.0 (no correlation).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If rel_stat is not a scalar or of shape (N,).
        """
        std_stat = np.abs(rel_stat * self.y)
        self.add_std_stat(std_stat, correlation=correlation)

    def add_std_syst(self, std_syst, correlation=1.0):
        r"""
        Add a contribution to the systematic covariance matrix from standard deviations.

        Parameters
        ----------
        std_syst : array-like, shape (N,) or scalar
            The standard deviations to be converted to covariance matrix.
        correlation : float, optional
            The correlation coefficient to be used in the conversion. Default is 1.0 (full correlation).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If std_syst is not a scalar or of shape (N,).
        """
        std_syst = np.asarray(std_syst)
        if std_syst.shape != (self.N,) and std_syst.shape != (1,):
            raise ValueError(
                f"std_syst must be a scalar or of shape (N,), but got {std_syst.shape}"
            )
        cov_syst = std_to_cov(std_syst, correlation=correlation)
        self.add_cov_syst(cov_syst)

    def add_rel_syst(self, rel_syst, correlation=1.0):
        r"""
        Add a contribution to the systematic covariance matrix from relative uncertainties.

        Parameters
        ----------
        rel_syst : array-like, shape (N,) or scalar
            The relative uncertainties to be converted to standard deviations.
        correlation : float, optional
            The correlation coefficient to be used in the conversion. Default is 1.0 (full correlation).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If rel_syst is not a scalar or of shape (N,).
        """
        std_syst = np.abs(rel_syst * self.y)
        self.add_std_syst(std_syst, correlation=correlation)

    def add_external_calibration(self, c, c_std, correlation=1.0, new_ylabel=None):
        r"""
        Apply an external calibration factor to the data and propagate its uncertainty.

        Parameters
        ----------
        c : float
            The calibration factor to be applied to the data.
        c_std : float
            The uncertainty of the calibration factor.
        correlation : float, optional
            The correlation coefficient to be used in the propagation of the calibration uncertainty. Default is 1.0 (full correlation).
        new_ylabel : str, optional
            The new label for the y-axis after calibration. If None, the label remains unchanged.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If c or c_std is not a scalar.

        Notes
        -----
        For x-dependent calibration factors, use `rescale_y` instead.
        """
        if np.asarray(c).shape != () or np.asarray(c_std).shape != ():
            raise ValueError("c and c_std must be scalars")
        self.y = c * self.y
        self.cov_stat *= c**2
        self.cov_syst *= c**2
        if new_ylabel is not None:
            self.ylabel = new_ylabel
        if c_std != 0:
            self.add_rel_syst(c_std / c, correlation=correlation)
            self.check_cov_pd()

    def rescale_y(
        self,
        factor,
        factor_std=None,
        stat_or_syst="syst",
        correlation=0.0,
        new_ylabel=None,
    ):
        r"""
        Rescale the y values by a given factor without propagating the uncertainty.

        Parameters
        ----------
        factor : array-like, shape (N,)
            The factor by which to rescale the y values. If an array is provided, it must be of shape (N,).
        factor_std : array-like, shape (N,)
            The uncertainty of the rescaling factor (not propagated).
        stat_or_syst : str, optional
            Specify whether the uncertainty is 'stat' (statistical) or 'syst' (systematic). Default is 'syst'.
        correlation : float, optional
            The correlation coefficient to be used for the uncertainty propagation. Default is 0.0 (no correlation).
        new_ylabel : str, optional
            The new label for the y-axis after rescaling. If None, the label remains unchanged

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If factor is not a scalar or of shape (N,).

        Notes
        -----
        For a global rescaling, use `add_external_calibration` instead.
        """
        factor = np.asarray(factor)
        factor_std = (
            np.asarray(factor_std) if factor_std is not None else np.zeros(factor.shape)
        )
        if factor.shape != (self.N,):
            raise ValueError(
                f"factor must be a scalar or of shape (N,), but got {factor.shape}"
            )
        if factor_std.shape != (self.N,):
            raise ValueError(
                f"factor_std must be a scalar or of shape (N,), but got {factor_std.shape}"
            )
        if stat_or_syst not in ["stat", "syst"]:
            raise ValueError("stat_or_syst must be 'stat' or 'syst'")
        self.y = factor * self.y
        factor_mat = np.outer(factor, factor)
        self.cov_stat = factor_mat * self.cov_stat
        self.cov_syst = factor_mat * self.cov_syst
        if new_ylabel is not None:
            self.ylabel = new_ylabel
        if np.any(factor_std != 0):
            rel_syst = np.abs(factor_std / factor)
            if stat_or_syst == "stat":
                self.add_rel_stat(rel_syst, correlation=correlation)
            elif stat_or_syst == "syst":
                self.add_rel_syst(rel_syst, correlation=correlation)
            self.check_cov_pd()

    def filter_data(self, mask):
        r"""
        Filter the data based on a boolean mask.

        Parameters
        ----------
        mask : array-like, shape (N,)
            A boolean array indicating which data points to keep.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If mask is not of length N.
        """
        mask = np.asarray(mask)
        if mask.shape != (self.N,):
            raise ValueError(
                f"The mask must be of length N={self.N}, but got {mask.shape}"
            )
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.cov_stat = self.cov_stat[np.ix_(mask, mask)]
        self.cov_syst = self.cov_syst[np.ix_(mask, mask)]
        self.N = len(self.x)
        self.check_cov_pd()

    def filter_xrange(self, x_min=None, x_max=None):
        r"""
        Filter the data based on x range.

        Parameters
        ----------
        x_min : float, optional
            The minimum x value to keep. If None, no lower bound is applied.
        x_max : float, optional
            The maximum x value to keep. If None, no upper bound is applied.

        Returns
        -------
        None
        """
        mask_lower = (
            self.x >= x_min if x_min is not None else np.ones(self.N, dtype=bool)
        )
        mask_upper = (
            self.x <= x_max if x_max is not None else np.ones(self.N, dtype=bool)
        )
        self.filter_data(mask_lower * mask_upper)

    def filter_yrange(self, y_min=None, y_max=None):
        r"""
        Filter the data based on y range.

        Parameters
        ----------
        y_min : float, optional
            The minimum y value to keep. If None, no lower bound is applied.
        y_max : float, optional
            The maximum y value to keep. If None, no upper bound is applied.

        Returns
        -------
        None
        """
        mask_lower = (
            self.y >= y_min if y_min is not None else np.ones(self.N, dtype=bool)
        )
        mask_upper = (
            self.y <= y_max if y_max is not None else np.ones(self.N, dtype=bool)
        )
        self.filter_data(mask_lower * mask_upper)

    def remove_datapoints(self, indices):
        r"""
        Remove data points based on their indices.

        Parameters
        ----------
        indices : array-like, shape (M,)
            An array of indices of the data points to be removed.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any index is out of bounds.
        """
        indices = np.asarray(indices)
        if np.any(indices < 0) or np.any(indices >= self.N):
            raise ValueError(
                f"All indices must be in the range [0, {self.N}), but got {indices}"
            )
        mask = np.ones(self.N, dtype=bool)
        mask[indices] = False
        self.filter_data(mask)

    def downsample_by(self, factor):
        r"""
        Downsample the data by a given factor.

        Parameters
        ----------
        factor : int
            The downsampling factor. Must be a positive integer.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If factor is not a positive integer.
        """
        if not isinstance(factor, int) or factor <= 0:
            raise ValueError("factor must be a positive integer")
        mask = np.zeros(self.N, dtype=bool)
        mask[::factor] = True
        self.filter_data(mask)

    def downsample_to(self, N_points):
        r"""
        Downsample the data to a given number of points.

        Parameters
        ----------
        N_points : int
            The desired number of data points after downsampling. Must be a positive integer less than or equal to N.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If N_points is not a positive integer or greater than N.
        """
        raise NotImplementedError("downsample_to is not implemented yet")

    def interpolate(self, new_x):
        r"""
        Interpolate the data to new x values using linear interpolation.

        Parameters
        ----------
        new_x : array-like, shape (M,)
            The new x values to interpolate to.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If new_x is not a one-dimensional array.
        """
        raise NotImplementedError("interpolate is not implemented yet")

    def __str__(self):
        result = f"Datablock with {self.N} points(s): {self.xlabel} vs {self.ylabel}."
        if "ref" in self.metadata:
            result += f" Reference: {self.metadata['ref']}."
        return result

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        r"""
        Add two Datablock objects.

        Parameters
        ----------
        other : Datablock
            The other Datablock to be added.

        Returns
        -------
        Datablock
            A new Datablock containing the combined data.

        Raises
        ------
        ValueError
            If other is not a Datablock.

        Notes
        -----
        Metadata and labels are not fully preserved in the addition.
        """
        if not isinstance(other, Datablock):
            raise ValueError("Can only add Datablock to Datablock")

        new_metadata = {}
        if "ref" in self.metadata:
            new_metadata["ref"] = self.metadata["ref"]
        if "ref" in other.metadata:
            if "ref" in new_metadata:
                new_metadata["ref"] += ", " + other.metadata["ref"]
            else:
                new_metadata["ref"] = other.metadata["ref"]
        warnings.warn("Metadata and labels are not preserved in Datablock addition")

        new_x = np.concatenate((self.x, other.x))
        new_y = np.concatenate((self.y, other.y))
        new_cov_stat = sp.linalg.block_diag(self.cov_stat, other.cov_stat)
        new_cov_syst = sp.linalg.block_diag(self.cov_syst, other.cov_syst)
        return Datablock(
            new_x,
            new_y,
            "x",
            "y",
            cov_stat=new_cov_stat,
            cov_syst=new_cov_syst,
            metadata=new_metadata,
        )

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)


class Datawriter:
    r"""
    A class for writing Datablock data to files.

    Parameters
    ----------
    datablock : Datablock
        The Datablock object containing the data to be written.
    file_prefix : str
        The prefix for the output files.
    folder_path : str, optional
        The folder path where the files will be saved. Default is the current directory.
        If the folder does not exist, it will be created.
    quiet : bool, optional
        If True, suppresses output messages. Default is False.
    format : dict, optional
        A dictionary specifying the format for saving the data.
    """

    def __init__(
        self,
        datablock,
        file_prefix,
        folder_path="",
        quiet=False,
        format=dict(fmt="% .11e", delimiter="\t"),
    ):
        self.datablock = datablock
        self.folder = folder_path
        self.file_prefix = file_prefix
        self.header = ""
        self.footer = ""
        self.quiet = quiet
        self.format = format
        if "header" in self.datablock.metadata:
            self.header += self.datablock.metadata["header"]
        if "footer" in self.datablock.metadata:
            self.footer += self.datablock.metadata["footer"]
        os.makedirs(self.folder, exist_ok=True)

    def write_data(self, **filekwargs):
        r"""
        Write the data to a .dat file.
        The file will contain columns for x, y, statistical uncertainty, and systematic uncertainty.

        Parameters
        ----------
        filekwargs : dict
            Additional keyword arguments to be passed to np.savetxt.

        Returns
        -------
        None
        """
        data = np.column_stack(
            (
                self.datablock.x,
                self.datablock.y,
                np.sqrt(np.diag(self.datablock.cov_stat)),
                np.sqrt(np.diag(self.datablock.cov_syst)),
            )
        )
        np.savetxt(
            os.path.join(self.folder, f"{self.file_prefix}.dat"),
            data,
            header=self.header
            + "\n"
            + f"Columns {data.shape}: {self.datablock.xlabel}, {self.datablock.ylabel}, stat_unc, syst_unc",
            footer=self.footer,
            **self.format,
            **filekwargs,
        )
        if not self.quiet:
            print(
                f'"{os.path.join(self.folder, f"{self.file_prefix}.dat")}" created {data.shape}.'
            )

    def write_cov_stat(self, **filekwargs):
        r"""
        Write the statistical covariance matrix to a .dat file.
        The file will contain the statistical covariance matrix.

        Parameters
        ----------
        filekwargs : dict
            Additional keyword arguments to be passed to np.savetxt.

        Returns
        -------
        None
        """
        np.savetxt(
            os.path.join(self.folder, f"{self.file_prefix}_COV_STAT.dat"),
            self.datablock.cov_stat,
            header=self.header
            + "\n"
            + f"Statistical covariance matrix {self.datablock.cov_stat.shape}",
            footer=self.footer,
            **self.format,
            **filekwargs,
        )
        if not self.quiet:
            print(
                f'"{os.path.join(self.folder, f"{self.file_prefix}_COV_STAT.dat")}" created {self.datablock.cov_stat.shape}.'
            )

    def write_cov_syst(self, **filekwargs):
        r"""
        Write the systematic covariance matrix to a .dat file.
        The file will contain the systematic covariance matrix.

        Parameters
        ----------
        filekwargs : dict
            Additional keyword arguments to be passed to np.savetxt.

        Returns
        -------
        None
        """
        np.savetxt(
            os.path.join(self.folder, f"{self.file_prefix}_COV_SYST.dat"),
            self.datablock.cov_syst,
            header=self.header
            + "\n"
            + f"Systematic covariance matrix {self.datablock.cov_syst.shape}",
            footer=self.footer,
            **self.format,
            **filekwargs,
        )
        if not self.quiet:
            print(
                f'"{os.path.join(self.folder, f"{self.file_prefix}_COV_SYST.dat")}" created {self.datablock.cov_syst.shape}.'
            )

    def write_all(self, **filekwargs):
        r"""
        Write all data to .dat files.
        This includes the main data file, the statistical covariance matrix file, and the systematic covariance matrix file.

        Parameters
        ----------
        filekwargs : dict
            Additional keyword arguments to be passed to np.savetxt.
            This argument is shared between all three writing functions.
            For a finer control, call the individual writing functions instead.

        Returns
        -------
        None
        """
        if not self.quiet:
            print(f"- {self.file_prefix} - - - - - - - - - - - - - -")
        self.write_data(**filekwargs)
        self.write_cov_stat(**filekwargs)
        self.write_cov_syst(**filekwargs)


class Datareader:
    r"""
    A class for reading Datablock data from files.

    Parameters
    ----------
    folder : str
        The folder path where the files are located.
    prefix : str
        The prefix for the input files.
    """

    def __init__(self, folder, prefix):
        self.folder = folder
        self.prefix = prefix
        self.file_data = os.path.join(folder, f"{prefix}.dat")
        self.file_cov_stat = os.path.join(folder, f"{prefix}_COV_STAT.dat")
        self.file_cov_syst = os.path.join(folder, f"{prefix}_COV_SYST.dat")

    def read_all(self, **filekwargs):
        r"""
        Read all data from .dat files and return a Datablock object.
        This includes the main data file, the statistical covariance matrix file, and the systematic covariance matrix file.

        Parameters
        ----------
        filekwargs : dict
            Additional keyword arguments to be passed to np.loadtxt.

        Returns
        -------
        Datablock
            A Datablock object containing the data read from the files.

        Raises
        ------
        ValueError
            If the shapes of the covariance matrices do not match the expected dimensions.
            If the uncertainties (squared) do not match the diagonals of the covariance matrices.
        """
        # Load data from file
        data = np.loadtxt(self.file_data, **filekwargs)

        # Load comments from file
        comments = []
        with open(self.file_data, encoding="utf-8") as f:
            for line in f:
                if line.lstrip().startswith("#"):
                    comments.append(line.lstrip()[1:].lstrip().rstrip("\n"))

        # Fill labels and metadata from comments
        xlabel = "x"
        ylabel = "y"
        metadata = {}
        try:
            metadata["ref"] = uncite(comments[0])
        except:
            warnings.warn("Could not extract reference from data file.")
        try:
            columns = ": ".join(comments[-1].split(": ")[1:]).split(", ")
            xlabel = columns[0]
            ylabel = columns[1]
        except:
            warnings.warn("Could not extract column names from data file.")

        # Prepare data and create Datablock
        only_one_datapoint = len(data.shape) == 1
        if only_one_datapoint:
            data = data.reshape(1, -1)
        x = data[..., 0]
        y = data[..., 1]
        stat_unc = data[..., 2]
        syst_unc = data[..., 3]
        cov_stat = np.loadtxt(self.file_cov_stat, **filekwargs)
        cov_syst = np.loadtxt(self.file_cov_syst, **filekwargs)
        if only_one_datapoint:
            cov_stat = cov_stat.reshape((1, 1))
            cov_syst = cov_syst.reshape((1, 1))
        if cov_stat.shape != (len(x), len(x)):
            raise ValueError(
                f"Statistical covariance matrix must be of shape {(len(x), len(x))}, but got {cov_stat.shape}"
            )
        if cov_syst.shape != (len(x), len(x)):
            raise ValueError(
                f"Systematic covariance matrix must be of shape {(len(x), len(x))}, but got {cov_syst.shape}"
            )
        if not np.allclose(stat_unc, np.sqrt(np.diag(cov_stat))):
            raise ValueError(
                "Statistical uncertainties (squared) do not match the diagonal of the statistical covariance matrix"
            )
        if not np.allclose(syst_unc, np.sqrt(np.diag(cov_syst))):
            raise ValueError(
                "Systematic uncertainties (squared) do not match the diagonal of the systematic covariance matrix"
            )
        return Datablock(
            x,
            y,
            xlabel,
            ylabel,
            cov_stat=cov_stat,
            cov_syst=cov_syst,
            metadata=metadata,
        )
