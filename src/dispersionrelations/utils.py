import os
import re
import json
import warnings
from datetime import datetime
import warnings

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
        from dispersionrelations.utils import sqrt_custom_branch_cut
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


def sqrtRHC(z, sheet=1):
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
    return sqrt_custom_branch_cut(z, t_bc=0, sheet=sheet)


def sqrtLHC(z, sheet=1):
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
    return sqrt_custom_branch_cut(z, t_bc=np.pi, sheet=sheet)


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


def cite(ref):
    r"""
    Produces a LaTeX citation command.

    Parameters
    ----------
    ref : str
        The reference key.

    Returns
    -------
    citation : str
        The LaTeX citation command :code:`\cite{ref}`.

    See Also
    --------
    uncite : Inverse operation.
    """
    return r"\cite{" + ref + "}"


def uncite(citation):
    r"""
    Inverse operation of :func:`cite`.
    Extracts the reference key from a LaTeX citation command.

    Parameters
    ----------
    citation : str
        The LaTeX citation command :code:`\cite{ref}`.

    Returns
    -------
    ref : str
        The reference key.

    Raises
    ------
    ValueError
        If the input is not in the correct format.

    See also
    --------
    cite : Forward operation.
    """
    if citation[:6] != r"\cite{" or citation[-1] != "}":
        raise ValueError(f"Invalid citation format: {citation}")
    result = citation[6:-1]
    if result == "":
        warnings.warn("Empty citation key.")
    return result


def save_to_json(data, filename : str = None, folder : str = "", metadata : dict = {}):
    r"""
    Saves a dictionary to a JSON file.

    Parameters
    ----------
    data : dict
        The data to be saved.
    filename : str
        The name of the file to save the data to. If None, the current date and time will be used.
    folder : str
        The folder to save the file in. Default is the current folder.
    metadata : dict
        Additional metadata to be included in the JSON file. Default is an empty dictionary.

    Returns
    -------
    None
    """

    if filename is None:
        filename = datetime.now().strftime("%Y_%m%d_%H%M%S")

    if os.path.exists(os.path.join(folder, filename + ".json")):
        warnings.warn(f"File {filename} already exists in folder {folder}. Renaming the old file to avoid overwriting.")
        os.rename(os.path.join(folder, filename + ".json"), os.path.join(folder, filename + "_old.json"))

    with open(os.path.join(folder, filename + ".json"), "w") as f:
        json.dump({**data, **metadata}, f, indent=4, ensure_ascii=False)


def load_from_json(filename : str, folder : str = ""):
    r"""
    Loads a dictionary from a JSON file.

    Parameters
    ----------
    filename : str
        The name of the file to load the data from.
    folder : str
        The folder to load the file from. Default is the current folder.

    Returns
    -------
    data : dict
        The loaded data.
    """
    with open(os.path.join(folder, filename + ".json"), "r") as f:
        return json.load(f)




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