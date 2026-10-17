# ┌──────────────────────────────────────────────────────────┐
# │                         Typing                           │
# └──────────────────────────────────────────────────────────┘
from __future__ import annotations  # for type hints in python==3.9
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

# ┌──────────────────────────────────────────────────────────┐
# │                  CONFORMAL VARIABLES                     │
# └──────────────────────────────────────────────────────────┘

def conformal_z(
    s: NDArray | complex,
    sp: float,
    s0: float,
) -> NDArray | complex:
    r"""
    Conformal variable, as defined in e.g. :cite:`Balz:2025auk`.

    Parameters
    ----------
    s : array_like
        A complex number or sequence of complex numbers.
    sp : float
        Position of the first threshold.
    s0 : float
        The value of s mapped to z=0.

    Returns
    -------
    z : array_like
        The same shape as input `s`.

    Raises
    ------
    ValueError
        If `s0` is not less than `sp`.

    Notes
    -----

    The conformal variable is defined as

    .. math:: z(s, s_+, s_0) = \frac{\sqrt{s_+ - s} - \sqrt{s_+ - s_0}}{\sqrt{s_+ - s} + \sqrt{s_+ - s_0}}.
    """
    if s0 >= sp:
        raise ValueError(f"s0={s0} must be less than sp={sp}")
    sqrt_A = np.sqrt(sp - s + 0j)
    sqrt_B = np.sqrt(sp - s0 + 0j)
    return (sqrt_A - sqrt_B) / (sqrt_A + sqrt_B)
