import numpy as np


def Kallen(x, y, z):
    r"""
    The Källén triangle function.

    Parameters
    ----------
    x, y, z : array_like
        A complex number or sequence of complex numbers.

    Returns
    -------
    λ : array_like
        The broadcasted shape from `x`, `y`, `z`.

    Notes
    -----

    The function is defined as :cite:`Kallen:1964lxa`

    .. math:: \lambda(x, y, z) = x^2 + y^2 + z^2 - 2 (xy + yz + zx) .

    An alternate form can be derived for squared inputs

    .. math:: \lambda(q^2, M_1^2, M_2^2) = (q^2 - (M_1 + M_2)^2)(q^2 - (M_1 - M_2)^2) .
    """
    return x**2 + y**2 + z**2 - 2 * (x * y + y * z + z * x)


def momentum_cms(s, M1, M2):
    r"""
    The center-of-mass momentum of a two-body system.

    Parameters
    ----------
    s : array_like
        Four-momentum squared of the two-body system.
    M1 : float, complex
        Mass of the first particle.
    M2 : float, complex
        Mass of the second particle.

    Returns
    -------
    q : array_like
        The same shape as input `s`.

    Notes
    -----
    The CMS momentum is defined as

    .. math:: q(s, M_1, M_2) = \frac{1}{2}\sqrt{\frac{\lambda(s, M_1^2, M_2^2)}{s}},

    where :math:`\lambda(s, M_1^2, M_2^2)` is the Källén triangle function.

    Examples
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from dispersionrelations.kinematics import momentum_cms
    >>> M1 = 0.3
    >>> M2 = 0.5
    >>> E_plot = np.linspace(M1+M2, 2, 500)
    >>> s_plot = E_plot**2
    >>> plt.plot(E_plot, momentum_cms(s_plot, M1, M2))

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from dispersionrelations.kinematics import momentum_cms
        M1 = 0.3
        M2 = 0.5
        E_plot = np.linspace(M1+M2, 2, 500)
        s_plot = E_plot**2
        plt.plot(E_plot, momentum_cms(s_plot, M1, M2))

    See Also
    --------
    dispersionrelations.kinematics.Kallen
    """
    return np.sqrt(Kallen(s, M1**2, M2**2) / s + 0j) / 2


def phase_space_twobody(s, M1, M2):
    r"""
    Phase space function for a two-particle system.

    Parameters
    ----------
    s : array_like
        Four-momentum squared of the two-body system.
    M1 : float, complex
        Mass of the first particle.
    M2 : float, complex
        Mass of the second particle.

    Returns
    -------
    ρ : array_like
        The same shape as input `s`.

    Notes
    -----
    The phase space function is defined as

    .. math:: \rho(s, M_1, M_2) = \frac{1}{16\pi}\frac{2q}{\sqrt{s}} = \frac{1}{16\pi}\frac{\sqrt{\lambda(s, M_1, M_2)}}{s},

    where :math:`q` is the CMS momentum of the particles.

    Examples
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from dispersionrelations.kinematics import phase_space_twobody
    >>> M1 = 0.3
    >>> M2 = 0.5
    >>> E_plot = np.linspace(M1+M2, 2, 500)
    >>> s_plot = E_plot**2
    >>> plt.plot(E_plot, phase_space_twobody(s_plot, M1, M2))

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from dispersionrelations.kinematics import phase_space_twobody
        M1 = 0.3
        M2 = 0.5
        E_plot = np.linspace(M1+M2, 2, 500)
        s_plot = E_plot**2
        plt.plot(E_plot, phase_space_twobody(s_plot, M1, M2))

    See Also
    --------
    dispersionrelations.kinematics.momentum_cms, dispersionrelations.kinematics.Kallen
    """
    return 1 / (16 * np.pi) * np.sqrt(Kallen(s, M1**2, M2**2) / s**2 + 0j)
