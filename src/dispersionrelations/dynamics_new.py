r"""
Composition-based rewrite of the channel and particle classes in :mod:`dispersionrelations.dynamics`.

Temporary module: it lives alongside the old classes until the refactor is done.
"""

# ┌──────────────────────────────────────────────────────────┐
# │                         Typing                           │
# └──────────────────────────────────────────────────────────┘
from __future__ import annotations  # for type hints in python==3.9

from functools import partial
from typing import Literal
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
# │                    Internal imports                      │
# └──────────────────────────────────────────────────────────┘
from dispersionrelations.complex import Complex2RS
from dispersionrelations.interpolation import ComplexInterpolator
from dispersionrelations.kinematics import phase_space_twobody
from dispersionrelations.integrals import (
    DispersionIntegralRHC,
    disp_int_to_complex_2RS,
)
from dispersionrelations.dynamics import VertexFunction, TamingFactor

# ┌──────────────────────────────────────────────────────────┐
# │                        DYNAMICS                          │
# └──────────────────────────────────────────────────────────┘


class Channel:
    def __init__(self, threshold):
        self.threshold = threshold
        self.loop = Complex2RS()
        self.phase_space = None
        self.vertex_function_squared: VertexFunction | None = None
        self.taming_factor_squared: TamingFactor | None = None

    def vertex_function(
        self,
        s: NDArray | complex,
    ) -> NDArray | complex:
        if self.vertex_function_squared is None:
            raise NotImplementedError("Vertex function is not implemented")
        return np.sqrt(self.vertex_function_squared(s) + 0j)

    def taming_factor(
        self,
        s: NDArray | complex,
    ) -> NDArray | complex:
        if self.taming_factor_squared is None:
            raise NotImplementedError("Taming factor is not implemented")
        return np.sqrt(self.taming_factor_squared(s) + 0j)

    def add_dispersive_loop(
        self,
        loop_di: DispersionIntegralRHC,
    ) -> None:
        disp_int_to_complex_2RS(loop_di, self.loop)

    def add_precomputed_loop(
        self,
        loop1: ComplexInterpolator | None = None,
        loop2: ComplexInterpolator | None = None,
        loop_disc: ComplexInterpolator | None = None,
        index: int | None = 0,
    ) -> None:
        if loop1 is not None:
            self.loop.f_1.add_function(loop1, index)
        if loop2 is not None:
            self.loop.f_2.add_function(loop2, index)
        if loop_disc is not None:
            self.loop.f_disc.add_function(loop_disc, index)


class StableTwoBodyChannel(Channel):
    def __init__(
        self,
        m1: float,
        m2: float,
        vertex_function_squared: VertexFunction,
        taming_factor_squared: TamingFactor,
        build_dispersively: bool = True,
    ):
        self.m1 = m1
        self.m2 = m2
        super().__init__(threshold=(m1 + m2) ** 2)
        self.phase_space = partial(
            phase_space_twobody,
            m1_squared=m1**2,
            m2_squared=m2**2,
        )
        self.vertex_function_squared = partial(
            vertex_function_squared,
            m1_squared=m1**2,
            m2_squared=m2**2,
        )
        self.taming_factor_squared = partial(taming_factor_squared)
        if build_dispersively:
            self.add_dispersive_loop(
                DispersionIntegralRHC(
                    integrand=lambda s: (
                        self.phase_space(s)
                        * self.vertex_function_squared(s)
                        * self.taming_factor_squared(s)
                    ),
                    threshold=self.threshold,
                )
            )


class UnstableParticle:
    def __init__(
        self,
        decay_channel: Channel,
        m_squared: float | None = None,
        g_squared: float | None = None,
    ):
        self.decay_channel = decay_channel
        self.m_squared = m_squared
        self.g_squared = g_squared

    def set_pole_location(
        self,
        s_p: complex,
        pole_sheet: Literal[1, 2] = 2,
    ) -> None:
        loop_at_the_pole = self.decay_channel.loop(s_p, sheet=pole_sheet)
        g_squared = -np.imag(s_p) / np.imag(loop_at_the_pole)
        m_squared = np.real(s_p) + g_squared * np.real(loop_at_the_pole)
        if g_squared < 0:
            raise ValueError(
                f"g_squared={g_squared} should be positive. Check the pole position or the sheet."
            )
        if m_squared < 0:
            raise ValueError(
                f"m_squared={m_squared} should be positive. Check the pole position or the sheet."
            )
        self.g_squared = g_squared
        self.m_squared = m_squared

    def propagator(
        self,
        s: NDArray | complex,
        sheet: Literal[1, 2] = 1,
    ) -> NDArray | complex:
        if self.g_squared is None or self.m_squared is None:
            raise RuntimeError("The bare parameter values have not been initialized.")
        return 1 / (
            s - self.m_squared + self.g_squared * self.decay_channel.loop(s, sheet)
        )

    def spectral_function(
        self,
        s: NDArray | complex,
    ) -> NDArray | complex:
        return (
            1
            / (2j * np.pi)
            * self.propagator(s, sheet=1)
            * self.propagator(s, sheet=2)
            * self.g_squared
            * self.decay_channel.loop.discontinuity(s)
        )

    # def spectral_integral(
    #     self,
    #     s: NDArray | complex,
    #     f: Callable,
    #     lower_bound: complex,
    #     upper_bound: complex,
    #     gl_order: int = 100,
    #     split_n: int = 5,
    #     integration_split_points: list[float] = [2, 3, 5, 10, 50, 100],
    #     hook_parameter: float = 1,
    # ) -> NDArray | complex:
    #     # TODO fix integration (broadcasting done wrong)
    #     # TODO method needed for integrating along one axis
    #     # TODO method needed for integrating in the path [a, i1, i2, ..., in, b]
    #     integrand = lambda x: f(s, x) * self.spectral_function(x)
    #     middle_bound = (
    #         lower_bound * (1 - hook_parameter) + np.real(upper_bound) * hook_parameter
    #     )
    #     integral_1 = 0 * s + 0j
    #     integral_2 = 0 * s + 0j
    #     if hook_parameter != 0:
    #         integral_1 += integrate_gl(
    #             integrand,
    #             lower_bound,
    #             middle_bound,
    #             order=gl_order,
    #             split_n=split_n,
    #         )

    #     if np.imag(upper_bound) != 0:
    #         integral_2 += integrate_gl(
    #             integrand,
    #             middle_bound,
    #             upper_bound,
    #             order=gl_order,
    #             split_n=split_n,
    #         )

    #     return integral_1 + integral_2


class SemiStableTwoBodyChannel(Channel):
    def __init__(
        self,
        unstable_particle: UnstableParticle,
        spectator_mass: float,
        vertex_function_squared: VertexFunction,
        taming_factor_squared: TamingFactor,
        build_dispersively: bool = True,
    ):
        self.unstable_particle = unstable_particle
        self.spectator_mass = spectator_mass
        super().__init__(
            threshold=(
                spectator_mass + np.sqrt(unstable_particle.decay_channel.threshold)
            )
            ** 2
        )
        self.phase_space_s_x = lambda s, x: phase_space_twobody(s, x, spectator_mass**2)
        self.vertex_function_squared_s_x = lambda s, x: vertex_function_squared(
            s, x, spectator_mass**2
        )
        self.loop_imag_s_x = lambda s, x: (
            self.phase_space_s_x(s, x)
            * self.vertex_function_squared_s_x(s, x)
            * self.unstable_particle.spectral_function(x)
        )
        self.taming_factor_squared = partial(taming_factor_squared)

        # TODO decide what to do with the two functions below
        self.phase_space = lambda s: self.loop_imag(s)
        self.vertex_function_squared = np.vectorize(lambda s: 1)

        if build_dispersively:
            self.add_dispersive_loop(
                DispersionIntegralRHC(
                    integrand=lambda s: self.loop_imag(s),
                    threshold=self.threshold,
                )
            )

    def loop_imag(
        self,
        s: NDArray | complex,
        gl_order: int = 100,
        split_n: int = 20,
        hook_parameter: float = 1,
    ) -> NDArray | complex:
        s = np.asarray(s)
        result = np.zeros(s.shape, dtype=complex)

        lower_bound = self.unstable_particle.decay_channel.threshold
        upper_bound = (np.sqrt(s) - self.spectator_mass) ** 2

        if hook_parameter == 0:
            intervals = np.empty([*upper_bound.shape, 2], dtype=complex)
            intervals[...,0] = lower_bound
            intervals[...,1] = upper_bound
            intervals_split = split_intervals_logarithmically(intervals, split_n)
            result = integrate_intervals_partially_gl(
                self.loop_imag_s_x,
                s,
                intervals_split,
                order=gl_order,
                s_x_axis_overlap=True,
            )
            return result

        real_upper_bound_mask = np.imag(upper_bound) == 0
        complex_upper_bound_mask = ~real_upper_bound_mask

        if np.any(real_upper_bound_mask):
            real_s = s[real_upper_bound_mask]
            real_upper_bound = upper_bound[real_upper_bound_mask]
            real_intervals = np.empty([*real_upper_bound.shape, 2])
            real_intervals[...,0] = lower_bound
            real_intervals[...,1] = real_upper_bound
            real_intervals_split = split_intervals_logarithmically(real_intervals, split_n)
            result[real_upper_bound_mask] += integrate_intervals_partially_gl(
                self.loop_imag_s_x,
                real_s,
                real_intervals_split,
                order=gl_order,
                s_x_axis_overlap=True,
            )

        if np.any(complex_upper_bound_mask):
            complex_s = s[complex_upper_bound_mask]
            complex_upper_bound = upper_bound[complex_upper_bound_mask]
            complex_middle_bound = (
                    lower_bound * (1 - hook_parameter) + np.real(complex_upper_bound) * hook_parameter
                )
            complex_intervals_1 = np.empty([*complex_upper_bound.shape, 2], dtype=complex)
            complex_intervals_2 = np.empty([*complex_upper_bound.shape, 2], dtype=complex)
            complex_intervals_1[...,0] = lower_bound
            complex_intervals_1[...,1] = complex_middle_bound
            complex_intervals_2[...,0] = complex_middle_bound
            complex_intervals_2[...,1] = complex_upper_bound
            complex_intervals_split_1 = split_intervals_logarithmically(complex_intervals_1, split_n)
            complex_intervals_split_2 = split_intervals_evenly(complex_intervals_2, split_n)
            result[complex_upper_bound_mask] += integrate_intervals_partially_gl(
                self.loop_imag_s_x,
                complex_s,
                complex_intervals_split_1,
                order=gl_order,
                s_x_axis_overlap=True,
            )
            result[complex_upper_bound_mask] += integrate_intervals_partially_gl(
                self.loop_imag_s_x,
                complex_s,
                complex_intervals_split_2,
                order=gl_order,
                s_x_axis_overlap=True,
            )

        return result
