# ┌──────────────────────────────────────────────────────────┐
# │                         Typing                           │
# └──────────────────────────────────────────────────────────┘
from __future__ import annotations  # for type hints in python==3.9

import json
import os
from typing import Literal
from typing_extensions import override
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
from scipy.interpolate import RegularGridInterpolator

# ┌──────────────────────────────────────────────────────────┐
# │                    Internal imports                      │
# └──────────────────────────────────────────────────────────┘
from dispersionrelations.domains import (
    within_bounds,
    DomainedFunction,
    points_C1_to_R2,
    domain_R2_to_C1,
)
from dispersionrelations.complex import schwarz_reflection
from inspect import signature

# ┌──────────────────────────────────────────────────────────┐
# │                     INTERPOLATION                        │
# └──────────────────────────────────────────────────────────┘


class RecursiveGrid:
    def __init__(
        self,
        scale: float = 1.0,
        dimension: int = 2,
    ):
        self.scale = scale
        self.dim = dimension

    def generate_level(
        self,
        level: int = 0,
        n_gridpoints: int | tuple = 100,
        meshgrid_kwargs: dict = {"indexing": "ij"},
    ) -> tuple:
        # logger.debug(f"Generating level {level} with scale {self.scale}.")
        r = self.scale * 3**level
        # logger.debug(f"Calculated range r: {r}")
        if isinstance(n_gridpoints, int):
            x = np.linspace(-r, r, n_gridpoints)
            # logger.debug(f"Generated 1D grid points for x with shape {x.shape}")
            return np.meshgrid(*self.dim * [x], **meshgrid_kwargs)
        else:
            # logger.debug(f"Using provided n_gridpoints: {n_gridpoints}")
            if len(n_gridpoints) != self.dim:
                raise ValueError(
                    f"n_gridpoints must be an integer or a tuple of length {self.dim}."
                )
            x_nd = [np.linspace(-r, r, n) for n in n_gridpoints]
            # logger.debug(
                # f"Generated 1D grid points for each dimension with shapes {[x.shape for x in x_nd]}"
            # )
            return np.meshgrid(*x_nd, **meshgrid_kwargs)

    def determine_level(
        self,
        points: NDArray,
    ) -> NDArray:
        # logger.debug(f"Determining level for points with shape {points.shape}")
        return np.asarray(
            np.ceil(
                np.clip(
                    np.log(1 / self.scale * np.max(np.abs(points), axis=-1))
                    / np.log(3),
                    0,
                    None,
                )
            )
        ).astype(int)

    def level_index_dictionary(
        self,
        points: NDArray,
    ) -> NDArray:
        points = np.asarray(points)
        # logger.debug(f"Classifying points with shape {points.shape}")
        levels = self.determine_level(points)
        classified_points = {}
        for level in np.unique(levels):
            classified_points[level] = np.where(levels == level)
        # logger.debug(f"Classified points into levels: {list(classified_points.keys())}")
        return classified_points


class ComplexPrecomputatinator:
    def __init__(
        self,
        function: DomainedFunction,
        grid: RecursiveGrid,
        domain: Literal["full", "upper", "lower"] = "full",
    ):
        self.f = function
        self.grid = grid
        self.domain = domain

    def evaluate_level(
        self,
        level: int,
        n_gridpoints: int | tuple = 100,
        meshgrid_kwargs: dict = {"indexing": "ij"},
    ) -> dict:
        z_RE, z_IM = self.grid.generate_level(level, n_gridpoints, meshgrid_kwargs)

        if self.domain == "upper":
            upper_half_mask = z_IM[0, :] >= 0
            if np.any(~upper_half_mask):
                # logger.debug("Masking to leave only the upper complex plane.")
                # logger.debug(f"Initial shape={z_IM.shape}")
                z_RE = np.delete(z_RE, ~upper_half_mask, axis=1)
                z_IM = np.delete(z_IM, ~upper_half_mask, axis=1)
                # logger.debug(f"Resulting shape={z_IM.shape}")

        if self.domain == "lower":
            lower_half_mask = z_IM[0, :] <= 0
            if np.any(~lower_half_mask):
                # logger.debug("Masking to leave only the lower complex plane.")
                # logger.debug(f"Initial shape={z_IM.shape}")
                z_RE = np.delete(z_RE, ~lower_half_mask, axis=1)
                z_IM = np.delete(z_IM, ~lower_half_mask, axis=1)
                # logger.debug(f"Resulting shape={z_IM.shape}")

        z_C1 = z_RE + 1j * z_IM
        f_C1 = self.f(z_C1)
        f_R2 = points_C1_to_R2(f_C1)
        f_RE = f_R2[..., 0]
        f_IM = f_R2[..., 1]
        return {
            "level": level,
            "shape": z_RE.shape,
            "meshgrid": meshgrid_kwargs,
            "data": {
                "z_RE": z_RE,
                "z_IM": z_IM,
                "f_RE": f_RE,
                "f_IM": f_IM,
            },
        }

    def export_level(
        self,
        level: int,
        filepath: str,
        n_gridpoints: int | tuple = 100,
        meshgrid_kwargs: dict = {"indexing": "ij"},
        metadata: dict = {},
        **json_kwargs,
    ) -> None:
        result = {
            **metadata,
            **self.evaluate_level(level, n_gridpoints, meshgrid_kwargs),
        }
        logger.info(f"Saving level {level} to {filepath}")
        json.dump(result, open(filepath, "w"), **json_kwargs)

    def export_channel_level(
        self,
        channel: str,
        quantity: str,
        sheet: Literal[1, 2],
        level: int,
        folder: str,
        n_gridpoints: int | tuple = 100,
        meshgrid_kwargs: dict = {"indexing": "ij"},
        metadata: dict = {},
        **json_kwargs,
    ) -> None:
        filepath = os.path.join(
            folder,
            f"{channel}_{quantity}_sheet_{sheet}_grid_{self.grid.scale}_level_{level}.json",
        )
        self.export_level(
            level,
            filepath,
            n_gridpoints,
            meshgrid_kwargs=meshgrid_kwargs,
            metadata={"channel": channel, "quantity": quantity, "sheet": sheet, **metadata},
            **json_kwargs
        )


class ComplexInterpolator(DomainedFunction):
    def __init__(
        self,
        grid: RecursiveGrid,
        domain: Literal["full", "upper", "lower"] = "full",
    ):
        self.grid = grid
        self.domain = domain
        self.interpolators = {}
        self.highest_level = -1
        self.real_line_bounds = (0, 0)

    @override
    def in_domain(self, z: NDArray | complex) -> NDArray | bool:
        z = np.asarray(z)
        result_cp = self.__in_complex_grid_domain__(z)


        real_mask = np.isreal(z)
        if np.any(real_mask):
            result_re = np.zeros(z.shape, dtype=bool)
            result_re[real_mask] = self.__in_real_line_domain__(z[real_mask])
            return result_re + result_cp

        return result_cp

    def __in_real_line_domain__(self, z: NDArray | complex) -> NDArray | bool:
        if -1 in self.interpolators:
            return within_bounds(np.real(z), self.real_line_bounds)
        else:
            return np.zeros(z.shape, dtype=bool)

    def __in_complex_grid_domain__(self, z: NDArray | complex) -> NDArray | bool:
        return self.grid.determine_level(points_C1_to_R2(z)) <= self.highest_level

    def add_real_line_interpolator(
        self,
        z_re: NDArray,
        f_re: NDArray,
        f_im: NDArray,
        **interpolator_kwargs,
    ) -> None:
        if z_re.shape != f_re.shape or z_re.shape != f_im.shape:
            logger.error(
                f"Shape mismatch: z_re shape {z_re.shape}, f_re shape {f_re.shape}, f_im shape {f_im.shape}"
            )
            raise ValueError(
                "Shape of z_re, f_re, and f_im must all match for real line interpolator."
            )
        interp_re = RegularGridInterpolator(
            (z_re,), f_re, bounds_error=False, **interpolator_kwargs
        )
        interp_im = RegularGridInterpolator(
            (z_re,), f_im, bounds_error=False, **interpolator_kwargs
        )
        self.interpolators[-1] = lambda z: interp_re(z) + 1j * interp_im(z)
        self.real_line_bounds = (np.min(z_re), np.max(z_re))

    def add_level_interpolator(
        self,
        level: int,
        z_RE: NDArray,
        z_IM: NDArray,
        f_RE: NDArray,
        f_IM: NDArray,
        impose_schwarz_reflection: bool = True,
        **interpolator_kwargs,
    ) -> None:
        z_RE = np.asarray(z_RE)
        z_IM = np.asarray(z_IM)
        f_RE = np.asarray(f_RE)
        f_IM = np.asarray(f_IM)
        if z_RE.shape != f_RE.shape or z_IM.shape != f_IM.shape:
            logger.error(
                f"Shape mismatch: grid shape {z_RE.shape}, f_RE shape {f_RE.shape}"
            )
            raise ValueError(
                "Shape of f_RE and f_IM must match the shape of f_RE and f_IM."
            )

        if self.domain == "upper":
            upper_half_mask = z_IM[0, :] >= 0
            if np.any(~upper_half_mask):
                logger.debug("Masking to leave only the upper complex plane.")
                logger.debug(f"Initial shape={z_IM.shape}")
                z_RE = np.delete(z_RE, ~upper_half_mask, axis=1)
                z_IM = np.delete(z_IM, ~upper_half_mask, axis=1)
                f_RE = np.delete(f_RE, ~upper_half_mask, axis=1)
                f_IM = np.delete(f_IM, ~upper_half_mask, axis=1)
                logger.debug(f"Resulting shape={z_IM.shape}")

        if self.domain == "lower":
            lower_half_mask = z_IM[0, :] <= 0
            if np.any(~lower_half_mask):
                logger.debug("Masking to leave only the lower complex plane.")
                logger.debug(f"Initial shape={z_IM.shape}")
                z_RE = np.delete(z_RE, ~lower_half_mask, axis=1)
                z_IM = np.delete(z_IM, ~lower_half_mask, axis=1)
                f_RE = np.delete(f_RE, ~lower_half_mask, axis=1)
                f_IM = np.delete(f_IM, ~lower_half_mask, axis=1)
                logger.debug(f"Resulting shape={z_IM.shape}")

        z_re_ax = z_RE[:, 0]
        z_im_ax = z_IM[0, :]
        interp_RE = RegularGridInterpolator(
            (z_re_ax, z_im_ax), f_RE, bounds_error=False, **interpolator_kwargs
        )
        interp_IM = RegularGridInterpolator(
            (z_re_ax, z_im_ax), f_IM, bounds_error=False, **interpolator_kwargs
        )
        self.interpolators[level] = domain_R2_to_C1(
            lambda z_2D: (interp_RE(z_2D) + 1j * interp_IM(z_2D))
        )
        if impose_schwarz_reflection:
            if self.domain == "lower":
                self.interpolators[level] = schwarz_reflection(self.interpolators[level], upper = False)
            else:
                self.interpolators[level] = schwarz_reflection(self.interpolators[level], upper = True)

        if level > self.highest_level:
            self.highest_level = level

    def add_level_interpolator_from_file(
        self,
        filepath: str,
        impose_schwarz_reflection: bool = True,
        **interpolator_kwargs,
    ) -> None:
        file = json.load(open(filepath, "r"))
        self.add_level_interpolator(
            file["level"],
            file["data"]["z_RE"],
            file["data"]["z_IM"],
            file["data"]["f_RE"],
            file["data"]["f_IM"],
            impose_schwarz_reflection,
            **interpolator_kwargs,
        )

    def __compute_complex__(
        self,
        z: NDArray,
    ) -> NDArray:
        z = np.asarray(z)
        z_2D = points_C1_to_R2(z)
        masks = self.grid.level_index_dictionary(z_2D)
        result = np.empty(z.shape, dtype=complex)
        result[:] = np.nan
        for level, mask in masks.items():
            if level in self.interpolators:
                logger.debug(
                    f"Computing values for level {level} with {len(mask[0])} points."
                )
                z_masked = z[mask]
                result[mask] = self.interpolators[level](z_masked)
            else:
                logger.warning(
                    f"No interpolator found for level {level}. Points at this level will be set to NaN."
                )
        return result

    def __compute_real__(
        self,
        z: NDArray,
    ) -> NDArray:
        z = np.asarray(z)
        if np.imag(z).any():
            logger.warning(
                "Input contains complex values. Only the real part will be used for interpolation."
            )
        z_re = np.real(z)
        if -1 in self.interpolators:
            logger.debug(
                f"Computing values using real line interpolator for {z_re.shape[0]} points."
            )
            return self.interpolators[-1](z_re)
        else:
            return self.__compute_complex__(z)

    @override
    def __call__(
        self,
        z: NDArray,
    ) -> NDArray:
        z = np.asarray(z)
        real_mask = np.isreal(z)
        complex_mask = ~real_mask
        result = np.empty(z.shape, dtype=complex)
        result[:] = np.nan
        if real_mask.any():
            logger.debug(f"Computing real values for {real_mask.sum()} points.")
            result[real_mask] = self.__compute_real__(z[real_mask])
        if complex_mask.any():
            logger.debug(f"Computing complex values for {complex_mask.sum()} points.")
            result[complex_mask] = self.__compute_complex__(z[complex_mask])
        return result
