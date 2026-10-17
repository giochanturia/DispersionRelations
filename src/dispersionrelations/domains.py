# ┌──────────────────────────────────────────────────────────┐
# │                         Typing                           │
# └──────────────────────────────────────────────────────────┘
from __future__ import annotations  # for type hints in python==3.9
from typing import Callable
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
from abc import ABC, abstractmethod
from functools import wraps
import numpy as np
from inspect import signature

# ┌──────────────────────────────────────────────────────────┐
# │                        DOMAINS                           │
# └──────────────────────────────────────────────────────────┘


def within_bounds(
    x: NDArray | float,
    bounds: tuple[float, float],
) -> NDArray | bool:
    r"""
    Returns a mask for `x` that falls within `bounds`.

    Parameters
    ----------
    x : array_like
        A sequence of real numbers.
    bounds : tuple
        A tuple of two real numbers (lower_bound, upper_bound).

    Returns
    -------
    mask : array_like
        A boolean array of the same shape as `x`, where True indicates that the corresponding element of `x` is within the specified bounds.
    """

    bounds = np.sort(bounds)

    x = np.asarray(x)

    scalar_input = x.ndim == 0

    if scalar_input:
        x_s = x.item()
        return (x_s > bounds[0]) and (x_s < bounds[1])

    x_shape = x.shape
    if len(x_shape) > 1:
        x = x.flatten()
    diff = np.sign(x - np.asarray(bounds)[:, np.newaxis])
    diff[1] *= -1
    result = np.prod(diff > 0, axis=0, dtype=bool)
    return result.reshape(x_shape)


def within_complex_bounds(
    z: NDArray | complex,
    bounds_re: tuple[float, float],
    bounds_im: tuple[float, float],
) -> NDArray | bool:
    r"""
    Returns a mask for a complex array `z` that falls within complex bounds given by `bounds_re` and `bounds_im`.

    Parameters
    ----------
    z : array_like
        A sequence of complex numbers.
    bounds_re : tuple
        A tuple of two real numbers (lower_bound_re, upper_bound_re) for the real part.
    bounds_im : tuple
        A tuple of two real numbers (lower_bound_im, upper_bound_im) for the imaginary part.

    Returns
    -------
    mask : array_like
        A boolean array of the same shape as `z`, where True indicates that the corresponding element of `z` has its real part within `bounds_re` and its imaginary part within `bounds_im`.

    See Also
    --------
    within_bounds : For checking if real numbers are within bounds.
    """
    return within_bounds(np.real(z), bounds_re) * within_bounds(np.imag(z), bounds_im)


def points_RxR_to_R2(
    points_x: NDArray,
    points_y: NDArray,
) -> NDArray:
    points_x = np.asarray(points_x)
    points_y = np.asarray(points_y)
    if points_x.shape != points_y.shape:
        raise ValueError("Input points must have the same shape.")
    return np.stack((points_x, points_y), axis=-1)


def points_C1_to_R2(
    points: NDArray,
) -> NDArray:
    return points_RxR_to_R2(np.real(points), np.imag(points))


def points_R2_to_C1(
    points: NDArray,
) -> NDArray:
    points = np.asarray(points)
    if points.shape[-1] != 2:
        raise ValueError(
            "Input points must have shape (..., 2) for R^2 to C^1 conversion."
        )
    return points[..., 0] + 1j * points[..., 1]


def domain_R2_to_C1(f: Callable):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(z_C1: NDArray[np.complexfloating]):
            return func(points_C1_to_R2(z_C1))

        return wrapper

    return decorator(f)


def domain_C1_to_R2(f):
    def decorator(func):
        @wraps(func)
        def wrapper(z_R2: NDArray[np.floating]):
            return func(points_R2_to_C1(z_R2))

        return wrapper

    return decorator(f)


def exclusive_masks(
    masks: NDArray,
) -> NDArray:
    r"""
    Takes a list of masks and sets the value to True only for the first mask that is True for each point, ensuring that each point is assigned to exactly one mask.

    Parameters
    ----------
    masks : array_like
        A list of boolean arrays (masks) of the same shape, where each mask indicates a certain condition.

    Returns
    -------
    exclusive_masks : array_like
        A boolean array of the same shape as the input masks, where each point is True for exactly one mask (the first one that is True in the input list) and False for all others.
    """
    return masks * (np.cumsum(np.asarray(masks, dtype=bool), axis=0) == 1)


def always_in_domain(
    z: NDArray | complex,
) -> NDArray | bool:
    z = np.asarray(z)
    return np.ones(z.shape, dtype=bool)


class DomainedFunction(ABC, Callable):
    @abstractmethod
    def __call__(
        self,
        z: NDArray | complex,
    ) -> NDArray | complex:
        pass

    def in_domain(
        self,
        z: NDArray | complex,
    ) -> NDArray | bool:
        return always_in_domain(z)


class DomainedFunctionCollection(DomainedFunction):
    def __init__(
        self,
        functions: list[DomainedFunction],
    ):
        self.functions = functions

    def add_function(
        self,
        func: DomainedFunction,
        index: int | None = None,
    ) -> None:
        if index is None:
            index = len(self.functions)
        self.functions.insert(index, func)

    def __check_for_empty__(self):
        if len(self.functions) == 0:
            raise RuntimeError(
                "There are no functions in the DomainedFunctionCollection."
            )

    @override
    def __call__(
        self,
        z: NDArray | complex,
    ) -> NDArray | complex:
        self.__check_for_empty__()
        z = np.asarray(z)
        scalar_input = z.ndim == 0
        if scalar_input:
            z = z.reshape(1)
        masks = [func.in_domain(z) for func in self.functions]
        exclusive_masks_list = exclusive_masks(masks)
        result = np.empty(z.shape, dtype=complex)
        result[:] = np.nan
        for func, mask in zip(self.functions, exclusive_masks_list):
            if np.any(mask):
                result[mask] = func(z[mask])
        if scalar_input:
            return result[0]
        return result

    @override
    def in_domain(
        self,
        z: NDArray | complex,
    ) -> NDArray | bool:
        self.__check_for_empty__()
        z = np.asarray(z)
        scalar_input = z.ndim == 0
        if scalar_input:
            z = z.reshape(1)
        masks = np.asarray([func.in_domain(z) for func in self.functions])
        result = np.any(masks, axis=0)
        if scalar_input:
            return bool(result[0])
        return result


class Domainify(DomainedFunction):
    def __init__(
        self,
        function: Callable,
        in_domain: Callable[[NDArray | complex], NDArray | bool] | None = None,
    ):
        self.function = function
        if in_domain is not None:
            self.in_domain = in_domain

        if 1 != len(signature(self.function).parameters):
            raise ValueError(
                rf"The number of arguments of function must be equal to one, but it is {len(signature(self.function).parameters)}."
            )

    @override
    def __call__(
        self,
        z: NDArray,
    ) -> NDArray:
        # TODO what if the points are not `in_domain`?
        return self.function(z)
