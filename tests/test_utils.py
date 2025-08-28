import numpy as np
from dispersionrelations.utils import *


def test_sqrt_along_rhc():
    sample_points = np.linspace(0, 1e4, 10)
    np.testing.assert_allclose(np.sqrt(sample_points), sqrtRHC(sample_points))
    np.testing.assert_allclose(np.sqrt(sample_points), np.imag(sqrtLHC(sample_points)))


def assert_reflection(f, z, cut_angle=0.0, cut_in_imag=True):
    z_refl = np.conj(z)
    rotation = np.exp(1j * cut_angle)
    z_rot = rotation * z
    z_refl_rot = rotation * z_refl
    f_val = f(z_rot)
    f_val_refl = f(z_refl_rot)
    if cut_in_imag:
        np.testing.assert_allclose(f_val, np.conj(f_val_refl))
    else:
        np.testing.assert_allclose(f_val, -np.conj(f_val_refl))


def test_sqrt_reflection():
    sample_points = np.linspace(-1e4, 1e4, 3) + 1j * np.linspace(-1e2, 1e2, 3) ** 2
    assert_reflection(sqrtRHC, sample_points, cut_in_imag=False)
    assert_reflection(sqrtLHC, sample_points, cut_in_imag=False)
    # np.random.seed(42)
    test_angle = (np.random.rand() - 0.5) * 2 * np.pi
    assert_reflection(
        lambda z: sqrt_custom_branch_cut(z, test_angle),
        sample_points,
        test_angle,
        cut_in_imag=False,
    )


def test_log_reflection():
    sample_points = np.linspace(-1e4, 1e4, 4) + 1j * np.linspace(-1e2, 1e2, 4) ** 2
    assert_reflection(lambda z: logRHC(z) - 1j * np.pi, sample_points, cut_in_imag=True)
    assert_reflection(lambda z: logLHC(z) - 1j * np.pi, sample_points, cut_in_imag=True)
    assert_reflection(lambda z: logC(z) + 1j * np.pi, sample_points, cut_in_imag=True)
    # np.random.seed(42)
    test_angle = (np.random.rand() - 0.5) * 2 * np.pi
    assert_reflection(
        lambda z: log_custom_branch_cut(z, test_angle) - 1j * np.pi,
        sample_points,
        test_angle,
        cut_in_imag=True,
    )


def test_phase_extraction():
    E_1 = np.linspace(0, 1.1, 1000)
    s_1 = E_1**2
    t_1 = 2 * np.pi * 3 * np.sin(2 * np.pi * s_1)
    t_2 = 0.0 * s_1
    t_3 = -t_1
    np.testing.assert_allclose(extract_phase(np.exp(1j * t_1)), t_1)
    np.testing.assert_allclose(extract_phase(np.exp(1j * t_2)), t_2)
    np.testing.assert_allclose(extract_phase(np.exp(1j * t_3)), t_3)
