import pytest
import numpy as np
from dispersionrelations.complex import *
from dispersionrelations.domains import *


# # # # # # # # # # # # # # # # # # # # #
# SCHWARZ # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #

def test_schwarz_reflection_scalar():

    def non_schwarz_func(z):
        return z - 1j

    schwarzified_func_upper = schwarz_reflection(non_schwarz_func)
    schwarzified_func_lower = schwarz_reflection(non_schwarz_func, upper = False)

    scalar_upper = 2. + 3j
    np.testing.assert_almost_equal(
        schwarzified_func_upper(np.conj(scalar_upper)),
        np.conj(schwarzified_func_upper(scalar_upper)),
    )
    np.testing.assert_almost_equal(schwarzified_func_upper(2.), 2. - 1j)
    np.testing.assert_almost_equal(schwarzified_func_upper(2.0 + 1e-8j), 2.0 - 1j)
    np.testing.assert_almost_equal(schwarzified_func_upper(2.0 - 1e-8j), 2.0 + 1j)

    scalar_lower = 2. - 3j

    np.testing.assert_almost_equal(
        schwarzified_func_lower(np.conj(scalar_lower)),
        np.conj(schwarzified_func_lower(scalar_lower)),
    )
    np.testing.assert_almost_equal(schwarzified_func_lower(2.), 2. - 1j)
    np.testing.assert_almost_equal(schwarzified_func_lower(2.0 - 1e-8j), 2.0 - 1j)
    np.testing.assert_almost_equal(schwarzified_func_lower(2.0 + 1e-8j), 2.0 + 1j)


def test_schwarz_reflection_array():

    def non_schwarz_func(z):
        return z - 1j

    schwarzified_func_upper = schwarz_reflection(non_schwarz_func)
    schwarzified_func_lower = schwarz_reflection(non_schwarz_func, upper=False)

    size = (2,3,4)
    arr_real = 1.5 * np.random.random_sample(size = size)
    arr_imag = 1.5 * np.random.random_sample(size = size)

    arr_up = arr_real + 1j * arr_imag
    arr_low = arr_real - 1j * arr_imag

    np.testing.assert_almost_equal(
        schwarzified_func_upper(np.conj(arr_up)),
        np.conj(schwarzified_func_upper(arr_up))
    )

    np.testing.assert_almost_equal(schwarzified_func_upper(arr_real), arr_real - 1j)
    np.testing.assert_almost_equal(schwarzified_func_upper(arr_real + 1e-8j), arr_real - 1j)
    np.testing.assert_almost_equal(
        schwarzified_func_upper(arr_real - 1e-8j), np.conj(arr_real - 1j)
    )

    np.testing.assert_almost_equal(
        schwarzified_func_lower(np.conj(arr_low)),
        np.conj(schwarzified_func_lower(arr_low))
    )

    np.testing.assert_almost_equal(schwarzified_func_lower(arr_real), arr_real - 1j)
    np.testing.assert_almost_equal(schwarzified_func_lower(arr_real - 1e-8j), arr_real - 1j)
    np.testing.assert_almost_equal(
        schwarzified_func_lower(arr_real + 1e-8j), np.conj(arr_real - 1j)
    )


# # # # # # # # # # # # # # # # # # # # #
# HELPERS # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #


def assert_reflection(f, z, cut_angle=0.0, cut_in_imag=True):
    """Schwarz reflection principle test for functions with branch cuts."""
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


# # # # # # # # # # # # # # # # # # # # #
# TESTS # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #

z_positive_real_values = np.linspace(0, 1e4, 24)
z_negative_real_values = np.linspace(-1e4, 0, 24)
z_real_values = np.linspace(-1e4, 1e4, 42)
z_real_values_plus_imag = z_real_values + 1j * 1e-5
z_real_values_minus_imag = np.conj(z_real_values_plus_imag)


# # # # # # # # # # # # # # # # # # # # #
# sqrt_custom_branch_cut  # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #


def test_reflection_sqrt_custom_branch_cut():
    sample_points = z_real_values_plus_imag
    # np.random.seed(42)
    test_angle = (np.random.rand() - 0.5) * 2 * np.pi
    assert_reflection(
        lambda z: sqrt_custom_branch_cut(z, test_angle),
        sample_points,
        test_angle,
        cut_in_imag=False,
    )


# # # # # # # # # # # # # # # # # # # # #
# sqrt_rhc # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #


def test_real_positive_values_sqrt_rhc():
    sample_points = z_positive_real_values
    np.testing.assert_allclose(
        sqrt_rhc(sample_points),
        np.sqrt(sample_points),
    )


def test_real_positive_values_sqrt_rhc_sheet_2():
    sample_points = z_positive_real_values
    np.testing.assert_allclose(
        sqrt_rhc(sample_points, sheet=2),
        -np.sqrt(sample_points),
    )


def test_real_negative_values_sqrt_rhc():
    sample_points = z_negative_real_values
    np.testing.assert_allclose(
        sqrt_rhc(sample_points),
        1j * np.sqrt(-sample_points + 0j),
    )


def test_real_negative_values_sqrt_rhc_sheet_2():
    sample_points = z_negative_real_values
    np.testing.assert_allclose(
        sqrt_rhc(sample_points, sheet=2),
        -1j * np.sqrt(-sample_points + 0j),
    )


def test_invalid_sheets_sqrt_rhc():
    with pytest.raises(ValueError):
        invalid_sheets = [-1, 0, 3, None, "first", "second"]
        for sheet in invalid_sheets:
            _ = sqrt_rhc(1.0, sheet=sheet)


def test_reflection_sqrt_rhc():
    sample_points = z_real_values_plus_imag
    assert_reflection(sqrt_rhc, sample_points, cut_in_imag=False)


# # # # # # # # # # # # # # # # # # # # #
# sqrt_lhc # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #


def test_real_positive_values_sqrt_lhc():
    sample_points = z_positive_real_values
    np.testing.assert_allclose(
        np.imag(sqrt_lhc(sample_points)),
        np.sqrt(sample_points),
    )


def test_real_positive_values_sqrt_lhc_sheet_2():
    sample_points = z_positive_real_values
    np.testing.assert_allclose(
        np.imag(sqrt_lhc(sample_points, sheet=2)),
        -np.sqrt(sample_points),
    )


def test_real_negative_values_sqrt_lhc():
    sample_points = z_negative_real_values
    np.testing.assert_allclose(
        sqrt_lhc(sample_points),
        -1j * np.sqrt(sample_points + 0j),
    )


def test_real_negative_values_sqrt_lhc_sheet_2():
    sample_points = z_negative_real_values
    np.testing.assert_allclose(
        sqrt_lhc(sample_points, sheet=2),
        1j * np.sqrt(sample_points + 0j),
    )


def test_invalid_sheets_sqrt_lhc():
    with pytest.raises(ValueError):
        invalid_sheets = [-1, 0, 3, None, "first", "second"]
        for sheet in invalid_sheets:
            _ = sqrt_lhc(1.0, sheet=sheet)


def test_reflection_sqrt_lhc():
    sample_points = z_real_values_plus_imag
    assert_reflection(sqrt_lhc, sample_points, cut_in_imag=False)


# # # # # # # # # # # # # # # # # # # # #
# log_custom_branch_cut # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #

def test_reflection_log_custom_branch_cut():
    sample_points = z_real_values_plus_imag
    # np.random.seed(42)
    test_angle = (np.random.rand() - 0.5) * 2 * np.pi
    assert_reflection(
        lambda z: log_custom_branch_cut(z, test_angle) - 1j * np.pi,
        sample_points,
        test_angle,
        cut_in_imag=True,
    )

# # # # # # # # # # # # # # # # # # # # #
# log_rhc  # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #

def test_real_positive_values_log_rhc():
    sample_points = z_positive_real_values[z_positive_real_values != 0]
    np.testing.assert_allclose(
        log_rhc(sample_points),
        np.log(sample_points),
    )

def test_real_negative_values_log_rhc():
    sample_points = z_negative_real_values[z_negative_real_values != 0]
    np.testing.assert_allclose(
        log_rhc(sample_points),
        np.log(-sample_points + 0j) + 1j * np.pi,
    )

def test_reflection_log_rhc():
    sample_points = z_real_values_plus_imag
    assert_reflection(lambda z: log_rhc(z) - 1j * np.pi, sample_points, cut_in_imag=True)

# # # # # # # # # # # # # # # # # # # # #
# log_lhc  # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #

def test_real_positive_values_log_lhc():
    sample_points = z_positive_real_values[z_positive_real_values != 0]
    np.testing.assert_allclose(
        log_lhc(sample_points),
        np.log(sample_points) + 1j * np.pi,
    )

def test_real_negative_values_log_lhc():
    sample_points = z_negative_real_values[z_negative_real_values != 0]
    np.testing.assert_allclose(
        log_lhc(sample_points),
        np.log(-sample_points + 0j),
    )

def test_reflection_log_lhc():
    sample_points = z_real_values_plus_imag
    assert_reflection(lambda z: log_lhc(z) - 1j * np.pi, sample_points, cut_in_imag=True)

# # # # # # # # # # # # # # # # # # # # #
# log_c  # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #

def test_real_positive_values_log_c():
    sample_points = z_positive_real_values[z_positive_real_values != 0]
    np.testing.assert_allclose(
        log_c(sample_points),
        np.log(sample_points),
    )

def test_real_negative_values_log_c():
    sample_points = z_negative_real_values[z_negative_real_values != 0]
    np.testing.assert_allclose(
        log_c(sample_points),
        np.log(-sample_points + 0j) - 1j * np.pi,
    )

def test_reflection_log_c():
    sample_points = z_real_values_plus_imag
    assert_reflection(lambda z: log_c(z) + 1j * np.pi, sample_points, cut_in_imag=True)


# # # # # # # # # # # # # # # # # # # # #
# extract_phase # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #


def test_extract_phase_positive_start():
    E_1 = np.linspace(0, 1.1, 1000)
    s_1 = E_1**2
    t_1 = 2 * np.pi * 3 * np.sin(2 * np.pi * s_1)
    t_2 = 0.0 * s_1
    t_3 = -t_1
    np.testing.assert_allclose(extract_phase(np.exp(1j * t_1)), t_1)
    np.testing.assert_allclose(extract_phase(np.exp(1j * t_2)), t_2)
    np.testing.assert_allclose(extract_phase(np.exp(1j * t_3)), t_3)


# # # # # # # # # # # # # # # # # # # # #
# Complex2RS  # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #

def test_Complex2RS():

    CRS = Complex2RS()

    np.testing.assert_equal(len(CRS.sheet_1.functions), 0)
    np.testing.assert_equal(len(CRS.sheet_2.functions), 0)
    np.testing.assert_equal(len(CRS.discontinuity.functions), 0)

    def non_schwarz_func(z):
        return z - 1j

    schwarzified_func_upper = schwarz_reflection(non_schwarz_func)
    schwarzified_func_lower = schwarz_reflection(non_schwarz_func, upper=False)

    s1 = Domainify(schwarzified_func_upper)
    s2 = Domainify(schwarzified_func_lower)
    disc = Domainify(lambda x: schwarzified_func_upper(x) - schwarzified_func_lower(x))

    CRS.sheet_1.add_function(s1)
    CRS.sheet_2.add_function(s2)
    CRS.discontinuity.add_function(disc)

    val = 1. + 2j

    np.testing.assert_almost_equal(CRS.sheet_1(val), schwarzified_func_upper(val))
    np.testing.assert_almost_equal(CRS.sheet_2(val), schwarzified_func_lower(val))
    np.testing.assert_almost_equal(
        CRS.discontinuity(val),
        schwarzified_func_upper(val) - schwarzified_func_lower(val),
    )

    np.testing.assert_almost_equal(CRS(val), CRS.sheet_1(val))
    np.testing.assert_almost_equal(CRS(val, 1), CRS.sheet_1(val))
    np.testing.assert_almost_equal(CRS(val, 2), CRS.sheet_2(val))


    with pytest.raises(ValueError):
        CRS(val, 3)
