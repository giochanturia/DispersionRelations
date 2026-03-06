import pytest
import numpy as np
from dispersionrelations.utils import *

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
# sqrtRHC # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #


def test_real_positive_values_sqrtRHC():
    sample_points = z_positive_real_values
    np.testing.assert_allclose(
        sqrtRHC(sample_points),
        np.sqrt(sample_points),
    )


def test_real_positive_values_sqrtRHC_sheet_2():
    sample_points = z_positive_real_values
    np.testing.assert_allclose(
        sqrtRHC(sample_points, sheet=2),
        -np.sqrt(sample_points),
    )


def test_real_negative_values_sqrtRHC():
    sample_points = z_negative_real_values
    np.testing.assert_allclose(
        sqrtRHC(sample_points),
        1j * np.sqrt(-sample_points + 0j),
    )


def test_real_negative_values_sqrtRHC_sheet_2():
    sample_points = z_negative_real_values
    np.testing.assert_allclose(
        sqrtRHC(sample_points, sheet=2),
        -1j * np.sqrt(-sample_points + 0j),
    )


def test_invalid_sheets_sqrtRHC():
    with pytest.raises(ValueError):
        invalid_sheets = [-1, 0, 3, None, "first", "second"]
        for sheet in invalid_sheets:
            _ = sqrtRHC(1.0, sheet=sheet)


def test_reflection_sqrtRHC():
    sample_points = z_real_values_plus_imag
    assert_reflection(sqrtRHC, sample_points, cut_in_imag=False)


# # # # # # # # # # # # # # # # # # # # #
# sqrtLHC # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #


def test_real_positive_values_sqrtLHC():
    sample_points = z_positive_real_values
    np.testing.assert_allclose(
        np.imag(sqrtLHC(sample_points)),
        np.sqrt(sample_points),
    )


def test_real_positive_values_sqrtLHC_sheet_2():
    sample_points = z_positive_real_values
    np.testing.assert_allclose(
        np.imag(sqrtLHC(sample_points, sheet=2)),
        -np.sqrt(sample_points),
    )


def test_real_negative_values_sqrtLHC():
    sample_points = z_negative_real_values
    np.testing.assert_allclose(
        sqrtLHC(sample_points),
        -1j * np.sqrt(sample_points + 0j),
    )


def test_real_negative_values_sqrtLHC_sheet_2():
    sample_points = z_negative_real_values
    np.testing.assert_allclose(
        sqrtLHC(sample_points, sheet=2),
        1j * np.sqrt(sample_points + 0j),
    )


def test_invalid_sheets_sqrtLHC():
    with pytest.raises(ValueError):
        invalid_sheets = [-1, 0, 3, None, "first", "second"]
        for sheet in invalid_sheets:
            _ = sqrtLHC(1.0, sheet=sheet)


def test_reflection_sqrtLHC():
    sample_points = z_real_values_plus_imag
    assert_reflection(sqrtLHC, sample_points, cut_in_imag=False)


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
# logRHC  # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #

def test_real_positive_values_logRHC():
    sample_points = z_positive_real_values[z_positive_real_values != 0]
    np.testing.assert_allclose(
        logRHC(sample_points),
        np.log(sample_points),
    )

def test_real_negative_values_logRHC():
    sample_points = z_negative_real_values[z_negative_real_values != 0]
    np.testing.assert_allclose(
        logRHC(sample_points),
        np.log(-sample_points + 0j) + 1j * np.pi,
    )

def test_reflection_logRHC():
    sample_points = z_real_values_plus_imag
    assert_reflection(lambda z: logRHC(z) - 1j * np.pi, sample_points, cut_in_imag=True)

# # # # # # # # # # # # # # # # # # # # #
# logLHC  # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #

def test_real_positive_values_logLHC():
    sample_points = z_positive_real_values[z_positive_real_values != 0]
    np.testing.assert_allclose(
        logLHC(sample_points),
        np.log(sample_points) + 1j * np.pi,
    )

def test_real_negative_values_logLHC():
    sample_points = z_negative_real_values[z_negative_real_values != 0]
    np.testing.assert_allclose(
        logLHC(sample_points),
        np.log(-sample_points + 0j),
    )

def test_reflection_logLHC():
    sample_points = z_real_values_plus_imag
    assert_reflection(lambda z: logLHC(z) - 1j * np.pi, sample_points, cut_in_imag=True)

# # # # # # # # # # # # # # # # # # # # #
# logC  # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #

def test_real_positive_values_logC():
    sample_points = z_positive_real_values[z_positive_real_values != 0]
    np.testing.assert_allclose(
        logC(sample_points),
        np.log(sample_points),
    )

def test_real_negative_values_logC():
    sample_points = z_negative_real_values[z_negative_real_values != 0]
    np.testing.assert_allclose(
        logC(sample_points),
        np.log(-sample_points + 0j) - 1j * np.pi,
    )

def test_reflection_logC():
    sample_points = z_real_values_plus_imag
    assert_reflection(lambda z: logC(z) + 1j * np.pi, sample_points, cut_in_imag=True)


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


def test_extract_phase_negative_start():
    E_1 = np.linspace(0, 1.1, 1000)
    s_1 = E_1**2
    t_1 = 2 * np.pi * 3 * np.sin(2 * np.pi * s_1)
    t_2 = 0.0 * s_1
    t_3 = -t_1
    np.testing.assert_allclose(extract_phase(np.exp(1j * t_1)), t_1)
    np.testing.assert_allclose(extract_phase(np.exp(1j * t_2)), t_2)
    np.testing.assert_allclose(extract_phase(np.exp(1j * t_3)), t_3)


# # # # # # # # # # # # # # # # # # # # #
# cite / uncite # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #

def test_cite_format():
    text = "reference"
    cited_text = cite(text)
    assert cited_text[:6] == r"\cite{" and cited_text[-1] == "}"


def test_cite_uncite():
    text = "reference"
    cited_text = cite(text)
    uncited_text = uncite(cited_text)
    assert text == uncited_text


def test_uncite_invalid_format():
    with pytest.raises(ValueError):
        invalid_citations = [
            r"text without \cite",
            r"\cite{missing_end",
            r"missing_start}",
            r"\\cite{accidental_double_backslash}",
        ]
        for invalid_citation in invalid_citations:
            _ = uncite(invalid_citation)


def test_empty_uncite():
    with pytest.warns(UserWarning):
        empty_citation = r"\cite{}"
        result = uncite(empty_citation)
        assert result == ""


# # # # # # # # # # # # # # # # # # # # #
# scientific_notation # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #
    

def test_scientific_notation():
    assert scientific_notation(np.pi) == '3.14'
    assert scientific_notation(np.pi, 10) == '3.1415926536'
    assert scientific_notation(13.4) == r'1.34 \times 10^{1}'
    assert scientific_notation(13.4, 10) == r'1.34 \times 10^{1}'
    assert scientific_notation(0.134) == r'1.34 \times 10^{-1}'
    assert scientific_notation(0.000134) == r'1.34 \times 10^{-4}'
    assert scientific_notation(13495) == r'1.35 \times 10^{4}'
    assert scientific_notation(13495, 20) == r'1.3495 \times 10^{4}'


# # # # # # # # # # # # # # # # # # # # #
# PDG_rounding  # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #


def test_PDG_rounding_case_1():
    a1, b1 = rounding_PDG(0.827, 0.367)
    assert a1 == 0.8 and b1 == 0.4

def test_PDG_rounding_case_1_integer():
    a1, b1 = rounding_PDG(827, 367)
    assert a1 == 800 and b1 == 400

def test_PDG_rounding_case_2():
    a2, b2 = rounding_PDG(0.827, 0.119)
    assert a2 == 0.83 and b2 == 0.12

def test_PDG_rounding_case_2_integer():
    a2, b2 = rounding_PDG(827, 119)
    assert a2 == 830 and b2 == 120

def test_PDG_rounding_case_3():
    a3, b3 = rounding_PDG(0.827, 0.971)
    assert a3 == 0.8 and b3 == 1.0

def test_PDG_rounding_case_3_integer():
    a3, b3 = rounding_PDG(827, 971)
    assert a3 == 800 and b3 == 1000