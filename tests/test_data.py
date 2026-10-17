import numpy as np
import pytest
import warnings
import tempfile
import os

from dispersionrelations.data import *


# ---------------------------------------------------------------------------
# check_pd
# ---------------------------------------------------------------------------

class TestCheckPD:
    def test_identity_is_pd(self):
        assert check_pd(np.eye(3)) is True

    def test_diagonal_positive_is_pd(self):
        assert check_pd(np.diag([1.0, 2.0, 3.0])) is True

    def test_zero_matrix_not_pd(self):
        assert check_pd(np.zeros((3, 3))) is False

    def test_negative_definite_not_pd(self):
        assert check_pd(-np.eye(3)) is False

    def test_singular_matrix_not_pd(self):
        mat = np.array([[1.0, 1.0], [1.0, 1.0]])
        assert check_pd(mat) is False

    def test_1x1_positive(self):
        assert check_pd(np.array([[5.0]])) is True


# ---------------------------------------------------------------------------
# corr_to_cov
# ---------------------------------------------------------------------------

class TestCorrToCov:
    def test_identity_corr(self):
        std = np.array([2.0, 3.0])
        cov = corr_to_cov(np.eye(2), std)
        expected = np.diag([4.0, 9.0])
        np.testing.assert_allclose(cov, expected)

    def test_full_correlation(self):
        std = np.array([2.0, 3.0])
        corr = np.ones((2, 2))
        cov = corr_to_cov(corr, std)
        expected = np.array([[4.0, 6.0], [6.0, 9.0]])
        np.testing.assert_allclose(cov, expected)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            corr_to_cov(np.ones((2, 3)), np.array([1.0, 2.0, 3.0]))

    def test_incompatible_dims_raises(self):
        with pytest.raises(ValueError):
            corr_to_cov(np.eye(2), np.array([1.0, 2.0, 3.0]))

    def test_non_1d_std_raises(self):
        with pytest.raises(ValueError):
            corr_to_cov(np.eye(2), np.ones((2, 1)))


# ---------------------------------------------------------------------------
# cov_to_corr
# ---------------------------------------------------------------------------

class TestCovToCorr:
    def test_diagonal_cov(self):
        cov = np.diag([4.0, 9.0])
        corr = cov_to_corr(cov)
        np.testing.assert_allclose(corr, np.eye(2))

    def test_non_trivial_cov(self):
        cov = np.array([[4.0, 6.0], [6.0, 9.0]])
        corr = cov_to_corr(cov)
        np.testing.assert_allclose(corr, np.ones((2, 2)))

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            cov_to_corr(np.ones((2, 3)))

    def test_roundtrip(self):
        std = np.array([1.0, 2.0, 3.0])
        corr_orig = np.array([[1.0, 0.5, 0.2],
                               [0.5, 1.0, 0.3],
                               [0.2, 0.3, 1.0]])
        cov = corr_to_cov(corr_orig, std)
        corr_back = cov_to_corr(cov)
        np.testing.assert_allclose(corr_back, corr_orig)


# ---------------------------------------------------------------------------
# cov_to_std
# ---------------------------------------------------------------------------

class TestCovToStd:
    def test_basic(self):
        cov = np.diag([1.0, 4.0, 9.0])
        std = cov_to_std(cov)
        np.testing.assert_allclose(std, [1.0, 2.0, 3.0])

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            cov_to_std(np.ones((2, 3)))


# ---------------------------------------------------------------------------
# std_to_cov
# ---------------------------------------------------------------------------

class TestStdToCov:
    def test_no_correlation(self):
        std = np.array([1.0, 2.0, 3.0])
        cov = std_to_cov(std, correlation=0.0)
        np.testing.assert_allclose(cov, np.diag([1.0, 4.0, 9.0]))

    def test_full_correlation(self):
        std = np.array([2.0, 3.0])
        cov = std_to_cov(std, correlation=1.0)
        expected = np.array([[4.0, 6.0], [6.0, 9.0]])
        np.testing.assert_allclose(cov, expected)

    def test_partial_correlation(self):
        std = np.array([1.0, 1.0])
        cov = std_to_cov(std, correlation=0.5)
        np.testing.assert_allclose(cov[0, 1], 0.5)
        np.testing.assert_allclose(cov[0, 0], 1.0)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            std_to_cov(np.ones((2, 1)))


# ---------------------------------------------------------------------------
# chi2 family — shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def chi2_data():
    """3-point diagonal covariance setup."""
    fit = np.array([1.0, 2.0, 3.0])
    obs = np.array([1.1, 2.2, 2.8])
    std = np.array([0.1, 0.2, 0.2])
    cov = std_to_cov(std)
    cov_inv = np.linalg.inv(cov)
    return fit, obs, cov, cov_inv


class TestChi2:
    def test_perfect_fit_is_zero(self, chi2_data):
        fit, _, cov, _ = chi2_data
        assert chi2(fit, fit, cov) == pytest.approx(0.0)

    def test_basic_value(self, chi2_data):
        fit, obs, cov, _ = chi2_data
        # diagonal cov: sum((f-o)^2 / sigma^2)
        diff = fit - obs
        std = np.sqrt(np.diag(cov))
        expected = np.sum((diff / std) ** 2)
        assert chi2(fit, obs, cov) == pytest.approx(expected)

    def test_with_weights(self, chi2_data):
        fit, obs, cov, _ = chi2_data
        weights = 2.0 * np.ones_like(cov)
        val_weighted = chi2(fit, obs, cov, weights=weights)
        val_plain = chi2(fit, obs, cov / weights)
        assert val_weighted == pytest.approx(val_plain)


class TestChi2WithInverse:
    def test_matches_chi2(self, chi2_data):
        fit, obs, cov, cov_inv = chi2_data
        assert chi2_with_inverse(fit, obs, cov_inv) == pytest.approx(chi2(fit, obs, cov))

    def test_perfect_fit_is_zero(self, chi2_data):
        fit, _, _, cov_inv = chi2_data
        assert chi2_with_inverse(fit, fit, cov_inv) == pytest.approx(0.0)

    def test_with_weights(self, chi2_data):
        fit, obs, cov, cov_inv = chi2_data
        weights = 2.0 * np.ones_like(cov)
        val_wi = chi2_with_inverse(fit, obs, cov_inv, weights=weights)
        val_plain = chi2_with_inverse(fit, obs, cov_inv * weights)
        assert val_wi == pytest.approx(val_plain)


class TestChi2Vector:
    def test_sums_to_total_chi2(self, chi2_data):
        fit, obs, cov, _ = chi2_data
        vec = chi2_vector(fit, obs, cov)
        assert np.sum(vec) == pytest.approx(chi2(fit, obs, cov))

    def test_perfect_fit_is_zero(self, chi2_data):
        fit, _, cov, _ = chi2_data
        np.testing.assert_allclose(chi2_vector(fit, fit, cov), 0.0)

    def test_with_weights(self, chi2_data):
        fit, obs, cov, _ = chi2_data
        weights = 2.0 * np.ones_like(cov)
        vec = chi2_vector(fit, obs, cov, weights=weights)
        assert np.sum(vec) == pytest.approx(chi2(fit, obs, cov, weights=weights))


class TestChi2VectorWithInverse:
    def test_matches_chi2_vector(self, chi2_data):
        fit, obs, cov, cov_inv = chi2_data
        np.testing.assert_allclose(
            chi2_vector_with_inverse(fit, obs, cov_inv),
            chi2_vector(fit, obs, cov),
        )

    def test_with_weights(self, chi2_data):
        fit, obs, cov, cov_inv = chi2_data
        weights = 3.0 * np.ones_like(cov)
        vec = chi2_vector_with_inverse(fit, obs, cov_inv, weights=weights)
        assert np.sum(vec) == pytest.approx(chi2_with_inverse(fit, obs, cov_inv, weights=weights))


class TestChi2Reduced:
    def test_no_parameters(self, chi2_data):
        fit, obs, cov, _ = chi2_data
        n = len(obs)
        expected = chi2(fit, obs, cov) / n
        assert chi2_reduced(fit, obs, cov) == pytest.approx(expected)

    def test_with_parameters(self, chi2_data):
        fit, obs, cov, _ = chi2_data
        n = len(obs)
        k = 1
        expected = chi2(fit, obs, cov) / (n - k)
        assert chi2_reduced(fit, obs, cov, number_of_parameters=k) == pytest.approx(expected)


class TestChi2ReducedWithInverse:
    def test_matches_chi2_reduced(self, chi2_data):
        fit, obs, cov, cov_inv = chi2_data
        assert chi2_reduced_with_inverse(fit, obs, cov_inv) == pytest.approx(
            chi2_reduced(fit, obs, cov)
        )

    def test_with_parameters(self, chi2_data):
        fit, obs, cov, cov_inv = chi2_data
        assert chi2_reduced_with_inverse(fit, obs, cov_inv, number_of_parameters=1) == pytest.approx(
            chi2_reduced(fit, obs, cov, number_of_parameters=1)
        )


# ---------------------------------------------------------------------------
# Datablock
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_block():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 4.0, 9.0, 16.0])
    return Datablock(x, y, "x", "y^2")


@pytest.fixture
def block_with_cov():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    std_stat = np.array([0.1, 0.2, 0.3])
    std_syst = np.array([0.05, 0.05, 0.05])
    cov_stat = std_to_cov(std_stat)
    cov_syst = std_to_cov(std_syst, correlation=1.0)
    return Datablock(x, y, "x", "y", cov_stat=cov_stat, cov_syst=cov_syst)


class TestDatablocConstructor:
    def test_basic_attributes(self, simple_block):
        assert simple_block.N == 4
        np.testing.assert_array_equal(simple_block.x, [1, 2, 3, 4])
        np.testing.assert_array_equal(simple_block.y, [1, 4, 9, 16])
        assert simple_block.xlabel == "x"
        assert simple_block.ylabel == "y^2"

    def test_zero_covs_by_default(self, simple_block):
        np.testing.assert_array_equal(simple_block.cov_stat, np.zeros((4, 4)))
        np.testing.assert_array_equal(simple_block.cov_syst, np.zeros((4, 4)))

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            Datablock(np.array([1.0, 2.0]), np.array([1.0]), "x", "y")

    def test_cov_stat_set_via_constructor(self):
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        cov = np.diag([0.01, 0.04])
        db = Datablock(x, y, "x", "y", cov_stat=cov)
        np.testing.assert_allclose(db.cov_stat, cov)

    def test_metadata_stored(self):
        x = np.array([1.0])
        y = np.array([1.0])
        db = Datablock(x, y, "x", "y", metadata={"ref": "PDG2024"})
        assert db.metadata["ref"] == "PDG2024"


class TestDatablocCheckCovPD:
    def test_warns_when_not_pd(self, simple_block):
        # zero covariance is not PD
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            simple_block.check_cov_pd()
            assert len(w) == 1
            assert "positive definite" in str(w[0].message).lower()

    def test_no_warning_when_pd(self, block_with_cov):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            block_with_cov.check_cov_pd()
            assert len(w) == 0


class TestDatablocStdAccessors:
    def test_get_std_stat(self, block_with_cov):
        np.testing.assert_allclose(block_with_cov.get_std_stat(), [0.1, 0.2, 0.3])

    def test_get_std_syst(self, block_with_cov):
        np.testing.assert_allclose(block_with_cov.get_std_syst(), [0.05, 0.05, 0.05])

    def test_get_std_total_quadrature(self, block_with_cov):
        stat = block_with_cov.get_std_stat()
        syst = block_with_cov.get_std_syst()
        expected = np.sqrt(stat**2 + syst**2)
        np.testing.assert_allclose(block_with_cov.get_std_total(add_in_quadrature=True), expected)

    def test_get_std_total_linear(self, block_with_cov):
        stat = block_with_cov.get_std_stat()
        syst = block_with_cov.get_std_syst()
        np.testing.assert_allclose(block_with_cov.get_std_total(add_in_quadrature=False), stat + syst)


class TestDatablocAddCov:
    def test_add_cov_stat(self, simple_block):
        cov = np.diag([1.0, 2.0, 3.0, 4.0])
        simple_block.add_cov_stat(cov)
        np.testing.assert_allclose(simple_block.cov_stat, cov)

    def test_add_cov_stat_accumulates(self, simple_block):
        cov = np.diag([1.0, 1.0, 1.0, 1.0])
        simple_block.add_cov_stat(cov)
        simple_block.add_cov_stat(cov)
        np.testing.assert_allclose(simple_block.cov_stat, 2 * cov)

    def test_add_cov_stat_wrong_shape_raises(self, simple_block):
        with pytest.raises(ValueError):
            simple_block.add_cov_stat(np.eye(3))

    def test_add_cov_syst(self, simple_block):
        cov = np.diag([1.0, 2.0, 3.0, 4.0])
        simple_block.add_cov_syst(cov)
        np.testing.assert_allclose(simple_block.cov_syst, cov)

    def test_add_cov_syst_wrong_shape_raises(self, simple_block):
        with pytest.raises(ValueError):
            simple_block.add_cov_syst(np.eye(3))


class TestDatablocAddStd:
    def test_add_std_stat_uncorrelated(self, simple_block):
        std = np.array([0.1, 0.2, 0.3, 0.4])
        simple_block.add_std_stat(std)
        np.testing.assert_allclose(np.diag(simple_block.cov_stat), std**2)
        # off-diagonal should be zero
        np.testing.assert_allclose(simple_block.cov_stat[0, 1], 0.0)

    def test_add_std_stat_correlated(self, simple_block):
        std = np.array([1.0, 1.0, 1.0, 1.0])
        simple_block.add_std_stat(std, correlation=1.0)
        np.testing.assert_allclose(simple_block.cov_stat, np.ones((4, 4)))

    def test_add_std_syst_default_full_corr(self, simple_block):
        std = np.array([0.1, 0.1, 0.1, 0.1])
        simple_block.add_std_syst(std)
        # full correlation by default
        expected = np.outer(std, std)
        np.testing.assert_allclose(simple_block.cov_syst, expected)

    def test_add_rel_stat(self, simple_block):
        # 10% relative uncertainty on each point
        simple_block.add_rel_stat(0.1)
        expected_std = 0.1 * np.abs(simple_block.y)
        np.testing.assert_allclose(np.sqrt(np.diag(simple_block.cov_stat)), expected_std)

    def test_add_rel_syst(self, simple_block):
        simple_block.add_rel_syst(0.05)
        expected_std = 0.05 * np.abs(simple_block.y)
        np.testing.assert_allclose(np.sqrt(np.diag(simple_block.cov_syst)), expected_std)


class TestDatablocExternalCalibration:
    def test_scales_y(self, simple_block):
        y_orig = simple_block.y.copy()
        simple_block.add_external_calibration(2.0, 0.0)
        np.testing.assert_allclose(simple_block.y, 2.0 * y_orig)

    def test_scales_covs(self, block_with_cov):
        cov_stat_orig = block_with_cov.cov_stat.copy()
        cov_syst_orig = block_with_cov.cov_syst.copy()
        block_with_cov.add_external_calibration(3.0, 0.0)
        np.testing.assert_allclose(block_with_cov.cov_stat, 9.0 * cov_stat_orig)
        np.testing.assert_allclose(block_with_cov.cov_syst, 9.0 * cov_syst_orig)

    def test_adds_syst_uncertainty_when_c_std_nonzero(self, block_with_cov):
        cov_syst_before = block_with_cov.cov_syst.copy()
        block_with_cov.add_external_calibration(1.0, 0.1)
        # systematic uncertainty should have grown
        assert np.trace(block_with_cov.cov_syst) > np.trace(cov_syst_before * 1.0)

    def test_updates_ylabel(self, simple_block):
        simple_block.add_external_calibration(1.0, 0.0, new_ylabel="new_y")
        assert simple_block.ylabel == "new_y"

    def test_non_scalar_raises(self, simple_block):
        with pytest.raises(ValueError):
            simple_block.add_external_calibration(np.array([1.0, 2.0, 3.0, 4.0]), 0.0)


class TestDatablocRescaleY:
    def test_basic_rescale(self, simple_block):
        y_orig = simple_block.y.copy()
        factor = np.array([1.0, 2.0, 3.0, 4.0])
        simple_block.rescale_y(factor)
        np.testing.assert_allclose(simple_block.y, factor * y_orig)

    def test_rescales_covariances(self, block_with_cov):
        cov_stat_orig = block_with_cov.cov_stat.copy()
        factor = np.array([2.0, 2.0, 2.0])
        block_with_cov.rescale_y(factor)
        expected = np.outer(factor, factor) * cov_stat_orig
        np.testing.assert_allclose(block_with_cov.cov_stat, expected)

    def test_wrong_shape_raises(self, simple_block):
        with pytest.raises(ValueError):
            simple_block.rescale_y(np.array([1.0, 2.0]))

    def test_invalid_stat_or_syst_raises(self, simple_block):
        with pytest.raises(ValueError):
            simple_block.rescale_y(np.ones(4), factor_std=np.ones(4), stat_or_syst="bad")

    def test_factor_std_stat(self, simple_block):
        factor = np.ones(4)
        factor_std = np.array([0.1, 0.1, 0.1, 0.1])
        simple_block.rescale_y(factor, factor_std=factor_std, stat_or_syst='stat')
        assert np.trace(simple_block.cov_stat) > 0

    def test_updates_ylabel(self, simple_block):
        simple_block.rescale_y(np.ones(4), new_ylabel="rescaled")
        assert simple_block.ylabel == "rescaled"


class TestDatablocFilters:
    def test_filter_data(self, simple_block):
        mask = np.array([True, False, True, False])
        simple_block.filter_data(mask)
        assert simple_block.N == 2
        np.testing.assert_array_equal(simple_block.x, [1.0, 3.0])

    def test_filter_data_wrong_mask_raises(self, simple_block):
        with pytest.raises(ValueError):
            simple_block.filter_data(np.array([True, False]))

    def test_filter_xrange_lower(self, simple_block):
        simple_block.filter_xrange(x_min=2.0)
        assert np.all(simple_block.x >= 2.0)

    def test_filter_xrange_upper(self, simple_block):
        simple_block.filter_xrange(x_max=3.0)
        assert np.all(simple_block.x <= 3.0)

    def test_filter_xrange_both(self, simple_block):
        simple_block.filter_xrange(x_min=2.0, x_max=3.0)
        assert simple_block.N == 2

    def test_filter_yrange(self, simple_block):
        # y = [1, 4, 9, 16], keep y >= 4
        simple_block.filter_yrange(y_min=4.0)
        assert np.all(simple_block.y >= 4.0)

    def test_filter_yrange_both(self, simple_block):
        simple_block.filter_yrange(y_min=4.0, y_max=9.0)
        assert simple_block.N == 2

    def test_remove_datapoints(self, simple_block):
        simple_block.remove_datapoints([1, 3])
        assert simple_block.N == 2
        np.testing.assert_array_equal(simple_block.x, [1.0, 3.0])

    def test_remove_datapoints_out_of_bounds_raises(self, simple_block):
        with pytest.raises(ValueError):
            simple_block.remove_datapoints([10])

    def test_filter_preserves_cov_shape(self, block_with_cov):
        block_with_cov.filter_xrange(x_min=1.5)
        assert block_with_cov.cov_stat.shape == (block_with_cov.N, block_with_cov.N)
        assert block_with_cov.cov_syst.shape == (block_with_cov.N, block_with_cov.N)


class TestDatablocDownsample:
    def test_downsample_by_2(self, simple_block):
        simple_block.downsample_by(2)
        assert simple_block.N == 2
        np.testing.assert_array_equal(simple_block.x, [1.0, 3.0])

    def test_downsample_by_1_unchanged(self, simple_block):
        x_orig = simple_block.x.copy()
        simple_block.downsample_by(1)
        np.testing.assert_array_equal(simple_block.x, x_orig)

    def test_downsample_by_invalid_raises(self, simple_block):
        with pytest.raises(ValueError):
            simple_block.downsample_by(0)

    def test_downsample_by_float_raises(self, simple_block):
        with pytest.raises(ValueError):
            simple_block.downsample_by(2.0)

    def test_downsample_to_not_implemented(self, simple_block):
        with pytest.raises(NotImplementedError):
            simple_block.downsample_to(2)

    def test_interpolate_not_implemented(self, simple_block):
        with pytest.raises(NotImplementedError):
            simple_block.interpolate(np.array([1.5, 2.5]))


class TestDatablocStrRepr:
    def test_str_without_ref(self, simple_block):
        s = str(simple_block)
        assert "4" in s
        assert "x" in s
        assert "y^2" in s

    def test_str_with_ref(self):
        db = Datablock(np.array([1.0]), np.array([1.0]), "x", "y",
                       metadata={"ref": "PDG2024"})
        assert "PDG2024" in str(db)

    def test_repr_equals_str(self, simple_block):
        assert repr(simple_block) == str(simple_block)


class TestDatablocAddOperator:
    def test_add_concatenates(self, simple_block):
        other = Datablock(np.array([5.0, 6.0]), np.array([25.0, 36.0]), "x", "y^2")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            combined = simple_block + other
        assert combined.N == 6
        np.testing.assert_array_equal(combined.x, [1, 2, 3, 4, 5, 6])

    def test_add_block_diagonal_cov(self, block_with_cov):
        other = Datablock(np.array([4.0, 5.0]), np.array([8.0, 10.0]), "x", "y")
        other.add_std_stat(np.array([0.1, 0.1]))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            combined = block_with_cov + other
        # cross-block covariance should be zero
        assert combined.cov_stat[0, 3] == pytest.approx(0.0)
        assert combined.cov_stat[3, 0] == pytest.approx(0.0)

    def test_add_non_datablock_raises(self, simple_block):
        with pytest.raises(ValueError):
            simple_block + 5

    def test_radd_with_zero(self, simple_block):
        result = sum([simple_block])
        assert result is simple_block

    def test_sum_of_list(self, simple_block):
        other = Datablock(np.array([5.0]), np.array([25.0]), "x", "y")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            combined = sum([simple_block, other])
        assert combined.N == 5

    def test_add_merges_refs(self):
        db1 = Datablock(np.array([1.0]), np.array([1.0]), "x", "y",
                        metadata={"ref": "A"})
        db2 = Datablock(np.array([2.0]), np.array([2.0]), "x", "y",
                        metadata={"ref": "B"})
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            combined = db1 + db2
        assert "A" in combined.metadata["ref"]
        assert "B" in combined.metadata["ref"]


# ---------------------------------------------------------------------------
# Datawriter / Datareader roundtrip
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_block_for_io():
    x = np.linspace(1.0, 5.0, 5)
    y = x**2
    db = Datablock(x, y, "E", "sigma")
    db.add_std_stat(0.1 * np.ones(5))
    db.add_std_syst(0.05 * y, correlation=0.0)
    return db


class TestDatawriterDatareader:
    def test_roundtrip(self, sample_block_for_io):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = Datawriter(sample_block_for_io, "test", folder_path=tmpdir, quiet=True)
            writer.write_all()

            reader = Datareader(tmpdir, "test")
            db_read = reader.read_all()

        np.testing.assert_allclose(db_read.x, sample_block_for_io.x)
        np.testing.assert_allclose(db_read.y, sample_block_for_io.y)
        np.testing.assert_allclose(db_read.cov_stat, sample_block_for_io.cov_stat)
        np.testing.assert_allclose(db_read.cov_syst, sample_block_for_io.cov_syst)

    def test_roundtrip_preserves_labels(self, sample_block_for_io):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = Datawriter(sample_block_for_io, "test", folder_path=tmpdir, quiet=True)
            writer.write_all()

            reader = Datareader(tmpdir, "test")
            db_read = reader.read_all()

        assert db_read.xlabel == sample_block_for_io.xlabel
        assert db_read.ylabel == sample_block_for_io.ylabel

    def test_write_creates_files(self, sample_block_for_io):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = Datawriter(sample_block_for_io, "mydata", folder_path=tmpdir, quiet=True)
            writer.write_all()

            assert os.path.exists(os.path.join(tmpdir, "mydata.dat"))
            assert os.path.exists(os.path.join(tmpdir, "mydata_COV_STAT.dat"))
            assert os.path.exists(os.path.join(tmpdir, "mydata_COV_SYST.dat"))

    def test_write_data_only(self, sample_block_for_io):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = Datawriter(sample_block_for_io, "mydata", folder_path=tmpdir, quiet=True)
            writer.write_data()
            assert os.path.exists(os.path.join(tmpdir, "mydata.dat"))

    def test_metadata_header_footer(self, sample_block_for_io):
        sample_block_for_io.metadata["header"] = "MyHeader"
        sample_block_for_io.metadata["footer"] = "MyFooter"
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = Datawriter(sample_block_for_io, "test", folder_path=tmpdir, quiet=True)
            assert "MyHeader" in writer.header
            assert "MyFooter" in writer.footer

    def test_single_point_roundtrip(self):
        db = Datablock(np.array([1.0]), np.array([2.0]), "x", "y")
        db.add_std_stat(np.array([0.1]))
        db.add_std_syst(np.array([0.05]))
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = Datawriter(db, "single", folder_path=tmpdir, quiet=True)
            writer.write_all()
            db_read = Datareader(tmpdir, "single").read_all()
        np.testing.assert_allclose(db_read.x, db.x)
        np.testing.assert_allclose(db_read.y, db.y)
        assert db_read.N == 1



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
# rounding_pdg  # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #


def test_pdg_rounding_case_1():
    a1, b1 = rounding_pdg(0.827, 0.367)
    assert a1 == 0.8 and b1 == 0.4


def test_pdg_rounding_case_1_integer():
    a1, b1 = rounding_pdg(827, 367)
    assert a1 == 800 and b1 == 400


def test_pdg_rounding_case_2():
    a2, b2 = rounding_pdg(0.827, 0.119)
    assert a2 == 0.83 and b2 == 0.12


def test_pdg_rounding_case_2_integer():
    a2, b2 = rounding_pdg(827, 119)
    assert a2 == 830 and b2 == 120


def test_pdg_rounding_case_3():
    a3, b3 = rounding_pdg(0.827, 0.971)
    assert a3 == 0.8 and b3 == 1.0


def test_pdg_rounding_case_3_integer():
    a3, b3 = rounding_pdg(827, 971)
    assert a3 == 800 and b3 == 1000