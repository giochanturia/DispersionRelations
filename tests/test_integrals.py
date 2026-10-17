import numpy as np
import pytest
from dispersionrelations.integrals import *

@pytest.fixture
def rng():
    return np.random.default_rng(42)

RANDOM_SAMPLE_SIZE = 10

# ┌──────────────────────────────────────────────────────────┐
# │                      Partitions                          │
# └──────────────────────────────────────────────────────────┘

# ┌──────────────────────────────────────────────────────────┐
# │               Partitions: Sizes and shapes               │
# └──────────────────────────────────────────────────────────┘


def assert_partition_size(partition: Partition, split_n: int):
    assert partition.shape == (split_n + 1,)


def test_even_partition_sizes():
    partition_sizes = np.random.randint(low=1, high=10, size=(RANDOM_SAMPLE_SIZE,))
    for split_n in partition_sizes:
        assert_partition_size(even_partition(split_n), split_n)


def test_logarithmic_partition_sizes():
    partition_sizes = np.random.randint(low=1, high=10, size=(RANDOM_SAMPLE_SIZE,))
    bases = np.random.randint(2, 10, size=(3,))
    for base in bases:
        for split_n in partition_sizes:
            assert_partition_size(
                logarithmic_partition(split_n=split_n, base=base), split_n
            )


def test_distance_based_partition_sizes():
    current_distance_parallel = 0.1
    current_distance_transverse = 0.1
    partition_sizes = np.random.randint(low=1, high=10, size=(RANDOM_SAMPLE_SIZE,))
    for split_n in partition_sizes:
        assert_partition_size(
            distance_based_partition(
                split_n=split_n,
                distance_parallel=current_distance_parallel,
                distance_transverse=current_distance_transverse,
            ),
            split_n,
        )


def test_pole_based_partition_sizes():
    current_startpoint = 1.0
    current_endpoint = 3.0 + 1.0j
    current_pole_position = 2.0 + 0.1j
    partition_sizes = np.random.randint(low=1, high=10, size=(RANDOM_SAMPLE_SIZE,))
    for split_n in partition_sizes:
        assert_partition_size(
            pole_based_partition(
                split_n=split_n,
                startpoint=current_startpoint,
                endpoint=current_endpoint,
                pole_position=current_pole_position,
            ),
            split_n,
        )


# ┌──────────────────────────────────────────────────────────┐
# │               Partitions: Sizes and shapes               │
# └──────────────────────────────────────────────────────────┘


def assert_partition_endpoints(partition: Partition):
    assert np.all(partition[..., 0] == 0.0) * np.all(partition[..., -1] == 1.0)


def test_even_partition_endpoints():
    partition_sizes = np.random.randint(low=1, high=10, size=(RANDOM_SAMPLE_SIZE,))
    for split_n in partition_sizes:
        assert_partition_endpoints(even_partition(split_n))


def test_logarithmic_partition_endpoints():
    partition_sizes = np.random.randint(low=1, high=10, size=(RANDOM_SAMPLE_SIZE,))
    bases = np.random.randint(2, 10, size=(3,))
    for base in bases:
        for split_n in partition_sizes:
            assert_partition_endpoints(
                logarithmic_partition(split_n=split_n, base=base)
            )


def test_distance_based_partition_endpoints():
    current_distance_parallel = 0.1
    current_distance_transverse = 0.1
    partition_sizes = np.random.randint(low=1, high=10, size=(RANDOM_SAMPLE_SIZE,))
    for split_n in partition_sizes:
        assert_partition_endpoints(
            distance_based_partition(
                split_n=split_n,
                distance_parallel=current_distance_parallel,
                distance_transverse=current_distance_transverse,
            ),
        )


def test_pole_based_partition_endpoints():
    current_startpoint = 1.0
    current_endpoint = 3.0 + 1.0j
    current_pole_position = 2.0 + 0.1j
    partition_sizes = np.random.randint(low=1, high=10, size=(RANDOM_SAMPLE_SIZE,))
    for split_n in partition_sizes:
        assert_partition_endpoints(
            pole_based_partition(
                split_n=split_n,
                startpoint=current_startpoint,
                endpoint=current_endpoint,
                pole_position=current_pole_position,
            ),
        )


# ┌──────────────────────────────────────────────────────────┐
# │                  Partitions: step sizes                  │
# └──────────────────────────────────────────────────────────┘


def assert_equal_steps(diffs: NDArray):
    expected_diff = diffs[0]
    np.testing.assert_allclose(diffs, expected_diff, rtol=1e-10)


def test_even_partition_steps():
    partition_sizes = np.random.randint(low=1, high=100, size=(RANDOM_SAMPLE_SIZE,))
    for n in partition_sizes:
        partition = even_partition(n)
        diffs = np.diff(partition)
        assert_equal_steps(diffs)


def test_logarithmic_partition_steps():
    partition_sizes = np.random.randint(low=2, high=10, size=(RANDOM_SAMPLE_SIZE,))
    for n in partition_sizes:
        partition = logarithmic_partition(n)
        diffs = np.diff(np.log(partition[1:]))
        assert_equal_steps(diffs)


# ┌──────────────────────────────────────────────────────────┐
# │                  Partitions: precomputed                 │
# └──────────────────────────────────────────────────────────┘


class TestPrecomputedPartitions:

    def test_even_partition_2(self):
        result = even_partition(split_n=2)
        desired = np.array(
            [
                0 / 2,
                1 / 2,
                2 / 2,
            ]
        )
        np.testing.assert_allclose(result, desired)

    def test_even_partition_3(self):
        result = even_partition(split_n=3)
        desired = np.array(
            [
                0 / 3,
                1 / 3,
                2 / 3,
                3 / 3,
            ]
        )
        np.testing.assert_allclose(result, desired)

    def test_even_partition_10(self):
        result = even_partition(split_n=10)
        desired = np.array(
            [
                0 / 10,
                1 / 10,
                2 / 10,
                3 / 10,
                4 / 10,
                5 / 10,
                6 / 10,
                7 / 10,
                8 / 10,
                9 / 10,
                10 / 10,
            ]
        )
        np.testing.assert_allclose(result, desired)

    def test_logarithmic_partition_2(self):
        result = logarithmic_partition(split_n=2)
        desired = np.array(
            [
                0 / 1,
                1 / 2,
                1 / 1,
            ]
        )
        np.testing.assert_allclose(result, desired)

    def test_logarithmic_partition_3(self):
        result = logarithmic_partition(split_n=3)
        desired = np.array(
            [
                0 / 1,
                1 / 4,
                1 / 2,
                1 / 1,
            ]
        )
        np.testing.assert_allclose(result, desired)

    def test_logarithmic_partition_5(self):
        result = logarithmic_partition(split_n=5)
        desired = np.array(
            [
                0 / 1,
                1 / 16,
                1 / 8,
                1 / 4,
                1 / 2,
                1 / 1,
            ]
        )
        np.testing.assert_allclose(result, desired)

    def test_logarithmic_partition_10(self):
        result = logarithmic_partition(split_n=10)
        desired = np.array(
            [
                0 / 1,
                1 / 512,
                1 / 256,
                1 / 128,
                1 / 64,
                1 / 32,
                1 / 16,
                1 / 8,
                1 / 4,
                1 / 2,
                1 / 1,
            ]
        )
        np.testing.assert_allclose(result, desired)

    def test_logarithmic_partition_3_base_3(self):
        result = logarithmic_partition(split_n=3, base=3)
        desired = np.array(
            [
                0 / 1,
                1 / 9,
                1 / 3,
                1 / 1,
            ]
        )
        np.testing.assert_allclose(result, desired)

    def test_logarithmic_partition_5_base_3(self):
        result = logarithmic_partition(split_n=5, base=3)
        desired = np.array(
            [
                0 / 1,
                1 / 81,
                1 / 27,
                1 / 9,
                1 / 3,
                1 / 1,
            ]
        )
        np.testing.assert_allclose(result, desired)

    def test_logarithmic_partition_3_base_10(self):
        result = logarithmic_partition(split_n=3, base=10)
        desired = np.array(
            [
                0 / 1,
                1 / 100,
                1 / 10,
                1 / 1,
            ]
        )
        np.testing.assert_allclose(result, desired)

    def test_logarithmic_partition_5_base_10(self):
        result = logarithmic_partition(split_n=5, base=10)
        desired = np.array(
            [
                0 / 1,
                1 / 10000,
                1 / 1000,
                1 / 100,
                1 / 10,
                1 / 1,
            ]
        )
        np.testing.assert_allclose(result, desired)

    def test_distance_based_partition_3(self):
        result = distance_based_partition(
            3,
            distance_parallel=0.5,
            distance_transverse=0.5,
        )
        desired = np.array(
            [
                0.0000000000000000,
                0.3660254037844386,
                0.6339745962155613,
                1.0000000000000000,
            ]
        )
        np.testing.assert_allclose(result, desired)

    def test_distance_based_partition_5(self):
        result = distance_based_partition(
            5,
            distance_parallel=0.5,
            distance_transverse=0.5,
        )
        desired = np.array(
            [
                0.0000000000000000,
                0.2452372752527856,
                0.4208077798377319,
                0.5791922201622682,
                0.7547627247472144,
                1.0000000000000000,
            ]
        )
        np.testing.assert_allclose(result, desired)

    def test_distance_based_partition_10(self):
        result = distance_based_partition(
            10,
            distance_parallel=0.5,
            distance_transverse=0.5,
        )
        desired = np.array(
            [
                0.0000000000000000,
                0.1367287359973195,
                0.2452372752527856,
                0.3375401518835469,
                0.4208077798377319,
                0.5000000000000000,
                0.5791922201622682,
                0.6624598481164532,
                0.7547627247472144,
                0.8632712640026805,
                1.0000000000000000,
            ]
        )
        np.testing.assert_allclose(result, desired)


# ┌──────────────────────────────────────────────────────────┐
# │                  Partitions: symmetries                  │
# └──────────────────────────────────────────────────────────┘


def assert_centered_around_zero(array: NDArray):
    np.testing.assert_allclose(array, -np.flip(array), atol=1e-10)


class TestPartitionSymmetries:

    def test_distance_based_partition_parallel_symmetry(self):
        r"""If the point is at the center, the interval should be split symmetrically."""
        current_parallel = 0.5
        transverse_points = np.random.uniform(-3, 3, size=(RANDOM_SAMPLE_SIZE,))
        partition_sizes = np.random.randint(low=2, high=10, size=(RANDOM_SAMPLE_SIZE,))
        for split_n in partition_sizes:
            for current_transverse in transverse_points:
                centered_partition = (
                    distance_based_partition(
                        split_n, current_parallel, current_transverse
                    )
                    - 0.5
                )
                assert_centered_around_zero(centered_partition)

    def test_distance_based_partition_transverse_symmetry(self):
        r"""It should not matter if `distance_transverse` is positive or negative."""
        parallel_points = np.random.uniform(-3, 3, size=(RANDOM_SAMPLE_SIZE,))
        transverse_points = np.random.uniform(-3, 3, size=(RANDOM_SAMPLE_SIZE,))
        partition_size = 13
        for current_parallel in parallel_points:
            for current_transverse in transverse_points:
                np.testing.assert_allclose(
                    distance_based_partition(
                        partition_size, current_parallel, current_transverse
                    ),
                    distance_based_partition(
                        partition_size, current_parallel, -current_transverse
                    ),
                )

    def test_distance_vs_pole_based_partitions(self):
        r"""Pole-based partition should boil down to the distance-based one if the parameters are selected accordingly."""
        current_parallel = 0.5
        current_transverse = 0.5
        current_startpoint = 0.0
        current_endpoint = 1.0
        current_pole = current_parallel + 1j * current_transverse
        partition_sizes = np.random.randint(low=2, high=100, size=(RANDOM_SAMPLE_SIZE,))
        for split_n in partition_sizes:
            np.testing.assert_allclose(
                distance_based_partition(
                    split_n,
                    current_parallel,
                    current_transverse,
                ),
                pole_based_partition(
                    split_n,
                    current_startpoint,
                    current_endpoint,
                    current_pole,
                ),
            )


# ┌──────────────────────────────────────────────────────────┐
# │                 Partitions: Edge cases                   │
# └──────────────────────────────────────────────────────────┘


def test_even_partition_0input():
    with pytest.raises(ValueError):
        even_partition(0)


def test_even_partition_negative_input():
    with pytest.raises(ValueError):
        even_partition(-2)


def test_even_partition_nonint_input():
    with pytest.raises(ValueError):
        even_partition(2.5)


def test_logarithmic_partition_0input():
    with pytest.raises(ValueError):
        logarithmic_partition(0)


def test_logarithmic_partition_negative_input():
    with pytest.raises(ValueError):
        logarithmic_partition(-2)


def test_logarithmic_partition_nonint_input():
    with pytest.raises(ValueError):
        logarithmic_partition(2.5)


def test_db_partition_0input():
    with pytest.raises(ValueError):
        distance_based_partition(0, 3.2, -4.0)


def test_db_partition_negative_input():
    with pytest.raises(ValueError):
        distance_based_partition(-2, 3.2, -4.0)


def test_db_partition_nonint_input():
    with pytest.raises(ValueError):
        distance_based_partition(2.5, 3.2, -4.0)


def test_pb_partition_0input():
    with pytest.raises(ValueError):
        pole_based_partition(0, 2.0 + 1j, -6.0 - 2.5j, -3 - 2j)


def test_pb_partition_negative_input():
    with pytest.raises(ValueError):
        pole_based_partition(-2, 2.0 + 1j, -6.0 - 2.5j, -3 - 2j)


def test_pb_partition_nonint_input():
    with pytest.raises(ValueError):
        pole_based_partition(2.5, 2.0 + 1j, -6.0 - 2.5j, -3 - 2j)


# ┌──────────────────────────────────────────────────────────┐
# │                       Intervals                          │
# └──────────────────────────────────────────────────────────┘


def test_remove_dupl_neighbours_simple():
    arrays = [[0, 1, 1, 2, 3, 2, 4, 4]]

    solutions = [[0, 1, 2, 3, 2, 4]]

    for ind, arr in enumerate(arrays):
        np.testing.assert_allclose(remove_duplicate_neighbours(arr), solutions[ind])


def test_remove_dupl_neighbours_moredim():
    arrays = [
        [
            [[0, 4, 5, 2, 2], [1, 1, 7, 3.5, 3]],
            [[1.2, 3.4, 5.6, 7.8, 9.0], [2.2, 2.2, 5.0, 4.5, 4.5]],
        ]
    ]

    solutions = [[[[4, 5, 2], [1, 7, 3]], [[3.4, 5.6, 9.0], [2.2, 5.0, 4.5]]]]

    for ind, arr in enumerate(arrays):
        np.testing.assert_allclose(remove_duplicate_neighbours(arr), solutions[ind])


def test_remove_dupl_neighbours_same_ends():
    array = [[0, 1, 2, 5, 5, 2, 0]]

    with pytest.raises(ValueError):
        remove_duplicate_neighbours(array)


# def test_gl_quadrature():
#     np.testing.assert_almost_equal(integrate_gl(lambda x: x**2, 1, 2), 7 / 3)


def test_split_interval_simple():
    # intervals = [[1,2,3], [1,3,2], [2,1,3], [3,2,1]]
    parts = [even_partition(1), even_partition(2), even_partition(3)]
    intervals = [[1, 2, 3]]
    # parts = [even_partition(1)]

    solutions = [
        [1, 2, 3],
        [1, 1.5, 2, 2.5, 3],
        [1, 1 + 1.0 / 3.0, 1 + 2.0 / 3.0, 2.0, 2 + 1.0 / 3.0, 2 + 2.0 / 3.0, 3],
    ]

    for i, int in enumerate(intervals):
        for p, part in enumerate(parts):
            np.testing.assert_allclose(
                split_interval(int, part), solutions[i * len(intervals) + p]
            )


def test_split_intervals_reversed_order():
    parts = [even_partition(1), even_partition(2)]
    # parts = [even_partition(2)]
    intervals = [[3, 2, 1]]

    solutions = [[3, 2, 1], [3, 2.5, 2, 1.5, 1]]
    # solutions = [[3, 2.5, 2, 1.5, 1]]

    for i, int in enumerate(intervals):
        for p, part in enumerate(parts):
            np.testing.assert_allclose(
                split_interval(int, part), solutions[i * len(intervals) + p]
            )


def test_split_intervals_mixed_orders():
    parts = [even_partition(1), even_partition(2)]
    intervals = [[2, 1, 3]]

    solutions = [[2, 1, 3], [2, 1.5, 1, 2, 3]]

    for i, int in enumerate(intervals):
        for p, part in enumerate(parts):
            np.testing.assert_allclose(
                split_interval(int, part), solutions[i * len(intervals) + p]
            )


# ┌──────────────────────────────────────────────────────────┐
# │                      GL Routines                         │
# └──────────────────────────────────────────────────────────┘


def test_interval_to_gl_identity():
    interval = [-1, 1]
    gl_orders = [1, 2, 10]

    solution_points = [
        [[0]],
        [[-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]],
        [
            [
                -0.97390653,
                -0.86506337,
                -0.67940957,
                -0.43339539,
                -0.14887434,
                0.14887434,
                0.43339539,
                0.67940957,
                0.86506337,
                0.97390653,
            ]
        ],
    ]

    solution_weights = [
        [[2]],
        [[1, 1]],
        [
            [
                0.06667134,
                0.14945135,
                0.21908636,
                0.26926672,
                0.29552422,
                0.29552422,
                0.26926672,
                0.21908636,
                0.14945135,
                0.06667134,
            ]
        ],
    ]

    for o, order in enumerate(gl_orders):
        np.testing.assert_allclose(
            interval_to_gl(interval, order), (solution_points[o], solution_weights[o])
        )


def test_interval_to_gl_():
    intervals = [[28.5, -4.3], [1.0 + 2j, 12.0 + 8j]]
    gl_orders = [1, 2, 3]

    solutions = [
        [
            ([[12.1]], [[-32.8]]),
            ([[21.56854441, 2.63145559]], [[-16.4, -16.4]]),
            (
                [[24.80338538, 12.1, -0.60338538]],
                [[-9.11111111, -14.57777778, -9.11111111]],
            ),
        ],
        [
            ([[6.5 + 5.0j]], [[11.0 + 6.0j]]),
            (
                [[3.32457352 + 3.26794919j, 9.67542648 + 6.73205081j]],
                [[5.5 + 3.0j, 5.5 + 3.0j]],
            ),
            (
                [[2.23971832 + 2.67620999j, 6.5 + 5.0j, 10.76028168 + 7.32379001j]],
                [
                    [
                        3.05555556 + 1.66666667j,
                        4.88888889 + 2.66666667j,
                        3.05555556 + 1.66666667j,
                    ]
                ],
            ),
        ],
    ]

    for i, interv in enumerate(intervals):
        for o, order in enumerate(gl_orders):
            np.testing.assert_allclose(interval_to_gl(interv, order), (solutions[i][o]))


############# Tests for integrate_interval_gl ############################


def test_integrate_interval_gl_1D():
    # lower = 2.
    # mid = 1.5
    # upper = 1.
    intervals = [
        [1.0, 2.0],
        [1.0, 1.5, 2.0],
        [1.5, 1.0, 2.0],
        [2.0, 1.5, 1.0],
        [2.0, 1.0, 1.5],
    ]
    solutions = [
        7.0 / 3.0,
        7.0 / 3.0,
        1.5416666666666665,
        -7.0 / 3.0,
        -1.5416666666666665,
    ]
    for i, interv in enumerate(intervals):
        np.testing.assert_almost_equal(
            integrate_interval_gl(lambda x: x**2.0, interv), solutions[i]
        )


def test_integrate_interval_gl_1D_slims():

    svals = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]

    solutions = [
        [0.0520012360260124, -0.14542300890509954],
        [0.28665011802678586, 0.2282790786880382],
    ]

    np.testing.assert_almost_equal(
        integrate_interval_gl(lambda x: x**2.0, np.sin(svals)), solutions
    )


#################### Tests for  integrate_interval_second_variable_gl ###############


def test_integrate_interval_second_variable_gl_const_integrand():
    intervals = np.asarray([[[0.0, 1], [1.0, 3.0]], [[3.0, 5.5], [5.5, 17.8]]])

    integrands1 = [lambda y: 1.0, lambda y: y**2.0]

    for ints in integrands1:
        np.testing.assert_almost_equal(
            integrate_interval_gl(ints, intervals),
            integrate_interval_second_variable_gl(
                lambda x, y: ints(y),
                intervals[:, :, 1],
                intervals,
                y_bounds_depend_on_x=True,
            ),
        )


def test_integrate_interval_second_variable_gl():
    intervals = [
        np.asarray([-1.2, 24.7]),
        np.asarray([[0.1, 4.0 * np.pi], [2.7, 2.3]]),
        np.asarray([[[0.0, 1], [1.0, 3.0]], [[3.0, 5.5], [5.5, 17.8]]]),
    ]

    xvals = np.asarray([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

    def antideriv(x, a, b):
        x = np.asarray(x)
        xe = np.expand_dims(x, tuple(x.ndim + ax for ax in range(a.ndim)))
        ae = np.expand_dims(a, tuple(ax for ax in range(x.ndim)))
        be = np.expand_dims(b, tuple(ax for ax in range(x.ndim)))
        return 1 / xe * (np.sin(xe * be) - np.sin(xe * ae))

    for ints in intervals:
        np.testing.assert_almost_equal(
            integrate_interval_second_variable_gl(
                lambda x, y: np.cos(x * y), xvals, ints
            ),
            antideriv(xvals, ints[..., 0], ints[..., 1]),
        )


def test_integrate_interval_second_variable_gl_xlims():
    xvals = [
        np.asarray(3.0),
        np.asarray([-2.0, 2.0 + 1j, 4.0 * np.pi]),
        np.asarray([[-4.0, 4.2, -21], [4.0002, 0.01, 4.02]]),
        np.asarray(
            [[[1.0, 2.0, 1.5], [3.0, 4.0, 3.5]], [[5.0, 6.0, 5.5], [7.0, 8.0, 7.5]]]
        ),
    ]

    def make_interv(xval):
        return np.stack([-xval, xval**2], axis=-1)

    intervals = [
        np.asarray(make_interv(xvals[0])),
        np.asarray(make_interv(xvals[1])),
        np.asarray(make_interv(xvals[2])),
        np.asarray(make_interv(xvals[3])),
    ]

    def antideriv(x, a, b):
        x = np.asarray(x)
        return 1 / x * (np.sin(x * b) - np.sin(x * a))

    for i, xval in enumerate(xvals):
        np.testing.assert_almost_equal(
            integrate_interval_second_variable_gl(
                lambda x, y: np.cos(x * y),
                xval,
                split_interval(intervals[i], even_partition(42)),
                y_bounds_depend_on_x=True,
            ),
            antideriv(xval, intervals[i][..., 0], intervals[i][..., 1]),
        )


def test_integrate_gl():
    xvals = [
        np.asarray(3.0),
        np.asarray([-2.0, 2.0 + 1j, 4.0 * np.pi]),
        np.asarray([[-4.0, 4.2, -21], [4.0002, 0.01, 4.02]]),
        np.asarray(
            [[[1.0, 2.0, 1.5], [3.0, 4.0, 3.5]], [[5.0, 6.0, 5.5], [7.0, 8.0, 7.5]]]
        ),
    ]

    lower = [-xval for xval in xvals]
    upper = [xval**2 for xval in xvals]

    def antideriv(a, b):
        # x = np.asarray(x)
        # return 1 / x * (np.sin(x * b) - np.sin(x * a))
        return np.sin(b) - np.sin(a)

    for i, xval in enumerate(xvals):
        np.testing.assert_almost_equal(
            integrate_gl(lambda y: np.cos(y), lower[i], upper[i]),
            antideriv(lower[i], upper[i]),
        )


# ┌──────────────────────────────────────────────────────────┐
# │                        Residue                           │
# └──────────────────────────────────────────────────────────┘


class TestResidue:
    def test_simple_pole_unit_residue(self):
        # 1/(z-z0) has residue 1
        z0 = 1.0 + 2.0j
        r = residue(lambda z: 1.0 / (z - z0), z0)
        np.testing.assert_almost_equal(r, 1.0 + 0j, decimal=10)

    def test_simple_pole_scaled_residue(self):
        # 13/(z-z0) has residue 13
        z0 = 2.0 + 2.0j
        r = residue(lambda z: 13.0 / (z - z0), z0)
        np.testing.assert_almost_equal(r.real, 13.0, decimal=8)
        np.testing.assert_almost_equal(r.imag, 0.0, decimal=8)

    def test_double_pole(self):
        # 1/(z-z0)^2 has residue 0
        z0 = 1.0j
        r = residue(lambda z: 1.0 / (z - z0) ** 2, z0)
        np.testing.assert_almost_equal(abs(r), 0.0, decimal=8)

    def test_no_pole(self):
        # Analytic function, residue should be ~0
        r = residue(lambda z: np.exp(z), 0.0 + 0.0j)
        np.testing.assert_almost_equal(abs(r), 0.0, decimal=8)

    def test_complex_residue(self):
        # (1+2j)/(z - z0) has residue (1+2j)
        z0 = -1.0 - 1.0j
        c = 1.0 + 2.0j
        r = residue(lambda z: c / (z - z0), z0)
        np.testing.assert_almost_equal(r, c, decimal=8)


# ┌──────────────────────────────────────────────────────────┐
# │                  DispersionIntegralRHC                   │
# └──────────────────────────────────────────────────────────┘


def prepare_RHC_tests():

    threshold = 1.0
    cutoff = 2.0
    asymptotic = np.pi
    infinity = 1e6

    def integrand(s):
        s = np.asarray(s)
        results = np.zeros_like(s)

        mask0 = s <= threshold
        mask1 = (s > threshold) * (s < cutoff)
        mask2 = s >= cutoff

        results[mask1] = asymptotic * (
            3.0 * (s[mask1] - threshold) ** 2.0 / (cutoff - threshold) ** 2.0
            - 2.0 * (s[mask1] - threshold) ** 3.0 / (cutoff - threshold) ** 3.0
        )
        results[mask2] = asymptotic

        return results

    DI = DispersionIntegralRHC(
        integrand,
        threshold,
        infinity=infinity,
        integration_split_points=[
            1.0,
            2.0,
            2.5,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            25.0,
            50.0,
            75.0,
            100.0,
        ],
        integration_order=500,
    )

    # prepare solutions:
    SPMm1 = SimplePoleMonomial(-1)
    SPM0 = SimplePoleMonomial(0)
    SPM1 = SimplePoleMonomial(1)
    SPM2 = SimplePoleMonomial(2)

    def solution(s):
        s = s + 0j
        return (
            s
            * asymptotic
            / np.pi
            * (
                3.0
                / (cutoff - threshold) ** 2.0
                * (
                    SPM1(s, 0.0, threshold, cutoff)
                    - 2.0 * threshold * SPM0(s, 0.0, threshold, cutoff)
                    + threshold**2.0 * SPMm1(s, 0.0, threshold, cutoff)
                )
                - 2.0
                / (cutoff - threshold) ** 3.0
                * (
                    SPM2(s, 0.0, threshold, cutoff)
                    - 3.0 * threshold * SPM1(s, 0.0, threshold, cutoff)
                    + 3.0 * threshold**2.0 * SPM0(s, 0.0, threshold, cutoff)
                    - threshold**3.0 * SPMm1(s, 0.0, threshold, cutoff)
                )
                + SPMm1(s, 0, cutoff, np.inf)
            )
        )

    return DI, solution, integrand


def test_DispersionIntegralRHC_real(rng):

    DI, solution, ig = prepare_RHC_tests()

    vals = [
        3.0,
        1.5,
        np.asarray([-1.0, 0.0, 0.9, 1.1, 1.5, 2.3, 60.0]),
        12.0 * rng.random((4, 5)) - 1.0,
        12.0 * rng.random((4, 5, 6)) - 1.0,
    ]

    for i, val in enumerate(vals):
        np.testing.assert_allclose(DI(val), solution(val), rtol=2e-6)


def test_DispersionIntegralRHC_complex(rng):
    DI, solution, ig = prepare_RHC_tests()

    vals_real = [
        3.0,
        1.5,
        np.asarray([-1.0, 0.0, 0.9, 1.1, 1.5, 2.3, 60.0]),
        12.0 * rng.random((4, 5)) - 1.0,
    ]

    vals_imag = [
        1.5,
        -3.0,
        np.asarray([-10.0, 1e-3, 0.9, -1.1, 1.5, -2.3, 60.0]),
        12.0 * rng.random((4, 5)) - 1.0,
    ]

    vals = []

    for i in range(len(vals_real)):
        vals.append(vals_real[i] + 1j * vals_imag[i])

    for i, val in enumerate(vals):
        np.testing.assert_allclose(DI(val), solution(val), rtol=2e-5)

    for i, val in enumerate(vals_real):
        np.testing.assert_allclose(DI(np.conj(val + 1e-3j)), np.conj(DI(val + 1e-3j)))


def test_disp_int_to_complex_2RS_reals(rng):
    DI, solution, ig = prepare_RHC_tests()

    cf = Complex2RS()

    disp_int_to_complex_2RS(DI, cf)

    vals = [
        3.0,
        1.5,
        np.asarray([-1.0, 1e-3, 0.9, 1.1, 1.5, 2.3, 60.0]),
        12.0 * rng.random((4, 5)) - 1.0,
        12.0 * rng.random((4, 5, 6)) - 1.0,
    ]

    for i, val in enumerate(vals):
        np.testing.assert_allclose(cf(val, 1), solution(val), rtol=2e-6)
        # np.testing.assert_allclose(cf(np.conj(val + 1e-2j), 1), np.conj(solution(val + 1e-2j)), rtol=2e-6)

        np.testing.assert_allclose(cf(val, 2), np.conj(solution(val)), rtol=2e-6)
        # np.testing.assert_allclose(
        #     cf(np.conj(val), 2), np.conj(solution(val) - 2. * 1j * np.imag(solution(val))), rtol=2e-6
        # )
        np.testing.assert_allclose(
            cf.discontinuity(val), 2.0 * 1j * np.imag(solution(val)), rtol=1e-10
        )


def test_disp_int_to_complex_2RS_complex(rng):
    DI, solution, ig = prepare_RHC_tests()
    igs = schwarz_reflection(lambda z: 2j * ig(z))

    cf = Complex2RS()

    disp_int_to_complex_2RS(DI, cf)

    vals_real = [
        3.0,
        1.5,
        np.asarray([-1.0, 0.0, 0.9, 1.1, 1.5, 2.3, 60.0]),
        12.0 * rng.random((4, 5)) - 1.0,
    ]

    vals_imag = [
        1.5,
        -3.0,
        np.asarray([-10.0, 1e-3, 0.9, -1.1, 1.5, -2.3, 60.0]),
        12.0 * rng.random((4, 5)) - 1.0,
    ]

    vals = []

    for i in range(len(vals_real)):
        vals.append(vals_real[i] + 1j * vals_imag[i])

    for i, val in enumerate(vals):
        discval = igs(val)
        np.testing.assert_allclose(cf(val, 1), solution(val), rtol=2e-5)
        np.testing.assert_allclose(cf(np.conj(val + 1e-2j), 1), np.conj(solution(val + 1e-2j)), rtol=2e-5)

        np.testing.assert_allclose(cf(val, 2), solution(val) - discval, rtol=2e-5)
        np.testing.assert_allclose(
            cf(np.conj(val), 2), np.conj(solution(val) - discval), rtol = 2e-5
        )

        np.testing.assert_allclose(
        cf.discontinuity(val), discval, rtol=1e-10
        )


def test_DispersionIntegralRHC_fail_subtraction():

    threshold = 1.0
    cutoff = 2.0
    asymptotic = np.pi

    def integrand(s):
        s = np.asarray(s)
        results = np.zeros_like(s)

        mask0 = s <= threshold
        mask1 = (s > threshold) * (s < cutoff)
        mask2 = s >= cutoff

        results[mask1] = asymptotic * (
            3.0 * (s[mask1] - threshold) ** 2.0 / (cutoff - threshold) ** 2.0
            - 2.0 * (s[mask1] - threshold) ** 3.0 / (cutoff - threshold) ** 3.0
        )
        results[mask2] = asymptotic

        return results

    with pytest.raises(ValueError):
        DI = DispersionIntegralRHC(integrand, threshold, subtraction_point=1.5)
