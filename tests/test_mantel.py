from tests.test_common import create_test_coords, create_test_scattered
import pyssage.distances
import pyssage.mantel


def test_mantel():
    """
    correct answer calculated from PASSaGE 2

    Matrix 1: Distance Matrix 1
    Matrix 2: Angle Matrix 1
      Matrices are 355 x 355

    Observed Z = 2731350.67512
    Correlation = 0.29830
      t = 19.83088
      Left-tailed p = 1.00000
      Right-tailed p = 0.00000
      Two-tailed p = 0.00000

    although not asserted (as not part of function output), manual inspection of Z and t are also identical
    """
    coords = create_test_coords()
    distances = pyssage.distances.euclidean_distance_matrix(coords[:, 0], coords[:, 1])
    angles = pyssage.distances.euclidean_angle_matrix(coords[:, 0], coords[:, 1])
    r, p_value, output_text, _, _, _, _, _ = pyssage.mantel.mantel(distances, angles, [])
    for line in output_text:
        print(line)
    assert round(r, 5) == 0.29830
    assert round(p_value, 5) == 0.00000  # two-tailed p by default


def test_mantel_with_permutation():
    """

    cannot formally test permutation results

    """
    coords = create_test_coords()
    distances = pyssage.distances.euclidean_distance_matrix(coords[:, 0], coords[:, 1])
    angles = pyssage.distances.euclidean_angle_matrix(coords[:, 0], coords[:, 1])
    r, p_value, output_text, _, _, _, perm_values, _ = pyssage.mantel.mantel(distances, angles, [], permutations=100)
    for line in output_text:
        print(line)


def test_mantel_with_partial_single():
    """
    correct answer calculated from PASSaGE 2

    Mantel Test
    Matrix 1: Data Distance Matrix 2
    Matrix 2: Distance Matrix 1
    Matrices held constant: Angle Matrix 1
      Matrices are 355 x 355

    Observed Z = 2512061.39465
    Correlation = 0.26699
      t = 15.67369
      Left-tailed p = 1.00000
      Right-tailed p = 0.00000
      Two-tailed p = 0.00000

    """
    data, _ = create_test_scattered()
    coords = create_test_coords()
    distances = pyssage.distances.euclidean_distance_matrix(coords[:, 0], coords[:, 1])
    angles = pyssage.distances.euclidean_angle_matrix(coords[:, 0], coords[:, 1])
    data_distances = pyssage.distances.data_distance_matrix(data, pyssage.distances.data_distance_euclidean)

    r, p_value, output_text, _, _, _, _, _ = pyssage.mantel.mantel(data_distances, distances, [angles])
    for line in output_text:
        print(line)
    assert round(r, 5) == 0.26699


def test_mantel_with_partial_single_permuted():
    """

    cannot formally test permutation results

    """
    data, _ = create_test_scattered()
    coords = create_test_coords()
    distances = pyssage.distances.euclidean_distance_matrix(coords[:, 0], coords[:, 1])
    angles = pyssage.distances.euclidean_angle_matrix(coords[:, 0], coords[:, 1])
    data_distances = pyssage.distances.data_distance_matrix(data, pyssage.distances.data_distance_euclidean)

    r, p_value, output_text, _, _, _, _, _ = pyssage.mantel.mantel(data_distances, distances, [angles],
                                                                   permutations=100)
    for line in output_text:
        print(line)


def test_mantel_with_partial_multi():
    """

    Mantel Test
    Matrix 1: Data Distance Matrix 1
    Matrix 2: Data Distance Matrix 2
    Matrices held constant: Distance Matrix 1, Angle Matrix 1
      Matrices are 355 x 355

    Observed Z = 44597376.68626
    Correlation = 0.89973
      t = 55.06521
      Left-tailed p = 1.00000
      Right-tailed p = 0.00000
      Two-tailed p = 0.00000

    """
    data, _ = create_test_scattered()
    coords = create_test_coords()
    distances = pyssage.distances.euclidean_distance_matrix(coords[:, 0], coords[:, 1])
    angles = pyssage.distances.euclidean_angle_matrix(coords[:, 0], coords[:, 1])
    data_distances1 = pyssage.distances.data_distance_matrix(data, pyssage.distances.data_distance_euclidean)
    data_distances2 = pyssage.distances.data_distance_matrix(data, pyssage.distances.data_distance_manhattan)

    r, p_value, output_text, _, _, _, _, _ = pyssage.mantel.mantel(data_distances1, data_distances2,
                                                                   [distances, angles])
    for line in output_text:
        print(line)
    assert round(r, 5) == 0.89973


def test_mantel_with_partial_multi_permuted():
    """

    cannot formally test permutation results

    """
    data, _ = create_test_scattered()
    coords = create_test_coords()
    distances = pyssage.distances.euclidean_distance_matrix(coords[:, 0], coords[:, 1])
    angles = pyssage.distances.euclidean_angle_matrix(coords[:, 0], coords[:, 1])
    data_distances1 = pyssage.distances.data_distance_matrix(data, pyssage.distances.data_distance_euclidean)
    data_distances2 = pyssage.distances.data_distance_matrix(data, pyssage.distances.data_distance_manhattan)

    r, p_value, output_text, _, _, _, _, _ = pyssage.mantel.mantel(data_distances1, data_distances2,
                                                                   [distances, angles], permutations=100)
    for line in output_text:
        print(line)


# def test_code():
#     x = numpy.array([[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]])
#     print()
#     print(x)
#     order = numpy.array([2, 0, 1, 3])
#     print()
#     print(x[numpy.ix_(order, order)])
#     print()
#     print(x)
#
#     rand_order = numpy.arange(10)
#     numpy.random.shuffle(rand_order)
#     print(rand_order)
#     assert True
