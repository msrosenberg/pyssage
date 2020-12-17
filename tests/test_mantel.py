from tests.test_common import create_test_coords
import pyssage.distances
import pyssage.mantel
# import numpy


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
    distances = pyssage.distances.euc_dist_matrix(coords[:, 0], coords[:, 1])
    angles = pyssage.distances.euc_angle_matrix(coords[:, 0], coords[:, 1])
    r, p_value, output_text, _, _, _, _ = pyssage.mantel.mantel(distances, angles, [])
    for line in output_text:
        print(line)
    assert round(r, 5) == 0.29830
    assert round(p_value, 5) == 0.00000  # two-tailed p by default


def test_mantel_with_permutation():
    """

    cannot formally test permutation results

    """
    coords = create_test_coords()
    distances = pyssage.distances.euc_dist_matrix(coords[:, 0], coords[:, 1])
    angles = pyssage.distances.euc_angle_matrix(coords[:, 0], coords[:, 1])
    r, p_value, output_text, _, _, _, _ = pyssage.mantel.mantel(distances, angles, [], permutations=100)
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
