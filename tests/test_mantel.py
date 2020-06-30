from tests.test_common import test_coords
import pyssage.distances
import pyssage.mantel


def test_residuals_from_simple_matrix_regression():
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
    coords = test_coords()
    distances = pyssage.distances.euc_dist_matrix(coords[:, 0], coords[:, 1])
    angles = pyssage.distances.euc_angle_matrix(coords[:, 0], coords[:, 1])
    r, p_value, output_text, _, _, _ = pyssage.mantel.mantel(distances, angles, [])
    for line in output_text:
        print(line)
    assert round(r, 5) == 0.29830
    assert round(p_value, 5) == 0.00000  # two-tailed p by default
