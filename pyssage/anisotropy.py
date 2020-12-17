from math import pi
from typing import Tuple
import numpy
import pyssage.mantel
from pyssage.utils import create_output_table, check_for_square_matrix

__all__ = ["bearing_analysis"]


def bearing_analysis(data: numpy.ndarray, distances: numpy.ndarray, angles: numpy.ndarray,
                     nbearings: int = 36) -> Tuple[list, list]:
    """
    Conduct a bearing analysis to test for anisotropic patterns in scattered data, method originally described in:

    Falsetti, A.B., and R.R. Sokal. 1993. Genetic structure of human populations in the British Isles. Annals of
    Human Biology 20:215-229.

    :param data: an n x n matrix representing distances among data values
    :param distances: an n x n matrix representing geographic distances among data points
    :param angles: an n x n matrix representing geographic angles among data points
    :param nbearings: the number of bearings to test; the default is 36 (every 5 degrees)
    :return: a tuple containing a list of output values and a list of text output
    """
    n = check_for_square_matrix(data)
    if (n != check_for_square_matrix(distances)) or (n != check_for_square_matrix(angles)):
        raise ValueError("input matrices must be the same size")

    angle_width = pi / nbearings
    output = []
    for a in range(nbearings):
        test_angle = a * angle_width
        b_matrix = distances * numpy.square(numpy.cos(angles - test_angle))
        r, p_value, _, _, _, _, _ = pyssage.mantel.mantel(data, b_matrix, [])
        output.append([a*180/nbearings, r, p_value])

    # create basic output text
    output_text = list()
    output_text.append("Bearing Analysis")
    output_text.append("")
    output_text.append("Tested {} vectors".format(nbearings))
    output_text.append("")
    col_headers = ("Bearing", "Correlation", "Prob")
    col_formats = ("f", "f", "f")
    create_output_table(output_text, output, col_headers, col_formats)
    return output, output_text
