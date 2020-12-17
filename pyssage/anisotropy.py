from math import pi
import numpy
import pyssage.mantel
from pyssage.utils import create_output_table


def bearing_analysis(data: numpy.ndarray, distances: numpy.ndarray, angles: numpy.ndarray, nbearings: int):
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
