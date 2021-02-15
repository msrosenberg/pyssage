from math import pi, sin, cos, atan2, sqrt, degrees, tan
from typing import Tuple
from collections import namedtuple
import pyssage.mantel
from pyssage.utils import create_output_table, check_for_square_matrix
from pyssage.common import OUT_FRMT
import numpy
import scipy.stats

__all__ = ["angular_correlation_analysis", "bearing_analysis"]


def bearing_analysis(data: numpy.ndarray, distances: numpy.ndarray, angles: numpy.ndarray, nbearings: int = 36,
                     npermutations: int = 0) -> Tuple[list, list]:
    """
    Conduct a bearing analysis to test for anisotropic patterns in scattered data, method originally described in:

    Falsetti, A.B., and R.R. Sokal. 1993. Genetic structure of human populations in the British Isles. Annals of
    Human Biology 20:215-229.

    :param data: an n x n matrix representing distances among data values
    :param distances: an n x n matrix representing geographic distances among data points
    :param angles: an n x n matrix representing geographic angles among data points
    :param nbearings: the number of bearings to test; the default is 36 (every 5 degrees)
    :param npermutations: the number of random permutations used to test the correlations; the default is 0
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
        r, p_value, _, _, _, rand_p, _ = pyssage.mantel.mantel(data, b_matrix, [], npermutations)
        if npermutations > 0:
            output.append([a*180/nbearings, r, p_value, rand_p])
        else:
            output.append([a*180/nbearings, r, p_value])

    # create basic output text
    output_text = list()
    output_text.append("Bearing Analysis")
    output_text.append("")
    output_text.append("Tested {} vectors".format(nbearings))
    output_text.append("")
    if npermutations > 0:
        col_headers = ("Bearing", "Correlation", "Prob", "RandProb")
        col_formats = ("f", "f", "f", "f")
    else:
        col_headers = ("Bearing", "Correlation", "Prob")
        col_formats = ("f", "f", "f")
    create_output_table(output_text, output, col_headers, col_formats)
    bearing_output = namedtuple("bearing_output", ["output_values", "output_text"])
    return bearing_output(output, output_text)


def angular_correlation_analysis(x: numpy.ndarray, y: numpy.ndarray, v: numpy.ndarray,
                                 n_vectors: int = 360) -> Tuple[float, float, float, float, list, list]:
    """
    Conduct an angular correlation analysis, method originally described in:

    Simon, G. 1997. An angular version of spatial correlations, with exact significance tests. Geographical
    Analysis 29:267-278.

    :param x: a vector containing the x-coordinates
    :param y: a vector containing the y-coordinates
    :param v: a vector containing the data associated with each coordinate pair
    :param n_vectors: an integer representing the number of vectors to test; default = 360 (i.e., 1° intervals)

    :return: a tuple containing the maximum correlation, the angle of the maximum correlation, the F-statistic
             for this correlation, it's significance, a list containing an output table for the tested angles,
             and a list containing textual output for the analysis
    """
    if (len(x) != len(y)) or (len(x) != len(v)):
        raise ValueError("Input vectors must be same length")

    x_centered = x - numpy.mean(x)
    y_centered = y - numpy.mean(y)
    v_centered = v - numpy.mean(v)
    sum_xx = numpy.sum(numpy.square(x_centered))
    sum_yy = numpy.sum(numpy.square(y_centered))
    sum_vv = numpy.sum(numpy.square(v_centered))
    sum_xy = numpy.sum(numpy.multiply(x_centered, y_centered))
    sum_xv = numpy.sum(numpy.multiply(x_centered, v_centered))
    sum_yv = numpy.sum(numpy.multiply(y_centered, v_centered))
    output = []
    for a in range(n_vectors):
        theta = 2*pi*a/n_vectors
        cos_t = cos(theta)
        sin_t = sin(theta)
        r = (cos_t*sum_xv + sin_t*sum_yv) / sqrt((sum_xx*cos_t**2 + 2*sin_t*cos_t*sum_xy + sum_yy*sin_t**2)*sum_vv)
        output.append([degrees(theta), r])
    theta_max = degrees(atan2((sum_xv*sum_xy - sum_yv*sum_xx), (sum_yv*sum_xy - sum_xv*sum_yy)))
    while theta_max < 0:
        theta_max += 360
    r_max = sqrt((sum_yy*sum_xv**2 - 2*sum_xv*sum_yv*sum_xy + sum_xx*sum_yv**2)/((sum_xx*sum_yy - sum_xy**2)*sum_vv))
    n = len(x)
    f = ((n - 3) * r_max**2) / (2 * (1 - r_max**2))
    p_value = 1 - scipy.stats.f.cdf(f, 2, n-3)

    theta_ref = theta_max + 180
    if theta_ref > 360:
        theta_ref -= 360
    min_ang = min(theta_max, theta_ref)
    max_ang = max(theta_max, theta_ref)
    output_text = list()
    output_text.append("Angular Correlation Analysis")
    output_text.append("")
    output_text.append("Input data contain {} data points".format(n))
    output_text.append("")
    output_text.append("Maximum Correlation = " + format(r_max, OUT_FRMT))
    output_text.append("Bearing of Maximum Correlation = " + format(min_ang, OUT_FRMT) + "° / " +
                       format(max_ang, OUT_FRMT)+"°")
    output_text.append("F = " + format(f, OUT_FRMT))
    output_text.append("p = " + format(p_value, OUT_FRMT))
    output_text.append("")

    angular_corelation_output = namedtuple("angular_correlation_output", ["r_max", "theta_max", "F", "p",
                                                                          "output_values", "output_text"])
    return angular_corelation_output(r_max, theta_max, f, p_value, output, output_text)
