from math import sqrt, pi, degrees
from typing import Optional, Tuple
import numpy
import scipy.stats
from pyssage.connections import Connections
from pyssage.utils import create_output_table


def check_variance_assumption(x: Optional[str]) -> None:
    valid = ("random", "normal", None)
    if x not in valid:
        raise ValueError(x + " is not a valid variance assumption. Valid values are: " +
                         ", ".join((str(i) for i in valid)))


def morans_i(y: numpy.ndarray, weights: Connections, alt_weights: Optional[numpy.ndarray] = None,
             variance: Optional[str] = "random"):
    check_variance_assumption(variance)
    n = len(y)
    mean_y = numpy.average(y)
    dev_y = y - mean_y  # deviations from mean
    w = weights.as_binary()
    if alt_weights is not None:  # multiply to create non-binary weights, if necessary
        w = w * alt_weights
    sumyij = numpy.sum(numpy.outer(dev_y, dev_y) * w, dtype=numpy.float64)
    sumy2 = numpy.sum(numpy.square(dev_y), dtype=numpy.float64)  # sum of squared deviations from mean
    sumw = numpy.sum(w, dtype=numpy.float64)  # sum of weight matrix
    sumw2 = sumw**2
    moran = n * sumyij / (sumw * sumy2)
    expected = -1 / (n - 1)
    if variance is None:
        sd, z, p = None, None, None
    else:
        s1 = numpy.sum(numpy.square(w + numpy.transpose(w))) / 2
        s2 = numpy.sum(numpy.square(numpy.sum(w, axis=0) + numpy.sum(w, axis=1)))
        if variance == "normal":
            v = ((n**2 * s1) - n*s2 + 3*sumw2) / ((n**2 - 1) * sumw2)
        else:  # random
            b2 = n * numpy.sum(numpy.power(dev_y, 4)) / (sumy2**2)
            v = ((n*((n**2 - 3*n + 3)*s1 - n*s2 + 3*sumw2) - b2*((n**2 - n)*s1 - 2*n*s2 + 6*sumw2)) /
                 ((n - 1)*(n - 2)*(n - 3)*sumw2)) - 1/(n - 1)**2
        sd = sqrt(v)  # convert to standard dev
        z = abs(moran - expected) / sd
        p = scipy.stats.norm.sf(z)*2  # two-tailed test

    return weights.min_scale, weights.max_scale, weights.n_pairs(), expected, moran, sd, z, p


def gearys_c(y: numpy.ndarray, weights: Connections, alt_weights: Optional[numpy.ndarray] = None,
             variance: Optional[str] = "random"):
    check_variance_assumption(variance)
    n = len(y)
    mean_y = numpy.average(y)
    dev_y = y - mean_y  # deviations from mean
    w = weights.as_binary()
    if alt_weights is not None:  # multiply to create non-binary weights, if necessary
        w *= alt_weights
    sumdif2 = numpy.sum(numpy.square(w * (dev_y[:, numpy.newaxis] - dev_y)), dtype=numpy.float64)
    sumy2 = numpy.sum(numpy.square(dev_y), dtype=numpy.float64)  # sum of squared deviations from mean
    sumw = numpy.sum(w, dtype=numpy.float64)  # sum of weight matrix
    sumw2 = sumw**2
    geary = (n - 1) * sumdif2 / (2 * sumw * sumy2)
    if variance is None:
        sd, z, p = None, None, None
    else:
        s1 = numpy.sum(numpy.square(w + numpy.transpose(w))) / 2
        s2 = numpy.sum(numpy.square(numpy.sum(w, axis=0) + numpy.sum(w, axis=1)))
        if variance == "normal":
            v = ((2*s1 + s2)*(n - 1) - 4*sumw2) / (2*(n + 1)*sumw2)
        else:  # random
            b2 = n * numpy.sum(numpy.power(dev_y, 4)) / (sumy2 ** 2)
            nn2n3 = n * (n - 2) * (n - 3)
            v = ((n - 1)*s1*(n**2 - 3*n + 3 - (n - 1)*b2) / (nn2n3*sumw2) -
                 (n - 1)*s2*(n**2 + 3*n - 6 - (n**2 - n + 2)*b2) / (4*nn2n3*sumw2) +
                 (n**2 - 3 - b2*(n - 1)**2) / nn2n3)
        sd = sqrt(v)  # convert to standard dev
        z = abs(geary - 1) / sd
        p = scipy.stats.norm.sf(z)*2  # two-tailed test

    return weights.min_scale, weights.max_scale, weights.n_pairs(), 1, geary, sd, z, p


def correlogram(data: numpy.ndarray, dist_class_connections: list, metric: morans_i,
                variance: Optional[str] = "random"):
    if metric == morans_i:
        metric_title = "Moran's I"
        exp_format = "f"
    elif metric == gearys_c:
        metric_title = "Geary's c"
        exp_format = "d"
    else:
        metric_title = ""
        exp_format = ""
    output = []
    for dc in dist_class_connections:
        output.append(metric(data, dc, variance=variance))

    # create basic output text
    output_text = list()
    output_text.append(metric_title + " Correlogram")
    output_text.append("")
    output_text.append("# of data points = {}".format(len(data)))
    if variance is not None:
        output_text.append("Distribution assumption = {}".format(variance))
    output_text.append("")
    col_headers = ("Min dist", "Max dist", "# pairs", "Expected", metric_title, "SD", "Z", "Prob")
    col_formats = ("f", "f", "d", exp_format, "f", "f", "f", "f")
    create_output_table(output_text, output, col_headers, col_formats)

    return output, output_text


def bearing_correlogram(data: numpy.ndarray, dist_class_connections: list, angles: numpy.ndarray, n_bearings: int = 18,
                        metric=morans_i, variance: Optional[str] = "random"):
    if metric == morans_i:
        metric_title = "Moran's I"
        exp_format = "f"
    elif metric == gearys_c:
        metric_title = "Geary's c"
        exp_format = "d"
    else:
        metric_title = ""
        exp_format = ""

    # calculate bearings and bearing weight matrices
    bearings = []
    bearing_weights = []
    for b in range(n_bearings):
        a = b * pi / n_bearings
        bearings.append(a)
        bearing_weights.append(numpy.square(numpy.cos(angles - a)))

    output = []
    for i, b in enumerate(bearing_weights):
        for dc in dist_class_connections:
            tmp_out = list(metric(data, dc, b, variance))
            tmp_out.insert(2, degrees(bearings[i]))
            output.append(tmp_out)

    # create basic output text
    output_text = list()
    output_text.append(metric_title + " Bearing Correlogram")
    output_text.append("")
    output_text.append("# of data points = {}".format(len(data)))
    if variance is not None:
        output_text.append("Distribution assumption = {}".format(variance))
    output_text.append("")
    col_headers = ("Min dist", "Max dist", "Bearing", "# pairs", "Expected", metric_title, "SD", "Z", "Prob")
    col_formats = ("f", "f", "f", "d", exp_format, "f", "f", "f", "f")
    create_output_table(output_text, output, col_headers, col_formats)

    return output, output_text
