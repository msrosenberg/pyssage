from typing import Optional, Tuple, Union
from math import sqrt, sin, cos, acos, pi, atan2, radians
import numpy
from pyssage.classes import Number
from pyssage.connections import Connections
from pyssage.utils import flatten_half, euclidean_angle

# __all__ = ["euc_dist_matrix"]

_EARTH_RADIUS = 6371.0087714  # default radius of the Earth for spherical calculations


def euclidean_distance_matrix(x: numpy.ndarray, y: Optional[numpy.ndarray] = None,
                              z: Optional[numpy.ndarray] = None) -> numpy.ndarray:
    """
    Calculate a Euclidean distance matrix from coordinates in one, two, or three dimensions

    :param x: the coordinates along the x-axis
    :param y: the coordinates along the y-axis, if two or three dimensions (default = None)
    :param z: the coordinates along the z-axis, if three dimensions (default = None)
    :return: a square, symmetric matrix containing the calculated distances
    """
    n = len(x)
    if (y is not None) and (z is not None):
        nd = 3
        if (n != len(y)) or (n != len(z)):
            raise ValueError("Coordinate vectors must be same length")
    elif y is not None:
        nd = 2
        if n != len(y):
            raise ValueError("Coordinate vectors must be same length")
    elif z is not None:
        raise ValueError("Cannot process z-coordinates without y-coordinates")
    else:
        nd = 1
    output = numpy.zeros((n, n))
    for i in range(n):
        for j in range(i):
            dist = 0
            if nd == 1:
                dist = abs(x[i] - x[j])
            elif nd == 2:
                dist = sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            elif nd == 3:
                dist = sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2 + (z[i] - z[j])**2)
            output[i, j] = dist
            output[j, i] = dist
    return output


def spherical_distance(lat1: float, lat2: float, lon1: float, lon2: float,
                       earth_radius: float = _EARTH_RADIUS) -> float:
    """
    Returns the geodesic distance along the globe in km, for two points represented by longitudes and latitudes

    new version, reliant on fewer predetermined constants
    radius of earth is now an input parameter

    :param lat1: latitude of the first point
    :param lat2: latitude of the second point
    :param lon1: longitude of the first point
    :param lon2: longitude of the second point
    :param earth_radius: the radius of the Earth, the default value is the WGS84 average
    :return: the distance between the points, in units equivalent to those of earth_radius (default = kilometers)
    """
    # convert to radians
    # lon1 *= pi/180
    # lon2 *= pi/180
    # lat1 *= pi/180
    # lat2 *= pi/180
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    dlon = abs(lon1 - lon2)  # difference in longitudes
    angle = sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(dlon)
    if angle >= 1:
        return 0
    else:
        return acos(angle)*earth_radius


def spherical_distance_matrix(lon: numpy.ndarray, lat: numpy.ndarray,
                              earth_radius: float = _EARTH_RADIUS) -> numpy.ndarray:
    """
    construct an n x n matrix containing spherical distances from latitudes and longitudes

    :param lon: a single column containing the longitudes
    :param lat: a single column containing the latitudes
    :param earth_radius: the radius of the Earth, the default value is the WGS84 average
    :return: a square, symmetric matrix containing the calculated distances, in units equivalent to those of
             earth_radius (default = kilometers)
    """
    n = len(lon)
    if n != len(lat):
        raise ValueError("Coordinate vectors must be same length")
    output = numpy.zeros((n, n))
    for i in range(n):
        for j in range(i):
            dist = spherical_distance(lat[i], lat[j], lon[i], lon[j], earth_radius)
            output[i, j] = dist
            output[j, i] = dist
    return output


def spherical_angle(lat1: float, lat2: float, lon1: float, lon2: float, mode: str = "midpoint") -> float:
    """
    calculate the spherical angle between a pair of points

    :param lat1: the latititude of the first point
    :param lat2: the latitidue of the second point
    :param lon1: the longitude of the first point
    :param lon2: the longitude of the second point
    :param mode: a string representing the method for performing the calculation. Valid values are 'initial' and
                 'midpoint' (default)
    :return: the calculated angle as a floating point number, in radians
    """
    valid_modes = ("midpoint", "initial")
    if mode not in valid_modes:
        raise ValueError("invalid mode for spherical angle calculation")

    # convert to radians
    # lon1 *= pi/180
    # lon2 *= pi/180
    # lat1 *= pi/180
    # lat2 *= pi/180
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    dlon = lon2 - lon1
    if mode == "initial":  # initial bearing
        y = sin(dlon)*cos(lat2)
        x = cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(dlon)
    else:  # midpoint bearing
        # find midpoint
        bx = cos(lat2) * cos(dlon)
        by = cos(lat2) * sin(dlon)
        latmid = atan2(sin(lat1) + sin(lat2), sqrt((cos(lat1) + bx) ** 2 + by ** 2))
        lonmid = lon1 + atan2(by, cos(lat1) + bx)
        # shift longitude back into -pi to pi range (-180 to 180)
        while lonmid >= pi:
            lonmid -= 2 * pi
        while lonmid < -pi:
            lonmid += 2 * pi
        dlonmid = lonmid - lon1
        y = sin(dlonmid)*cos(latmid)
        x = cos(lat1)*sin(latmid) - sin(lat1)*cos(latmid)*cos(dlonmid)
    bearing = pi/2 - atan2(y, x)
    while bearing < 0:
        bearing += 2*pi
    while bearing >= 2*pi:
        bearing -= 2*pi
    return bearing


def spherical_angle_matrix(lon: numpy.ndarray, lat: numpy.ndarray, mode: str = "midpoint") -> numpy.ndarray:
    """
    construct an n by n matrix containing the angles (in radians) describing the spherical bearing between pairs of
    latitudes and longitudes

    this output is not symmetric as the bearing from point 1 to point 2 on the surface of a sphere is not simply
    the opposite of point 2 to point 1

    :param lon: a single column containing the longitudes
    :param lat: a single column containing the latitudes
    :param mode: a string stating whether the output should be "initial" bearing or "midpoint" bearing
                 (default = "midpoint")
    :return: a square matrix containing the bearings
    """
    n = len(lon)
    if n != len(lat):
        raise ValueError("Coordinate vectors must be same length")
    output = numpy.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                angle = spherical_angle(lat[i], lat[j], lon[i], lon[j], mode)
                output[i, j] = angle
    return output


def euclidean_angle_matrix(x: numpy.ndarray, y: numpy.ndarray, do360: bool = False) -> numpy.ndarray:
    """
    construct an n by n matrix containing the angles (in radians) describing the bearing between pairs of points

    by default these are stored in symmetric form with the angle between 0 and pi

    if the 360 degree option is chosen, the output matrix is no longer symmetric as the angle from i to j
    will be the opposite of the angle from j to i (e.g., 90 vs 270 or 0 vs 180)

    :param x: a single column containing the x-coordinates
    :param y: a single column containing the y-coordinates
    :param do360: a flag that controls whether the output should be over 180 degrees (default = False) or 360 degrees
    :return: a square matrix containing the angles. this matrix will be symmetric if do360 is False and asymmetric
             if true
    """
    n = len(x)
    if n != len(y):
        raise ValueError("Coordinate vectors must be same length")
    output = numpy.zeros((n, n))
    for i in range(n):
        for j in range(i):
            if i != j:
                if do360:
                    angle = euclidean_angle(x[i], y[i], x[j], y[j], do360=True)
                    output[i, j] = angle
                    angle = euclidean_angle(x[j], y[j], x[i], y[i], do360=True)
                    output[j, i] = angle
                else:
                    angle = euclidean_angle(x[i], y[i], x[j], y[j])
                    output[i, j] = angle
                    output[j, i] = angle
    return output


def shortest_path_distances(distances: numpy.ndarray, connections: Connections) -> Tuple[numpy.ndarray, dict]:
    """
    create a shortest-path/geodesic distance matrix from a set of inter-point distances and a connection/network
    scheme

    the connections must be given in the boolean matrix format

    This uses the Floyd-Warshall algorithm
    See Corman, T.H., Leiserson, C.E., and Rivest, R.L., 'Introduction to Algorithms', section 26.2, p. 558-562.

    trace_mat is a dictionary tracing the shortest path

    the algorithm will work on connection networks which are not fully spanning (i.e., there are no paths between
    some pairs of points), reporting infinity for the distance between such pairs

    :param distances: an n x n matrix containing distances among the n points
    :param connections: a Connections object containing connections or edges among the n points describing its
                        network
    :return: a tuple containing an n x n matrix with the shortest-path/geodesic distances and a dictionary
             containing trace data among the network for use in path reconstruction
    """
    n = len(distances)
    output = numpy.copy(distances)
    empty = numpy.invert(connections.as_boolean())
    # for the purposes of this algorithm, points must be connected to themselves
    for i in range(n):
        empty[i, i] = False
    trace_mat = {(i, j): j for i in range(n) for j in range(n)}
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if (not empty[i, k]) and (not empty[j, k]):
                    if empty[i, j]:
                        output[i, j] = output[i, k] + output[k, j]
                        empty[i, j] = False
                        trace_mat[i, j] = trace_mat[i, k]
                    else:
                        if output[i, j] > output[i, k] + output[k, j]:
                            output[i, j] = output[i, k] + output[k, j]
                            trace_mat[i, j] = trace_mat[i, k]
    # the following removes "connections" among point pairs with no connected path
    for i in range(n):
        for j in range(n):
            if i != j:  # points cannot be unconnected from themselves
                if (trace_mat[i, j] == j) and not connections[i, j]:
                    trace_mat.pop((i, j))  # remove path from trace matrix
                    output[i, j] = float("inf")  # change distance to infinity
    return output, trace_mat


def trace_path(i: int, j: int, trace_matrix: dict) -> list:
    """
    given the trace matrix from the geodesic distance/shortest path function, report the path between points i and j

    :param i: the index of the starting point
    :param j: the index of the ending point
    :param trace_matrix: a dictionary containing trace data output from shortest_path_distances()
    :return: a list indicating the path traveled from point i to point j, inclusive. if there is no such path in the
             network, the resultant list is empty
    """
    if (i, j) not in trace_matrix:  # no path between i and j
        return []
    else:
        output = [i]
        while i != j:
            i = trace_matrix[i, j]
            output.append(i)
        return output


def create_distance_classes(dist_matrix: numpy.ndarray, class_mode: str, mode_value: Number, 
                            set_max_dist: Union[float, str, None] = None) -> numpy.ndarray:
    """
    automatically create distance classes for a distance matrix based on one of four criteria:

    1. set class width: user sets a fixed width they desire for each class; the function determines how many classes
       are necessary
    2. set pair count: user sets a fixed count they desire for each class; the function determines how many classes
       are necessary. actual distance counts can vary from desired due to ties
    3. determine class width: the user sets the number of classes they desire; the function determines a width such
       that each class will have the same width
    4. determine pair count: the user sets the number of classes they desire; the function determines class boundaries
       so each class has the same number of distances. actual distance counts may vary due to ties

    the output is a two column numpy ndarray, representing lower and upper bounds of each class
    the lower bound is inclusive, the upper bound is exclusive. the algorithm will automatically increase the limit
    of the largest class by a tiny fraction, if necessary, to guarantee all distances are included in a class

    :param dist_matrix: an n x n matrix containing distances among the n points
    :param class_mode: a string specifying the mode used to create the distance classes. Valid values are
                       "set class width", "set pair count", "determine class width", "determine pair count"
    :param mode_value: an additional number whose specific meaning changes depending on the class_mode
    :param set_max_dist: an optional parameter one case use to set the maximum bound for class creation. Normally
                         classes are created up-to-and-including the largest observed distance in this matrix. Setting a
                         value for this parameter will restrict class creation up to (and including) that value. To
                         specify the maximum as a percentage of the observed max, enter the fraction as a string between
                         0 and 1 (e.g., "0.5" to indicate 50%). When a value is set all automatic divisions will only
                         include distances below the value. As an example, if one is requesting 10 classes of equal
                         width, those ten classes will be created as 1/10th of this set maximum value, rather than the
                         observed. Default = None (include all distances)
    :return: a two-column matrix where each row contains the lower and upper bounds (first and second columns,
             respectively) of each class. The lower bound is inclusive of the class, the upper bound is exclusive
    """
    maxadj = 1.0000001
    valid_modes = ("set class width", "set pair count", "determine class width", "determine pair count")
    if class_mode not in valid_modes:
        raise ValueError("Invalid class_mode")

    if set_max_dist is None:
        maxdist = numpy.max(dist_matrix)
    else:
        if isinstance(set_max_dist, str):
            p = eval(set_max_dist)
            if not (0 < p <= 1):
                raise ValueError()
            maxdist = p * numpy.max(dist_matrix)
        else:
            maxdist = set_max_dist

    limits = []
    nclasses = 0
    if "width" in class_mode:
        class_width = 0
        if class_mode == "set class width":
            class_width = mode_value
            nclasses = int(maxdist // class_width) + 1
        elif class_mode == "determine class width":
            nclasses = mode_value
            class_width = maxdist * maxadj / nclasses
        for c in range(nclasses):
            limits.append([c*class_width, (c+1)*class_width])

    elif "pair" in class_mode:
        distances = flatten_half(dist_matrix)  # only need to flatten half the matrix
        if set_max_dist is not None:  # filter by input maximum distance
            distances = distances[distances < maxdist]
        distances.sort()  # we only need to do this for these modes
        total = len(distances)
        pairs_per_class = 0
        if class_mode == "set pair count":
            pairs_per_class = mode_value
            nclasses = total // pairs_per_class
            if total % pairs_per_class != 0:
                nclasses += 1
        elif class_mode == "determine pair count":
            nclasses = mode_value
            pairs_per_class = total / nclasses

        lower = 0
        for c in range(nclasses):
            i = round((c+1) * pairs_per_class)
            if i >= total:
                upper = distances[total - 1] * maxadj
            else:
                upper = distances[i - 1]
            limits.append([lower, upper])
            lower = upper

    return numpy.array(limits)


def data_distance_euclidean(x: numpy.ndarray, y: numpy.ndarray) -> float:
    """
    returns the Euclidean distance between two vectors

    :param x: a one-dimensional numpy.ndarray
    :param y: a one-dimensional numpy.ndarray
    :return: a floating-point number representing the distance
    """
    return numpy.sqrt(numpy.sum(numpy.square(x-y)))


def data_distance_squared_euclidean(x: numpy.ndarray, y: numpy.ndarray) -> float:
    """
    returns the squared Euclidean distance between two vectors

    :param x: a one-dimensional numpy.ndarray
    :param y: a one-dimensional numpy.ndarray
    :return: a floating-point number representing the distance
    """
    return float(numpy.sum(numpy.square(x-y)))


def data_distance_manhattan(x: numpy.ndarray, y: numpy.ndarray) -> float:
    """
    returns the Manhattan distance between two vectors

    :param x: a one-dimensional numpy.ndarray
    :param y: a one-dimensional numpy.ndarray
    :return: a floating-point number representing the distance
    """
    return float(numpy.sum(numpy.abs(x - y)))


def data_distance_canberra(x: numpy.ndarray, y: numpy.ndarray) -> float:
    """
    returns the Canberra distance between two vectors

    :param x: a one-dimensional numpy.ndarray
    :param y: a one-dimensional numpy.ndarray
    :return: a floating-point number representing the distance
    """
    return float(numpy.sum(numpy.abs(x - y)/(x + y)))


def data_distance_hamming(x: numpy.ndarray, y: numpy.ndarray) -> float:
    """
    returns the Hamming distance between two vectors

    :param x: a one-dimensional numpy.ndarray
    :param y: a one-dimensional numpy.ndarray
    :return: a floating-point number representing the distance
    """
    return numpy.count_nonzero(x - y) / len(x)


def data_distance_jaccard(x: numpy.ndarray, y: numpy.ndarray) -> float:
    """
    returns the Jaccard distance between two vectors

    :param x: a one-dimensional numpy.ndarray
    :param y: a one-dimensional numpy.ndarray
    :return: a floating-point number representing the distance
    """
    cnt = 0
    for i in range(len(x)):
        if (x[i] != 0) or (y[i] != 0):
            cnt += 1
    return numpy.count_nonzero(x - y) / cnt


def data_distance_cosine(x: numpy.ndarray, y: numpy.ndarray) -> float:
    """
    returns the cosine distance between two vectors

    :param x: a one-dimensional numpy.ndarray
    :param y: a one-dimensional numpy.ndarray
    :return: a floating-point number representing the distance
    """
    return 1 - numpy.sum(x*y)/sqrt(numpy.sum(numpy.square(x)) * numpy.sum(numpy.square(y)))


def data_distance_czekanowski(x: numpy.ndarray, y: numpy.ndarray) -> float:
    """
    returns the Czekanowski distance between two vectors

    :param x: a one-dimensional numpy.ndarray
    :param y: a one-dimensional numpy.ndarray
    :return: a floating-point number representing the distance
    """
    return 1 - 2*numpy.sum(numpy.minimum(x, y))/numpy.sum(x + y)


def data_distance_correlation(x: numpy.ndarray, y: numpy.ndarray) -> float:
    """
    returns the correlation distance between two vectors

    :param x: a one-dimensional numpy.ndarray
    :param y: a one-dimensional numpy.ndarray
    :return: a floating-point number representing the distance
    """
    meanx = numpy.average(x)
    meany = numpy.average(y)
    num = numpy.sum((x - meanx)*(y - meany))
    den = sqrt(numpy.sum(numpy.square(x - meanx)) * numpy.sum(numpy.square(y - meany)))
    return 1 - num/den


def data_distance_squared_correlation(x: numpy.ndarray, y: numpy.ndarray) -> float:
    """
    returns the squared correlation distance between two vectors

    :param x: a one-dimensional numpy.ndarray
    :param y: a one-dimensional numpy.ndarray
    :return: a floating-point number representing the distance
    """
    meanx = numpy.average(x)
    meany = numpy.average(y)
    num = numpy.sum((x - meanx)*(y - meany))
    den = numpy.sum(numpy.square(x - meanx)) * numpy.sum(numpy.square(y - meany))
    return 1 - (num**2)/den


def data_distance_matrix(data: numpy.ndarray, distance_measure=data_distance_euclidean) -> numpy.ndarray:
    """
    calculate a distance matrix from an input matrix, where multivariates distances are calculated between rows of
    the input matrix. If the input matrix has r rows and columns, the output will be an r x r matrix

    any function which can accept a pair of numpy vectors (one-dimensional ndarrays) as input can be substituted
    for the distance measure. Euclidean distances is the default

    :param data: the input matrix as a numpy.ndarray, containing r rows and c columns
    :param distance_measure: the function via which to calculate the distances; the default is Euclidean distances
                             (data_distance_euclidean)
    :return: a square matrix containing the calculated distances among the rows of data
    """
    n = len(data)
    output = numpy.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i):
            dist = distance_measure(data[i, :], data[j, :])
            output[i, j] = dist
            output[j, i] = dist
    return output
