from typing import Tuple
from math import sqrt
import numpy
from pyssage.classes import Point, Triangle, VoronoiEdge, VoronoiTessellation, VoronoiPolygon
from pyssage.utils import euclidean_angle


class Connections:
    def __init__(self, n: int, symmetric: bool = True):
        self._symmetric = symmetric
        self._n = n
        self._connections = {i: set() for i in range(n)}

    def __len__(self):
        return self._n

    def __getitem__(self, item):
        i, j = item[0], item[1]
        if j in self._connections[i]:
            return True
        else:
            return False

    def is_symmetric(self) -> bool:
        return self._symmetric

    def connected_from(self, i) -> set:
        return self._connections[i]

    def connected_to(self, j) -> set:
        if self.is_symmetric():
            return self._connections[j]
        else:
            result = set()
            for i in range(self._n):
                if j in self._connections[i]:
                    result.add(i)
            return result

    def store(self, i: int, j: int) -> None:
        self._connections[i].add(j)
        if self._symmetric:
            self._connections[j].add(i)

    def as_boolean(self) -> numpy.ndarray:
        output = numpy.zeros((self._n, self._n), dtype=bool)
        for i in range(self._n):
            for j in self._connections[i]:
                output[i, j] = True
        return output

    def as_binary(self) -> numpy.ndarray:
        output = numpy.zeros((self._n, self._n), dtype=int)
        for i in range(self._n):
            for j in self._connections[i]:
                output[i, j] = 1
        return output

    def as_reverse_binary(self) -> numpy.ndarray:
        output = numpy.ones((self._n, self._n), dtype=int)
        for i in range(self._n):
            for j in self._connections[i]:
                output[i, j] = 0
        return output

    def as_pair_list(self) -> list:
        output = []
        for i in range(self._n):
            for j in self._connections[i]:
                if self.is_symmetric() and (i < j):  # if symmetric, do not output reverses
                    output.append([i, j])
                else:
                    output.append([i, j])
        return output

    def as_point_dict(self) -> dict:
        output = {}
        for i in range(self._n):
            output[i] = self._connections[i].copy()
        return output


def create_point_list(x: numpy.ndarray, y: numpy.ndarray) -> list:
    """
    create a list of Points from a pair of vectors containing x and y coordinates

    :param x: a single column containing the x-coordinates
    :param y: a single column containing the y-coordinates
    :return: a list containing Point objects represent each point
    """
    point_list = []
    for i in range(len(x)):
        point_list.append(Point(x[i], y[i]))
    return point_list


def calculate_delaunay_triangles(x: numpy.ndarray, y: numpy.ndarray) -> Tuple[list, list]:
    """
    Calculate the triangles that are used to form a delaunay tessellation or connection scheme

    Note: fails badly if multiple points are identical. need to trim duplicates in advance?
    """
    triangle_list = []
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    dx = xmax - xmin
    dy = ymax - ymin
    xmid = (xmax + xmin) / 2
    ymid = (ymax + ymin) / 2
    if dx > dy:
        dmax = dx
    else:
        dmax = dy
    # create extra-large triangle to encompass the entire observed set of points
    t_point1 = Point(xmid - 2*dmax, ymid - dmax, False)
    t_point2 = Point(xmid, ymid + dmax, False)
    t_point3 = Point(xmid + 2*dmax, ymid - dmax, False)

    triangle = Triangle(t_point1, t_point2, t_point3)
    triangle.find_circle()
    triangle_list.append(triangle)
    point_list = create_point_list(x, y)

    # add points one at a time
    for p in point_list:
        xp = p.x
        yp = p.y
        edge_list = []
        # Set up the edge list. If the point xp,yp is in the circle of each triangle, then the three edges of the
        # triangle are added to the buffer and the triangle is removed from the list
        remove_triangles = []
        # print(len(triangle_list))
        for triangle in triangle_list:
            # check to see if xp,yp is in the circle that circumscribes triangle. If xc,yc is the center of the
            # circle then if the distance between xc,yc and xp,yp is less than the radius of the circle,
            # it is inside.
            if sqrt((xp - triangle.xc)**2 + (yp - triangle.yc)**2) < triangle.radius():
                # if it is inside, add all of the edges of the triangle to our list, then get rid of the triangle
                for j in range(3):
                    for jj in range(j + 1, 3):
                        new_edge = (triangle.points[j], triangle.points[jj])
                        edge_list.append(new_edge)
                remove_triangles.append(triangle)
        for triangle in remove_triangles:
            triangle_list.remove(triangle)

        # Find repeated edges and get rid of them
        for j in range(len(edge_list) - 1):
            edge_1 = edge_list[j]
            for k in range(j + 1, len(edge_list)):
                edge_2 = edge_list[k]
                if (edge_1 is not None) and (edge_2 is not None):
                    if ((edge_1[0] == edge_2[0]) and (edge_1[1] == edge_2[1])) or \
                            ((edge_1[0] == edge_2[1]) and (edge_1[1] == edge_2[0])):
                        # set the repeated edges to None
                        edge_list[j] = None
                        edge_list[k] = None

        # Make new triangles for the current point, skipping over the removed edges.
        for edge in edge_list:
            if edge is not None:
                triangle = Triangle(edge[0], edge[1], p)
                triangle.find_circle()
                triangle_list.append(triangle)

    return triangle_list, point_list


def delaunay_tessellation(x: numpy.ndarray, y: numpy.ndarray) -> Tuple[VoronoiTessellation, Connections]:
    n = len(x)
    if n != len(y):
        raise ValueError("Coordinate vectors must be same length")

    triangles, points = calculate_delaunay_triangles(x, y)
    connections = delaunay_connections(triangles, points)
    tessellation = calculate_tessellation(triangles, points)

    return tessellation, connections


def calculate_tessellation(triangle_list: list, point_list: list) -> VoronoiTessellation:
    """
    calculate Delaunay tessellation from previous calculated triangles and store in a modified
    Voronoi data store, including info on polygons, edges, and vertices of the tessellation
    """
    # vertices and edges
    triangle_edges = {t.center: [] for t in triangle_list}
    vertex_list = []
    edge_list = []
    polygon_list = []
    point_to_polygon = {}
    for i, p in enumerate(point_list):
        new_polygon = VoronoiPolygon()
        polygon_list.append(new_polygon)
        new_polygon.point = p
        new_polygon.name = "Polygon surrounding point " + str(i)
        point_to_polygon[p] = new_polygon
    # create an extra polygon to represent the outside area
    inf_polygon = VoronoiPolygon()
    polygon_list.append(inf_polygon)
    inf_polygon.name = "Infinity"
    inf_polygon.infinity = True

    for i, triangle1 in enumerate(triangle_list):
        vertex_list.append(triangle1.center)
        for j in range(i + 1, len(triangle_list)):
            triangle2 = triangle_list[j]
            # check to see if triangles have common edge
            common_pnts = []
            for k in range(3):
                for kk in range(3):
                    if triangle1.points[kk] == triangle2.points[k]:
                        common_pnts.append(triangle1.points[kk])
            if len(common_pnts) > 1:
                new_edge = VoronoiEdge()
                edge_list.append(new_edge)
                # associate new edge with triangles
                triangle_edges[triangle1.center].append(new_edge)
                triangle_edges[triangle2.center].append(new_edge)
                # find right and left polygons
                if triangle2.yc == triangle1.yc:
                    if triangle2.xc > triangle2.xc:
                        new_edge.start_vertex = triangle1.center
                        new_edge.end_vertex = triangle2.center
                    else:
                        new_edge.start_vertex = triangle2.center
                        new_edge.end_vertex = triangle1.center
                    if common_pnts[0] not in point_list:  # one of the three boundary points
                        if common_pnts[1].y > triangle1.yc:
                            lp = common_pnts[1]
                            rp = common_pnts[0]
                        else:
                            lp = common_pnts[0]
                            rp = common_pnts[1]
                    else:
                        if common_pnts[0].y > triangle1.yc:
                            lp = common_pnts[0]
                            rp = common_pnts[1]
                        else:
                            lp = common_pnts[1]
                            rp = common_pnts[0]
                else:
                    if triangle2.yc > triangle1.yc:
                        new_edge.start_vertex = triangle1.center
                        new_edge.end_vertex = triangle2.center
                    else:
                        new_edge.start_vertex = triangle2.center
                        new_edge.end_vertex = triangle1.center
                    if common_pnts[0] not in point_list:
                        if euclidean_angle(triangle1.xc, triangle1.yc, common_pnts[1].x, common_pnts[1].y) > \
                                euclidean_angle(triangle1.xc, triangle1.yc, triangle2.xc, triangle2.yc):
                            lp = common_pnts[1]
                            rp = common_pnts[0]
                        else:
                            lp = common_pnts[0]
                            rp = common_pnts[1]
                    elif common_pnts[1] not in point_list:
                        if euclidean_angle(triangle1.xc, triangle1.yc, common_pnts[0].x, common_pnts[0].y) > \
                                euclidean_angle(triangle1.xc, triangle1.yc, triangle2.xc, triangle2.yc):
                            lp = common_pnts[0]
                            rp = common_pnts[1]
                        else:
                            lp = common_pnts[1]
                            rp = common_pnts[0]
                    else:
                        if common_pnts[0].x > common_pnts[1].x:
                            lp = common_pnts[1]
                            rp = common_pnts[0]
                        else:
                            lp = common_pnts[0]
                            rp = common_pnts[1]
                # attach edge to the right and left polygons
                if lp in point_to_polygon:
                    new_edge.left_polygon = point_to_polygon[lp]
                else:
                    new_edge.left_polygon = inf_polygon
                if rp in point_to_polygon:
                    new_edge.right_polygon = point_to_polygon[rp]
                else:
                    new_edge.right_polygon = inf_polygon
                new_edge.left_polygon.edges.append(new_edge)
                new_edge.right_polygon.edges.append(new_edge)

    # find CW and CCW edges of both start and end vertices for edges
    for edge in edge_list:
        # find predecessor edges
        other_edges = [e for e in triangle_edges[edge.start_vertex]]
        other_edges.remove(edge)
        if len(other_edges) == 2:
            edge1 = other_edges[0]
            edge2 = other_edges[1]
            edge_angle = euclidean_angle(edge.start_vertex.x, edge.start_vertex.y,
                                         edge.end_vertex.x, edge.end_vertex.y, do360=True)
            if edge1.start_vertex == edge.start_vertex:
                edge_angle1 = euclidean_angle(edge.start_vertex.x, edge.start_vertex.y, edge.end_vertex.x,
                                              edge.end_vertex.y, do360=True)
            else:
                edge_angle1 = euclidean_angle(edge.end_vertex.x, edge.end_vertex.y, edge.start_vertex.x,
                                              edge.start_vertex.y, do360=True)
            if edge2.start_vertex == edge.start_vertex:
                edge_angle2 = euclidean_angle(edge.start_vertex.x, edge.start_vertex.y, edge.end_vertex.x,
                                              edge.end_vertex.y, do360=True)
            else:
                edge_angle2 = euclidean_angle(edge.end_vertex.x, edge.end_vertex.y, edge.start_vertex.x,
                                              edge.start_vertex.y, do360=True)
            if edge_angle < edge_angle1:
                if (edge_angle1 < edge_angle2) or (edge_angle > edge_angle2):
                    edge.ccw_predecessor = edge1
                    edge.cw_predecessor = edge2
                else:
                    edge.ccw_predecessor = edge2
                    edge.cw_predecessor = edge1
            elif (edge_angle < edge_angle2) or (edge_angle1 > edge_angle2):
                edge.ccw_predecessor = edge2
                edge.cw_predecessor = edge1
            elif edge_angle1 < edge_angle2:
                edge.ccw_predecessor = edge1
                edge.cw_predecessor = edge2
        elif len(other_edges) == 1:
            edge.ccw_predecessor = other_edges[0]
            edge.cw_predecessor = other_edges[0]
        # find successor edges
        other_edges = [e for e in triangle_edges[edge.end_vertex]]
        other_edges.remove(edge)
        if len(other_edges) == 2:
            edge1 = other_edges[0]
            edge2 = other_edges[1]
            edge_angle = euclidean_angle(edge.start_vertex.x, edge.start_vertex.y,
                                         edge.end_vertex.x, edge.end_vertex.y, do360=True)
            if edge1.start_vertex == edge.end_vertex:
                edge_angle1 = euclidean_angle(edge.start_vertex.x, edge.start_vertex.y, edge.end_vertex.x,
                                              edge.end_vertex.y, do360=True)
            else:
                edge_angle1 = euclidean_angle(edge.end_vertex.x, edge.end_vertex.y, edge.start_vertex.x,
                                              edge.start_vertex.y, do360=True)
            if edge2.start_vertex == edge.end_vertex:
                edge_angle2 = euclidean_angle(edge.start_vertex.x, edge.start_vertex.y, edge.end_vertex.x,
                                              edge.end_vertex.y, do360=True)
            else:
                edge_angle2 = euclidean_angle(edge.start_vertex.x, edge.start_vertex.y, edge.end_vertex.x,
                                              edge.end_vertex.y, do360=True)
            if edge_angle < edge_angle1:
                if (edge_angle1 < edge_angle2) or (edge_angle > edge_angle2):
                    edge.ccw_successor = edge1
                    edge.cw_successor = edge2
                else:
                    edge.ccw_successor = edge2
                    edge.cw_successor = edge1
            elif (edge_angle < edge_angle2) or (edge_angle1 > edge_angle2):
                edge.ccw_successor = edge2
                edge.cw_successor = edge1
            elif edge_angle1 < edge_angle2:
                edge.ccw_successor = edge1
                edge.cw_successor = edge2
        elif len(other_edges) == 1:
            edge.ccw_predecessor = other_edges[0]
            edge.cw_predecessor = other_edges[0]

    tessellation = VoronoiTessellation()
    for v in vertex_list:
        tessellation.vertices.append(v)
    for e in edge_list:
        tessellation.edges.append(e)
    for p in polygon_list:
        tessellation.polygons.append(p)
    return tessellation


def delaunay_connections(triangle_list: list, point_list: list):
    n = len(point_list)
    output = Connections(n)
    for triangle in triangle_list:
        for i in range(3):
            p1 = triangle.points[i]
            for j in range(i):
                p2 = triangle.points[j]
                if (p1 in point_list) and (p2 in point_list):
                    output.store(point_list.index(p1), point_list.index(p2))
    return output


# def convert_connection_format(input_data, input_frmt: str, output_frmt: str):
#     """
#     convert from one connection format to another
#
#     if input format is a pair list, we assume that the largest indexed point is the number of points
#     (or put another way, we assume the last point is not unconnected from all of the others)
#     """
#     if input_frmt == "pairlist":
#         n = max(numpy.ndarray(input_data))
#     elif input_frmt == "pntdict":
#         n = max(input_data)
#     else:
#         n = len(input_data)
#     output = setup_connection_output(output_frmt, n)
#     if input_frmt == "pairlist":
#         for i in input_data:
#             store_connection(output, i[0], i[1], output_frmt)
#     elif input_frmt == "pntdict":
#         for i in input_data:
#             for j in input_data[i]:
#                 store_connection(output, i, j, output_frmt)
#     elif input_frmt == "boolmatrix":
#         for i in range(n):
#             for j in range(i):
#                 if input_data[i, j]:
#                     store_connection(output, i, j, output_frmt)
#     elif input_frmt == "binmatrix":
#         for i in range(n):
#             for j in range(i):
#                 if input_data[i, j] == 1:
#                     store_connection(output, i, j, output_frmt)
#     elif input_frmt == "revbinmatrix":
#         for i in range(n):
#             for j in range(i):
#                 if input_data[i, j] == 0:
#                     store_connection(output, i, j, output_frmt)
#     return output


# def setup_connection_output(output_frmt: str, n: int):
#     """
#     checks that the output format for a connection function is valid and returns the correct type of data storage
#     """
#     if output_frmt == "boolmatrix":
#         return numpy.zeros((n, n), dtype=bool)
#     elif output_frmt == "binmatrix":
#         return numpy.zeros((n, n), dtype=int)
#     elif output_frmt == "revbinmatrix":
#         return numpy.ones((n, n), dtype=int)
#     elif output_frmt == "pairlist":
#         return []
#     elif output_frmt == "pntdict":
#         return {}
#     else:
#         raise ValueError("{} is not a valid output format for connections".format(output_frmt))


# def store_connection(output, i: int, j: int, output_frmt: str):
#     """
#     stores a connection into the data storage, based on the specific format
#     automatically symmetrizes matrix storage
#     """
#     if output_frmt == "boolmatrix":
#         output[i, j] = True
#         output[j, i] = True
#     elif output_frmt == "binmatrix":
#         output[i, j] = 1
#         output[j, i] = 1
#     elif output_frmt == "revbinmatrix":
#         output[i, j] = 0
#         output[j, i] = 0
#     elif output_frmt == "pairlist":
#         output.append([i, j])
#     elif output_frmt == "pntdict":
#         output.setdefault(i, set()).add(j)
#         output.setdefault(j, set()).add(i)


def check_input_distance_matrix(distances: numpy.ndarray) -> int:
    if distances.ndim != 2:
        raise ValueError("distance matrix must be two-dimensional")
    elif distances.shape[0] != distances.shape[1]:
        raise ValueError("distance matrix must be square")
    else:
        return len(distances)


def relative_neighborhood_network(distances: numpy.ndarray):
    """
    calculate connections among points based on a relative neighborhood network
    """
    n = check_input_distance_matrix(distances)
    output = Connections(n)
    for i in range(n):
        for j in range(i):
            good = True
            for k in range(n):
                if (k != i) and (k != j):
                    if (distances[k, j] < distances[i, j]) and (distances[k, i] < distances[i, j]):
                        good = False
            if good:
                output.store(i, j)
    return output


def gabriel_network(distances: numpy.ndarray):
    """
    calculate connections among points based on a Gabriel network
    """
    n = check_input_distance_matrix(distances)
    output = Connections(n)
    sq_distances = numpy.square(distances)
    for i in range(n):
        for j in range(i):
            good = True
            for k in range(n):
                if (k != i) and (k != j):
                    if sq_distances[i, j] > sq_distances[k, j] + sq_distances[k, i]:
                        good = False
            if good:
                output.store(i, j)
    return output


def minimum_spanning_tree(distances: numpy.ndarray):
    """
    calculate connections among points based on a minimum spanning tree

    Although I invented this algorithm myself, it sort of follows the suggestion made in Kruskal, Joseph B., Jr. 1956.
    On the shortest spanning subtree of a graph and the traveling salesman problem.  Proceedings of the
    American Mathematical Society 7(1):48-50.
    """
    n = check_input_distance_matrix(distances)
    output = Connections(n)
    used = [i for i in range(n)]
    cnt = 1
    while cnt < n:
        new_point = cnt
        old_point = 0
        for i in range(cnt):
            for j in range(cnt, n):
                if distances[used[i], used[j]] < distances[used[old_point], used[new_point]]:
                    old_point, new_point = i, j
        # make connection
        output.store(used[old_point], used[new_point])
        used[cnt], used[new_point] = used[new_point], used[cnt]  # swap out a used point with an unused point
        cnt += 1
    return output


def connect_distance_range(distances: numpy.ndarray, maxdist: float, mindist: float = 0):
    """
    calculate connections based on a distance range, defined by maxdist and mindist

    points are not connected to themselves, even with a distance of zero
    """
    n = check_input_distance_matrix(distances)
    output = Connections(n)
    for i in range(n):
        for j in range(i):
            if mindist <= distances[i, j] <= maxdist:
                output.store(i, j)
    return output


def least_diagonal_network(x: numpy.ndarray, y: numpy.ndarray, distances: numpy.ndarray):
    """
    calculate connections among points based on a least diagonal network
    """
    n = check_input_distance_matrix(distances)
    if (n != len(x)) or (n != len(y)):
        raise ValueError("The coordinate arrays and the distance matrix must have the same length")
    output = Connections(n)
    # flatten distances into one dimension (half matrix only), but also track position in matrix
    dists = []
    for i in range(n):
        for j in range(i):
            dists.append([distances[i, j], i, j])
    dists.sort()
    good_pairs = []
    m1, m2 = 1, 1
    b1, b2 = 0, 0
    # work through all pairs from closest to farthest
    for d in dists:
        i, j = d[1], d[2]  # need the point indices, not the actual distance
        if x[i] != x[j]:
            vertical1 = False
            m1 = (y[i] - y[j]) / (x[i] - x[j])  # calculate slope
            b1 = y[i] - m1*x[i]  # calculate intercept
        else:
            vertical1 = True
        # compare to previously added links
        k = 0
        good = True
        while k < len(good_pairs):
            pair = good_pairs[k]
            pair1, pair2 = pair[0], pair[1]
            if (i not in pair) and (j not in pair):
                if x[pair1] != x[pair2]:
                    vertical2 = False
                    m2 = (y[pair1] - y[pair2]) / (x[[pair1]] - x[pair2])  # calculate slope
                    b2 = y[pair1] - m2*x[pair1]  # calculate intercept
                else:
                    vertical2 = True
                check = True
                xc, yc = x[i], y[j]  # defaults; likely unnecessary
                if vertical1 and vertical2:
                    # if both line segments are vertical, they overlap if either point of one pair is between both
                    # points of the other pair
                    check = False
                    if x[i] == x[pair1]:
                        if (y[i] < y[pair1] < y[j]) or (y[i] > y[pair1] > y[j]) or \
                                (y[i] < y[pair2] < y[j]) or (y[i] > y[pair2] > y[j]):
                            good = False
                elif vertical1:
                    # one segment is vertical; calculate the y at that x position
                    xc = x[i]
                    yc = m2*xc + b2
                elif vertical2:
                    # one segment is vertical; calculate the y at that x position
                    xc = x[pair1]
                    yc = m1*xc + b1
                elif m1 == m2:
                    # segments have identical slopes; can only overlap if they have identical projected intercepts
                    check = False
                    if b1 == b2:
                        # segments do have identical intercepts; they overlap if either point of one pair is between
                        # both points of the other pair
                        if (y[i] < y[pair1] < y[j]) or (y[i] > y[pair1] > y[j]) or \
                                (y[i] < y[pair1] < y[j]) or (y[i] > y[pair1] > y[j]):
                            good = False
                else:
                    xc = (b2 - b1) / (m1 - m2)
                    yc = m1*xc + b1
                if check:  # did not get pre-checked from one of the parallel slope cases above
                    # xc, yc is the projected crossing point of the two line segments; the segments overlap if
                    # this point falls within both segments
                    if (((x[i] <= xc <= x[j]) or (x[i] >= xc >= x[j])) and
                        ((y[i] <= yc <= y[j]) or (y[i] >= yc >= y[j]))) and \
                            (((x[pair1] <= xc <= x[pair2]) or (x[pair1] >= xc >= x[pair2])) and
                             ((y[pair1] <= yc <= y[pair2]) or (y[pair1] >= yc >= y[pair2]))):
                        good = False
            if good:
                k += 1
            else:
                k = len(good_pairs)
        if good:
            good_pairs.append([i, j])
    for pair in good_pairs:
        output.store(pair[0], pair[1])
    return output


def nearest_neighbor_connections(distances: numpy.ndarray, k: int = 1, symmetric: bool = True):
    """
    connect each point to it's k nearest neighbors

    individual points can be connected to more than k points because of ties and because a pair are not necessarily
    each other's nearest neighbors
    """
    n = check_input_distance_matrix(distances)
    output = Connections(n, symmetric)
    for i in range(n):
        dists = []
        for j in range(n):
            if j != i:
                dists.append([distances[i, j], j])
        dists.sort()
        c = k
        while dists[c][0] == dists[c+1][0]:  # this accounts for ties
            c += 1
        for p in range(c):  # connect the c closest points to the ith point
            output.store(i, dists[p][1])
    return output
