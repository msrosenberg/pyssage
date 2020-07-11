from typing import Tuple
from math import sqrt
import numpy
from pyssage.classes import Point, Triangle, VoronoiEdge, VoronoiTessellation, VoronoiPolygon
from pyssage.utils import euclidean_angle, check_for_square_matrix

__all__ = ["Connections", "delaunay_tessellation", "relative_neighborhood_network", "gabriel_network",
           "minimum_spanning_tree", "connect_distance_range", "least_diagonal_network", "nearest_neighbor_connections"]


class Connections:
    """
    A Connections object is a special container meant to keep track of connections or edges among pairs of points

    Connnections objects can be either inherrently symmetric (default) or asymmetric. This needs to be set at the
    time of construction, along with the number of points that are potentially connected to each other.

    The default space is no points are connected. Connections can be added using the store() method. Once added, a
    connection cannot be removed. You can check the status of a particular connection between points i and j by
    using them as a tuple key, Connections[i, j] where the result is a boolean about their connection status.

    The connections can be exported into a variety of formats using the as_xxxxx() methods, where xxxxx indicates
    the desired form.
    """
    def __init__(self, n: int, symmetric: bool = True):
        self._symmetric = symmetric
        self._n = n
        self._connections = {i: set() for i in range(n)}

    def __len__(self):
        """
        the size of a Connections object is the number of points, not the number of connected pairs
        """
        return self._n

    def __getitem__(self, item):
        """
        given a tuple key (i, j), where i and j are both integers, returns a boolean indicating whether
        point i is connected to point j
        """
        i, j = item[0], item[1]
        if j in self._connections[i]:
            return True
        else:
            return False

    def __repr__(self):
        return str(self.as_point_dict())

    def is_symmetric(self) -> bool:
        """
        designates whether the connections have been created as inherently symmetric
        """
        return self._symmetric

    def connected_from(self, i) -> set:
        """
        return the set of points that are connected from point i

        connected_from() and connected_to() will be identical if the connections are inherently symmetric
        """
        return self._connections[i]

    def connected_to(self, j) -> set:
        """
        return the set of points that are connected to point j

        connected_from() and connected_to() will be identical if the connections are inherently symmetric
        """
        if self.is_symmetric():
            return self._connections[j]
        else:
            result = set()
            for i in range(self._n):
                if j in self._connections[i]:
                    result.add(i)
            return result

    def store(self, i: int, j: int) -> None:
        """
        store a connection from point i to point j. if the connections are inherently symmetric, also store j to i
        """
        self._connections[i].add(j)
        if self._symmetric:
            self._connections[j].add(i)

    def as_boolean(self) -> numpy.ndarray:
        """
        return a expression of the connections as an n x n boolean matrix (numpy.ndarray)

        in this form True = connected and False = not connected
        """
        output = numpy.zeros((self._n, self._n), dtype=bool)
        for i in range(self._n):
            for j in self._connections[i]:
                output[i, j] = True
        return output

    def as_binary(self) -> numpy.ndarray:
        """
        return a expression of the connections as an n x n binary (0/1) matrix (numpy.ndarray)

        in this form 0 = not connected and 1 = connected
        """
        output = numpy.zeros((self._n, self._n), dtype=int)
        for i in range(self._n):
            for j in self._connections[i]:
                output[i, j] = 1
        return output

    def as_reverse_binary(self) -> numpy.ndarray:
        """
        return a expression of the connections as an n x n reverse binary (1/0) matrix (numpy.ndarray)

        in this form 1 = not connected and 0 = connected (this is useful when you need to use the connections as
        a form of "distance" such that you want connected items to have the smaller "distance" than unconnected
        items
        """
        output = numpy.ones((self._n, self._n), dtype=int)
        for i in range(self._n):
            for j in self._connections[i]:
                output[i, j] = 0
        return output

    def as_pair_list(self) -> list:
        """
        return a expression of the connections as a list of point pairs

        The primary list contains a series of sublists where each sublist contains the indices of two points
        that should be connected (e.g., [0, 2]).

        If the connections are asymmetric, the logic is from the first point to the second point. If the connection
        goes both ways, both [i, j] and [j, i] will be in the list.

        If the connections are symmetric, only one instance of the pair will be included (i.e, you will not see
        both [i, j] and [j ,i] in the list)
        """
        output = []
        for i in range(self._n):
            for j in self._connections[i]:
                if self.is_symmetric() and (i < j):  # if symmetric, do not output reverses
                    output.append([i, j])
                else:
                    output.append([i, j])
        return output

    def as_point_dict(self) -> dict:
        """
        return a expression of the connections a dictionary

        The dictionary will contain a key for every point, represented as the index of the point (0 to n-1)
        The value associated with each key is a set containing the points that the key is connected to. For example
        2: {0, 1, 3}

        would indicate that point 2 is connected to points 0, 1, and 3

        symmetry is not assumed, so the reverse connection would have to be found in the corresponding set, e.g.,
        0: {2}
        would show that point 0 is also connected to point 2
        """
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
    calculate the triangles that are used to form a delaunay tessellation or connection scheme

    Note: fails badly if multiple points are identical. need to trim duplicates in advance?

    this is not meant to be used as an independent algorithm; it is a piece of delaunay_tessellation()

    :param x: the x coordinates of n points
    :param y: the y coordinates of n points
    :return: returns a tuple containing two lists, one containing identified triangles and one containing the
             coordinates as distinct Point objects (needed for an intermediate analysis)
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
    """
    perform a Delaunay/Voronoi tessellation and return both the tessellation object and a connections object
    containing the corresponding triangles

    the method will likely crash if there are multiple points with identical x,y coordinates

    :param x: the x coordinates of n points
    :param y: the y coordinates of n points
    :return: returns a tuple containing a VoronoiTessellation object and a Connections object
    """
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

    this is not meant to be used as an independent algorithm; it is a piece of delaunay_tessellation()

    :param triangle_list: a list containing the trianngles identified by a previous function
    :param point_list: a list containing points assembled by a previous function
    :return: returns a VoronoiTessellation object
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


def delaunay_connections(triangle_list: list, point_list: list) -> Connections:
    """
    given a pre-determined list of triangles and points representing the triangle vertices, creates
    connections for all triangles

    this is not meant to be used as an independent algorithm; it is a piece of delaunay_tessellation()

    :param triangle_list: a list containing the trianngles identified by a previous function
    :param point_list: a list containing points assembled by a previous function
    :return: returns a Connection object
    """
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


def relative_neighborhood_network(distances: numpy.ndarray) -> Connections:
    """
    calculate connections among points based on a relative neighborhood network

    :param distances: an n x n matrix containing distances among points
    :return: returns a Connections object
    """
    n = check_for_square_matrix(distances)
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


def gabriel_network(distances: numpy.ndarray) -> Connections:
    """
    calculate connections among points based on a Gabriel network

    :param distances: an n x n matrix containing distances among points
    :return: returns a Connections object
    """
    n = check_for_square_matrix(distances)
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


def minimum_spanning_tree(distances: numpy.ndarray) -> Connections:
    """
    calculate connections among points based on a minimum spanning tree

    Although I invented this algorithm myself, it sort of follows the suggestion made in Kruskal, Joseph B., Jr. 1956.
    On the shortest spanning subtree of a graph and the traveling salesman problem.  Proceedings of the
    American Mathematical Society 7(1):48-50.

    :param distances: an n x n matrix containing distances among points
    :return: returns a Connections object
    """
    n = check_for_square_matrix(distances)
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


def connect_distance_range(distances: numpy.ndarray, maxdist: float, mindist: float = 0) -> Connections:
    """
    calculate connections based on a distance range, defined by maxdist and mindist

    points are not connected to themselves, even with a distance of zero

    :param distances: an n x n matrix containing distances among points
    :param maxdist: the maximum distance between points to connect. this distance is exclusive
    :param mindist: the minimum distance between points to connect (default = 0). this distance is inclusive
    :return: returns a Connections object
    """
    n = check_for_square_matrix(distances)
    output = Connections(n)
    for i in range(n):
        for j in range(i):
            if mindist <= distances[i, j] < maxdist:
                output.store(i, j)
    return output


def least_diagonal_network(x: numpy.ndarray, y: numpy.ndarray, distances: numpy.ndarray) -> Connections:
    """
    calculate connections among points based on a least diagonal network

    :param x: the x coordinates of n points
    :param y: the y coordinates of n points
    :param distances: an n x n matrix containing the distances among the points defined by x and y
    :return: returns a Connections object
    """
    n = check_for_square_matrix(distances)
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


def nearest_neighbor_connections(distances: numpy.ndarray, k: int = 1, symmetric: bool = True) -> Connections:
    """
    connect each point to it's k nearest neighbors

    individual points can be connected to more than k points because of ties and because a pair are not necessarily
    each other's nearest neighbors

    :param distances: an n x n matrix containing the distances among a set of points
    :param k: the number of nearest neighbors to connect
    :param symmetric: should connections always be symmetric (defualt = True) or allow asymmetric connections because
                      the nearest neighbor of one point A does not necessarily have A as it's nearest neighbor
    :return: returns a Connections object
    """
    n = check_for_square_matrix(distances)
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


def distance_classes_to_connections(dist_classes: numpy.ndarray, distances: numpy.ndarray) -> list:
    """
    create a list of connections objects corresponding to each distance class

    :param dist_classes: an n x 2 matrix containing the lower and upper bounds of each distance class
    :param distances: an n x n matrix containing the distances among a set of points
    :return: returns a list containing the connections
    """
    output = []
    for c in dist_classes:
        new_connections = connect_distance_range(distances, mindist=c[0], maxdist=c[1])
        output.append(new_connections)
    return output
