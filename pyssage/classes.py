from typing import Union, Optional
from math import sqrt
import numpy


Number = Union[int, float]


class Point:
    def __init__(self, x: float = 0, y: float = 0, real: bool = True):
        self.x = x
        self.y = y
        self.real = real


class Triangle:
    def __init__(self, p1: Optional[Point] = None, p2: Optional[Point] = None, p3: Optional[Point] = None):
        self.points = [p1, p2, p3]
        self.xc = None
        self.yc = None
        self.center = None

    def x(self, i) -> float:
        if self.points[i] is None:
            return 0
        else:
            return self.points[i].x

    def y(self, i) -> float:
        if self.points[i] is None:
            return 0
        else:
            return self.points[i].y

    # def center(self):
    #     return Point(self.xc, self.yc)

    def find_circle(self) -> None:
        """
        Calculates the center of a circle that circumscribes the triangle
        Adapted from C code at www.ics.uci.edu/~eppstein/junkyard/circumcenter.html
        """
        # Use coordinates relative to point "a" of the triangle.
        xba = self.x(1) - self.x(0)
        yba = self.y(1) - self.y(0)
        xca = self.x(2) - self.x(0)
        yca = self.y(2) - self.y(0)
        # Squares of lengths of the edges incident to "a"
        balength = xba**2 + yba**2
        calength = xca**2 + yca**2
        # Calculate the denominator of the formulae
        denominator = 1 / (2 * (xba*yca - yba*xca))
        # Calculate offset (from "a") of circumcenter.
        xcirca = (yca*balength - yba*calength)*denominator
        ycirca = (xba*calength - xca*balength)*denominator
        self.xc = self.x(0) + xcirca
        self.yc = self.y(0) + ycirca
        self.center = Point(self.xc, self.yc)

    def radius(self) -> float:
        return sqrt((self.x(0) - self.xc)**2 + (self.y(0) - self.yc)**2)


class VoronoiEdge:
    def __init__(self):
        self.cw_predecessor = None
        self.ccw_predecessor = None
        self.cw_successor = None
        self.ccw_successor = None
        self.end_vertex = None
        self.start_vertex = None
        self.right_polygon = None
        self.left_polygon = None


class VoronoiPolygon:
    def __init__(self):
        self.infinity = False
        # self.vertices = []
        self.edges = []
        self.name = ""
        self.point = None

    def nvertices(self) -> int:
        # same as edges
        return len(self.edges)

    def nedges(self) -> int:
        return len(self.edges)

    def get_all_vertices(self) -> Optional[list]:
        """
        get vertices around polygon in order

        first tries the easier algortihm for internal; if external uses the more complicated check
        the infinite polyon returns None
        """

        # try standard algorithm for internal polygons
        if self.infinity:
            return None
        else:
            result = self.get_vertices()
            if result is None:
                # an external polygon, but not the infinite surrounding polygon
                edge = None
                previous_edge = None
                ordered_edges = [self.edges[0]]
                while edge != ordered_edges[0]:
                    if edge is None:
                        edge = self.edges[0]
                    tmp_edge = edge.cw_successor
                    if tmp_edge == previous_edge:
                        tmp_edge = edge.cw_predecessor
                    elif (tmp_edge.left_polygon != self) and (tmp_edge.right_polygon != self):
                        tmp_edge = edge.cw_predecessor
                    previous_edge = edge
                    edge = tmp_edge
                    ordered_edges.append(edge)
                vertices = []
                if (ordered_edges[0].start_vertex == ordered_edges[1].start_vertex) or \
                        (ordered_edges[0].start_vertex == ordered_edges[1].end_vertex):
                    vertices.append(ordered_edges[0].start_vetex)
                else:
                    vertices.append(ordered_edges[0].end_vetex)
                previous_vertex = vertices[0]
                for edge in ordered_edges:
                    if edge.end_vertex != previous_vertex:
                        vertices.append(edge.end_vertex)
                        previous_vertex = edge.end_vertex
                    else:
                        vertices.append(edge.start_vertex)
                        previous_vertex = edge.start_vertex
                return vertices
            else:
                return result

    def get_vertices(self) -> Optional[list]:
        """
        get vertices around polygon in order; only works for fully internal polygons
        """
        vertices = []
        edge = None
        start_edge = self.edges[0]
        do_quit = False
        while (not do_quit) and (edge != start_edge):
            if edge is None:
                edge = start_edge
            if edge.left_polygon == self:
                if not edge.end_vertex.real:
                    do_quit = True
                else:
                    vertices.append(edge.end_vertex)
                    edge = edge.cw_successor
            else:
                if not edge.start_vertex.real:
                    do_quit = True
                else:
                    vertices.append(edge.start_vertex)
                    edge = edge.cw_predecessor
        if do_quit:
            return None
        else:
            return vertices

    def area(self) -> Optional[float]:
        """
        area of the polygon, if it is internal. external polygons have no defined area and return None
        """
        vertices = self.get_vertices()
        if vertices is not None:
            area = 0
            nv = len(vertices)
            for i in range(nv - 1):
                area += vertices[i].x*vertices[i+1].y - vertices[i+1].x*vertices[i].y
            area += vertices[nv-1].x*vertices[0].y - vertices[0].x*vertices[nv-1].y
            return abs(area/2)
        else:
            return None

    def circumference(self) -> Optional[float]:
        """
        circumference of the polygon, if it is internal. external polygons have no defined circumference and return None
        """
        vertices = self.get_vertices()
        if vertices is not None:
            nv = len(vertices)
            circumference = 0
            for i in range(nv - 1):
                circumference += sqrt((vertices[i].x - vertices[i+1].x)**2 + (vertices[i].y - vertices[i+1].y)**2)
            circumference += sqrt((vertices[0].x - vertices[nv-1].x)**2 + (vertices[0].y - vertices[nv-1].y)**2)
            return circumference
        else:
            return None


class VoronoiTessellation:
    def __init__(self):
        self.vertices = []
        self.edges = []
        self.polygons = []

    def nvertices(self) -> int:
        return len(self.vertices)

    def nedges(self) -> int:
        return len(self.edges)

    def npolygons(self) -> int:
        return len(self.polygons)


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
        self.min_scale = 0
        self.max_scale = 0
        self.distances = None
        self.angles = None

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

    def mid_scale(self) -> float:
        return self.min_scale + (self.max_scale - self.min_scale)/2

    def n_pairs(self):
        """
        return the number of connected pairs of points

        for an asymmetric matrix, each connection counts as 1/2, so result may not be an integer
        """
        np = int(numpy.sum(self.as_binary()))
        if np % 2 == 0:  # return as an integer if possible
            return np // 2
        else:
            return np / 2

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
        output = numpy.zeros((self._n, self._n), dtype=float)
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
        output = numpy.ones((self._n, self._n), dtype=float)
        for i in range(self._n):
            output[i, i] = 0  # force diagonal to all zeroes
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
