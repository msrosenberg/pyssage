from typing import Union, Optional
from math import sqrt

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


