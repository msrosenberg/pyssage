from typing import Union, Optional
from math import sqrt

Number = Union[int, float]


class Point:
    def __init__(self, x: float = 0, y: float = 0):
        self.x = x
        self.y = y


class Triangle:
    def __init__(self, p1: Optional[Point] = None, p2: Optional[Point] = None, p3: Optional[Point] = None):
        self.points = [p1, p2, p3]
        self.xc = None
        self.yc = None

    def x(self, i):
        if self.points[i] is None:
            return 0
        else:
            return self.points[i].x

    def y(self, i):
        if self.points[i] is None:
            return 0
        else:
            return self.points[i].y

    def center(self):
        return Point(self.xc, self.yc)

    def find_circle(self):
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

    def radius(self):
        return sqrt((self.x(0) - self.xc)**2 + (self.y(0) - self.yc)**2)


class VoronoiEdge:
    def __init__(self):
        # self.cw_pre = 0
        # self.ccw_pre = 0
        # self.cw_suc = 0
        # self.ccw_suc = 0
        self.end_vertex = 0
        self.start_vertex = 0
        # self.right_polygon = 0
        # self.left_polygon = 0


class VoronoiTessellation:
    def __init__(self):
        # self.edge_around_polygon = []
        # self.edge_around_vertex = []
        # self.right_polygon = []
        # self.left_polygon = []
        # self.start_vertex = []
        # self.end_vertex = []
        # self.cw_predecessor = []
        # self.ccw_predecessor = []
        # self.cw_successor = []
        # self.ccw_successor = []
        # self.vertex_w = []
        # self.vertex_x = []
        # self.vertex_y = []
        self.vertices = []
        self.edges = []

    def start_vertex(self, i):
        return self.edges[i].start_vertex

    def end_vertex(self, i):
        return self.edges[i].end_vertex
