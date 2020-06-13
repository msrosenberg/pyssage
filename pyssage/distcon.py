from typing import Optional
from math import sqrt, sin, cos, acos, pi, atan
import numpy
from pyssage.classes import Point, Triangle, VoronoiEdge, VoronoiTessellation

# __all__ = ["euc_dist_matrix"]

CON1 = 24902.1483 * 1.60935
CON2 = 57.29577951
CONV = CON2 / 360


# def euc_dist(x1: float, x2: float, y1: float = 0, y2: float = 0, z1: float = 0, z2: float = 0) -> float:
#     return sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)


def euc_dist_matrix(x: numpy.ndarray, y: Optional[numpy.ndarray] = None,
                    z: Optional[numpy.ndarray] = None) -> numpy.ndarray:
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


def sph_dist(lat1: float, lat2: float, lon1: float, lon2: float) -> float:
    """
    Returns the geodesic distance along the globe in km, for two points represented by longitudes and latitudes
    """
    lon1 /= CON2
    lon2 /= CON2
    lat1 /= CON2
    lat2 /= CON2
    p = abs(lon1 - lon2)  # difference in longitudes
    angle = sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(p)
    if angle >= 1:
        return 0
    else:
        return acos(angle)*CON1*CONV


def sph_dist_matrix(lon: numpy.ndarray, lat: numpy.ndarray) -> numpy.ndarray:
    n = len(lon)
    if n != len(lat):
        raise ValueError("Coordinate vectors must be same length")
    output = numpy.zeros((n, n))
    for i in range(n):
        for j in range(i):
            dist = sph_dist(lat[i], lat[j], lon[i], lon[j])
            output[i, j] = dist
            output[j, i] = dist
    return output


def create_point_list(x: numpy.ndarray, y: numpy.ndarray) -> list:
    # turn coord arrays into a list of points
    point_list = []
    for i in range(len(x)):
        point_list.append(Point(x[i], y[i]))
    return point_list


def calculate_delaunay_triangles(x: numpy.ndarray, y: numpy.ndarray) -> list:
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
    t_point1 = Point(xmid - 2*dmax, ymid - dmax)
    t_point2 = Point(xmid, ymid + dmax)
    t_point3 = Point(xmid + 2*dmax, ymid - dmax)

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

    return triangle_list


def delaunay_tessellation(x: numpy.ndarray, y: numpy.ndarray):
    n = len(x)
    if n != len(y):
        raise ValueError("Coordinate vectors must be same length")

    triangles = calculate_delaunay_triangles(x, y)

    # create connections here

    # tessellation = calculate_tessellation(x, y, triangles)
    tessellation = calculate_tessellation(triangles)

    # return tessellation, connections
    return triangles, tessellation


"""
procedure DelaunayTessellation(Crds : TpasCoordinates; DoCon,DoTess : boolean; ConName,TessName : string);
var
   Triangles : TObjectList;
   ConMat : TpasBooleanMatrix;
   TessMat : TpasVoronoi;
   i,j,n : integer;
begin
     if DoTimeStamp then StartTimeStamp('Delaunay/Dirichlet Tessellation');
     Triangles := TObjectList.Create;
     n := Crds.N;
     case Crds.Dimensions of
          2 : CalcDelaunayTriangles(Crds,Triangles);
          3 : CalcDelaunayTetrahedra(Crds,Triangles);
     end;
     if ContinueProgress and DoCon then begin
        ConMat := TpasBooleanMatrix.Create(n);
        ConMat.MatrixName := ConName;
        { Set Connections Matrix }
        for i := 1 to Crds.N do
            for j := 1 to Crds.N do
                if Crds.IsGood[i] and Crds.IsGood[j] then
                   ConMat[i,j] := false;
        case Crds.Dimensions of
             2 : CalcDelaunayConnections(ConMat,Triangles);
             3 : CalcDelaunayConnections3D(ConMat,Triangles);
        end;
        if ContinueProgress then begin
           Data_AddData(ConMat);
           OutputAddLine('Delaunay connection matrix "' + ConName +
             '" constructed for coordinates "' + Crds.MatrixName +'".');
        end else ConMat.Free;
     end;
     if ContinueProgress and DoTess and (Crds.Dimensions = 2) then begin
        CalcTesselation(Crds,Triangles,TessMat);
        if ContinueProgress then begin
           TessMat.MatrixName := TessName;
           Data_AddData(TessMat);
           OutputAddLine('Dirichlet Tessellation "' + TessName +
             '" constructed for coordinates "' + Crds.MatrixName +'".');
        end else TessMat.Free;
     end;
     OutputAddBlankLine;
     Triangles.Free;
     if DoTimeStamp then EndTimeStamp;
end;



"""

"""

{ procedure to calculate connections from the triangles }
procedure CalcDelaunayConnections(CMat : TpasBooleanMatrix; TriList : TObjectList);
var
   ii,jj,i : integer;
   T : TpasTriangle;
begin
     ProgressRefresh(TriList.Count,'Making connections...');
     //if not HideinTray then ProgressForm.Show;
     // fill in connection matrix
     for i := 0 to TriList.Count - 1 do if ContinueProgress then begin
         T := TpasTriangle(TriList[i]);
         for ii := 1 to 2 do
             for jj := ii + 1 to 3 do
                 if (T.points[ii] <= CMat.n) and (T.points[jj] <= CMat.n) then begin
                    CMat[T.points[ii],T.points[jj]] := true;
                    CMat[T.points[jj],T.points[ii]] := true;
                 end;
         ProgressIncrement;
     end;
     ProgressClose;
end;

"""


# def euclidean_angle(x1: float, y1: float, x2: float, y2: float) -> float:
#     # This will calculate the angle between the two points
#     # The output ranges from 0 to pi (0 to 180 degrees)
#     dy = abs(y1 - y2)
#     dx = abs(x1 - x2)
#     if dy == 0:
#         return 0
#     elif dx == 0:
#         return pi / 2
#     else:
#         angle = atan(dy/dx)
#         if (y2 > y1) and (x2 > x1):
#             result = angle
#         elif (y2 > y1) and (x1 < x1):
#             result = pi - angle
#         elif (y2 < y1) and (x2 > x1):
#             result = pi - angle
#         elif (y2 < y1) and (x1 < x1):
#             result = angle
#         else:
#             return 0
#         while result > pi:
#             result -= pi
#         return result
#
#
# def euclidean_angle360(x1: float, y1: float, x2: float, y2: float) -> float:
#     # This will calculate the angle from point 1 to point 2
#     # The output ranges from 0 to 2pi (0 to 360 degrees)
#     dy = y2 - y1
#     dx = x2 - x1
#     if dy == 0:
#         if dx < 0:
#             return pi
#         else:
#             return 0
#     elif dx == 0:
#         if dy < 0:
#             return 3*pi/2
#         else:
#             return pi / 2
#     else:
#         angle = atan(abs(dy)/abs(dx))
#         if (dy > 0) and (dx > 0):
#             return angle
#         elif (dy > 0) and (dx < 0):
#             return pi - angle
#         elif (dy < 0) and (dx > 0):
#             return 2*pi - angle
#         elif (dy < 0) and (dx < 0):
#             return pi + angle
#         else:
#             return 0


# def calculate_tessellation(x: numpy.ndarray, y: numpy.ndarray, triangle_list: list):
def calculate_tessellation(triangle_list: list):
    # vertices and edges
    vertex_list = []
    edge_list = []
    for i, triangle1 in enumerate(triangle_list):
        vertex_list.append(triangle1.center())
        for j in range(i + 1, len(triangle_list)):
            triangle2 = triangle_list[j]
            cnt = 0
            # check to see if triangles have common edge
            for k in range(3):
                for kk in range(3):
                    if triangle1.points[kk] == triangle2.points[k]:
                        cnt += 1
            if cnt > 1:
                new_edge = VoronoiEdge()
                edge_list.append(new_edge)

                # find right and left polygons
                if triangle2.yc == triangle1.yc:
                    if triangle2.xc > triangle2.xc:
                        new_edge.start_vertex = triangle1.center()
                        new_edge.end_vertex = triangle2.center()
                    else:
                        new_edge.start_vertex = triangle2.center()
                        new_edge.end_vertex = triangle1.center()
                else:
                    if triangle2.yc > triangle2.yc:
                        new_edge.start_vertex = triangle1.center()
                        new_edge.end_vertex = triangle2.center()
                    else:
                        new_edge.start_vertex = triangle2.center()
                        new_edge.end_vertex = triangle1.center()
    tessellation = VoronoiTessellation()
    for v in vertex_list:
        tessellation.vertices.append(v)
    for e in edge_list:
        tessellation.edges.append(e)
    return tessellation
