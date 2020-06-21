from typing import Optional
from pyssage.classes import Number, _DEF_CONNECTION
import pyssage.distcon
import pyssage.utils
import matplotlib.pyplot as pyplot
from matplotlib.lines import Line2D
# import matplotlib.patches as mpatches
import numpy


def draw_transect(transect: numpy.array, unit_scale: Number = 1, title: str = "") -> None:
    x = [i*unit_scale for i in range(len(transect))]
    fig, axs = pyplot.subplots()
    axs.plot(x, transect)
    axs.set_xlabel("Position")
    axs.set_ylabel("Value")
    if title != "":
        axs.set_title(title)
    pyplot.show()


def draw_quadvar_result(quadvar: numpy.ndarray, rand_ci: Optional[numpy.ndarray] = None, title: str = "",
                        varlabel: str = "", randlabel: str = "") -> None:
    fig, axs = pyplot.subplots()
    axs.plot(quadvar[:, 0], quadvar[:, 1], label=varlabel)
    if rand_ci is not None:
        axs.plot(quadvar[:, 0], rand_ci, label=randlabel)
        pyplot.legend(loc="upper right")
    axs.set_xlabel("Scale")
    axs.set_ylabel("Variance")
    if title != "":
        axs.set_title(title)
    pyplot.show()


# def draw_triangles(triangles: list, coords, title: str = "") -> None:
#     # this was for debugging purposes only
#     fig, axs = pyplot.subplots()
#     minx = min(coords[:, 0])
#     maxx = max(coords[:, 0])
#     miny = min(coords[:, 1])
#     maxy = max(coords[:, 1])
#     for t in triangles:
#         x = [p.x for p in t.points]
#         x.append(t.points[0].x)
#         y = [p.y for p in t.points]
#         y.append(t.points[0].y)
#         line = Line2D(x, y)
#         axs.add_line(line)
#         # p = mpatches.Circle((t.xc, t.yc), t.radius(), fill=False, color="red")
#         # axs.add_patch(p)
#     pyplot.scatter(coords[:, 0], coords[:, 1], color="black")
#     axs.set_xlim(minx-0.5, maxx+0.5)
#     axs.set_ylim(miny-0.5, maxy+0.5)
#     if title != "":
#         axs.set_title(title)
#     pyplot.show()


def draw_tessellation(tessellation, xcoords: numpy.ndarray, ycoords: numpy.ndarray, title: str = "") -> None:
    fig, axs = pyplot.subplots()
    minx = min(xcoords)
    maxx = max(xcoords)
    miny = min(ycoords)
    maxy = max(ycoords)
    for e in tessellation.edges:
        x = [e.start_vertex.x, e.end_vertex.x]
        y = [e.start_vertex.y, e.end_vertex.y]
        line = Line2D(x, y)
        axs.add_line(line)
    pyplot.scatter(xcoords, ycoords, color="black")
    axs.set_xlim(minx-1, maxx+1)
    axs.set_ylim(miny-1, maxy+1)
    if title != "":
        axs.set_title(title)
    pyplot.show()


def check_connection_format(con_frmt: str) -> None:
    valid_formats = ("boolmatrix", "binmatrix", "revbinmatrix", "pairlist")
    if con_frmt not in valid_formats:
        raise ValueError("{} is not a valid connection format".format(con_frmt))


def draw_connections(connections, xcoords: numpy.ndarray, ycoords: numpy.ndarray,
                     connection_frmt: str = _DEF_CONNECTION, title: str = ""):
    check_connection_format(connection_frmt)
    fig, axs = pyplot.subplots()
    minx = min(xcoords)
    maxx = max(xcoords)
    miny = min(ycoords)
    maxy = max(ycoords)
    if connection_frmt == "pairlist":
        for c in connections:
            p1 = c[0]
            p2 = c[1]
            x = [xcoords[p1], xcoords[p2]]
            y = [ycoords[p1], ycoords[p2]]
            line = Line2D(x, y, zorder=1)
            axs.add_line(line)
    else:
        n = len(xcoords)
        for i in range(n):
            for j in range(i):
                if (connection_frmt == "boolmatrix") and connections[i, j]:
                    connect = True
                elif (connection_frmt == "binmatrix") and (connections[i, j] == 1):
                    connect = True
                elif (connection_frmt == "revbinmatrix") and (connections[i, j] == 0):
                    connect = True
                else:
                    connect = False
                if connect:
                    x = [xcoords[i], xcoords[j]]
                    y = [ycoords[i], ycoords[j]]
                    line = Line2D(x, y, zorder=1)
                    axs.add_line(line)
    pyplot.scatter(xcoords, ycoords, color="black", zorder=2)
    axs.set_xlim(minx-1, maxx+1)
    axs.set_ylim(miny-1, maxy+1)
    if title != "":
        axs.set_title(title)
    pyplot.show()


def draw_shortest_path(connections, xcoords: numpy.ndarray, ycoords: numpy.ndarray, trace_dict: dict,
                       startp: int, endp: int, connection_frmt: str = _DEF_CONNECTION, title: str = ""):
    check_connection_format(connection_frmt)
    fig, axs = pyplot.subplots()
    minx = min(xcoords)
    maxx = max(xcoords)
    miny = min(ycoords)
    maxy = max(ycoords)
    if connection_frmt == "pairlist":
        for c in connections:
            p1 = c[0]
            p2 = c[1]
            x = [xcoords[p1], xcoords[p2]]
            y = [ycoords[p1], ycoords[p2]]
            line = Line2D(x, y, zorder=1)
            axs.add_line(line)
    else:
        n = len(xcoords)
        for i in range(n):
            for j in range(i):
                if (connection_frmt == "boolmatrix") and connections[i, j]:
                    connect = True
                elif (connection_frmt == "binmatrix") and (connections[i, j] == 1):
                    connect = True
                elif (connection_frmt == "revbinmatrix") and (connections[i, j] == 0):
                    connect = True
                else:
                    connect = False
                if connect:
                    x = [xcoords[i], xcoords[j]]
                    y = [ycoords[i], ycoords[j]]
                    line = Line2D(x, y, zorder=1)
                    axs.add_line(line)

    trace_path = pyssage.distcon.trace_path(startp, endp, trace_dict)
    for i in range(len(trace_path)-1):
        p1 = trace_path[i]
        p2 = trace_path[i+1]
        x = [xcoords[p1], xcoords[p2]]
        y = [ycoords[p1], ycoords[p2]]
        line = Line2D(x, y, color="red", zorder=3, lw=2)
        axs.add_line(line)

    pyplot.scatter(xcoords, ycoords, color="black", zorder=2)
    axs.set_xlim(minx-1, maxx+1)
    axs.set_ylim(miny-1, maxy+1)
    if title != "":
        axs.set_title(title)
    pyplot.show()


def draw_distance_class_distribution(dist_matrix: numpy.ndarray, dist_class: numpy.ndarray, title: str = ""):
    fig, axs = pyplot.subplots()
    distances = pyssage.utils.flatten_half(dist_matrix)
    distances.sort()
    total = len(distances)
    y = [i+1 for i in range(total)]
    r = []
    i = 0
    c = 0
    while c < len(dist_class) and i < total:
        if distances[i] >= dist_class[c, 1]:
            r.append(i+1)
            c += 1
        i += 1
    r.append(total)

    axs.plot(distances, y, zorder=1)
    for i in range(len(dist_class)):
        c = dist_class[i]
        upper = c[1]
        y = [0, r[i], r[i]]
        x = [upper, upper, 0]
        line = Line2D(x, y, color="red", zorder=2)
        axs.add_line(line)

    axs.set_xlabel("Distance")
    axs.set_ylabel("Cumulative Count")
    if title != "":
        axs.set_title(title)
    pyplot.show()
