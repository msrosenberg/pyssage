from pyssage.classes import Number
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


def draw_quadvar_result(quadvar: numpy.array, title: str = "") -> None:
    fig, axs = pyplot.subplots()
    axs.plot(quadvar[:, 0], quadvar[:, 1])
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


def draw_tessellation(tessellation, coords, title: str = "") -> None:
    fig, axs = pyplot.subplots()
    minx = min(coords[:, 0])
    maxx = max(coords[:, 0])
    miny = min(coords[:, 1])
    maxy = max(coords[:, 1])
    for e in tessellation.edges:
        x = [e.start_vertex.x, e.end_vertex.x]
        y = [e.start_vertex.y, e.end_vertex.y]
        line = Line2D(x, y)
        axs.add_line(line)
    pyplot.scatter(coords[:, 0], coords[:, 1], color="black")
    axs.set_xlim(minx-1, maxx+1)
    axs.set_ylim(miny-1, maxy+1)
    if title != "":
        axs.set_title(title)
    pyplot.show()


def check_connection_format(con_frmt: str) -> None:
    valid_formats = ("boolmatrix", "binmatrix", "revbinmatrix", "pairlist")
    if con_frmt not in valid_formats:
        raise ValueError("{} is not a valid connection format".format(con_frmt))


def draw_connections(connections, coords, connection_frmt: str = "boolmatrix", title: str = ""):
    check_connection_format(connection_frmt)
    fig, axs = pyplot.subplots()
    minx = min(coords[:, 0])
    maxx = max(coords[:, 0])
    miny = min(coords[:, 1])
    maxy = max(coords[:, 1])
    if connection_frmt == "pairlist":
        for c in connections:
            p1 = coords[c[0]]
            p2 = coords[c[1]]
            x = [p1[0], p2[0]]
            y = [p1[1], p2[1]]
            line = Line2D(x, y)
            axs.add_line(line)
    else:
        n = len(coords)
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
                    x = [coords[i, 0], coords[j, 0]]
                    y = [coords[i, 1], coords[j, 1]]
                    line = Line2D(x, y)
                    axs.add_line(line)
    pyplot.scatter(coords[:, 0], coords[:, 1], color="black")
    axs.set_xlim(minx-1, maxx+1)
    axs.set_ylim(miny-1, maxy+1)
    if title != "":
        axs.set_title(title)
    pyplot.show()
