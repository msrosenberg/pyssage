from typing import Optional
from math import pi
from pyssage.classes import Number
import pyssage.connections
import pyssage.distances
import pyssage.utils
import matplotlib.pyplot as pyplot
from matplotlib.lines import Line2D
from matplotlib import collections
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


def add_connections_to_plot(axs, connections, xcoords: numpy.ndarray, ycoords: numpy.ndarray):
    if connections.is_symmetric():
        done = []
        for i in range(len(connections)):
            for j in connections.connected_from(i):
                if [i, j] not in done:  # trying to avoid duplicating the reverse line
                    x = [xcoords[i], xcoords[j]]
                    y = [ycoords[i], ycoords[j]]
                    line = Line2D(x, y, zorder=1)
                    axs.add_line(line)
                    done.append([j, i])
    else:
        for i in range(len(connections)):
            for j in connections.connected_from(i):
                if connections[j, i]:
                    if i < j:
                        arrow = "<|-|>"
                    else:
                        arrow = None
                else:
                    arrow = "-|>"
                if arrow is not None:
                    axs.annotate(s="", xytext=(xcoords[i], ycoords[i]), xy=(xcoords[j], ycoords[j]),
                                 arrowprops=dict(arrowstyle=arrow, edgecolor="C0"), zorder=1)


def draw_connections(connections, xcoords: numpy.ndarray, ycoords: numpy.ndarray, title: str = ""):
    minx = min(xcoords)
    maxx = max(xcoords)
    miny = min(ycoords)
    maxy = max(ycoords)
    fig, axs = pyplot.subplots()
    add_connections_to_plot(axs, connections, xcoords, ycoords)
    pyplot.scatter(xcoords, ycoords, color="black", zorder=2)
    axs.set_xlim(minx-1, maxx+1)
    axs.set_ylim(miny-1, maxy+1)
    if title != "":
        axs.set_title(title)
    pyplot.show()


def draw_shortest_path(connections, xcoords: numpy.ndarray, ycoords: numpy.ndarray, trace_dict: dict,
                       startp: int, endp: int, title: str = ""):
    minx = min(xcoords)
    maxx = max(xcoords)
    miny = min(ycoords)
    maxy = max(ycoords)
    fig, axs = pyplot.subplots()
    add_connections_to_plot(axs, connections, xcoords, ycoords)

    trace_path = pyssage.distances.trace_path(startp, endp, trace_dict)
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


def draw_correlogram(data: numpy.ndarray, metric_title: str = "", title: str = "", alpha: float = 0.05):
    # column order is: min_scale, max_scale, # pairs, expected, observed, sd, z, prob
    min_col = 0
    max_col = 1
    exp_col = 3
    obs_col = 4
    p_col = 7

    # plot at midpoint of distance range
    scale = numpy.array([x[min_col] + (x[max_col] - x[min_col])/2 for x in data])

    fig, axs = pyplot.subplots()
    # draw expected values
    y = [data[0, exp_col], data[0, exp_col]]
    x = [0, scale[len(scale)-1]]
    # x = [0, data[len(data)-1, scale_col]]
    line = Line2D(x, y, color="silver", zorder=1)
    axs.add_line(line)

    # draw base line
    axs.plot(scale, data[:, obs_col], zorder=2)
    # axs.plot(data[:, scale_col], data[:, obs_col], zorder=2)

    # mark significant scales
    sig_mask = [p <= alpha for p in data[:, p_col]]
    # x = data[sig_mask, scale_col]
    x = scale[sig_mask]
    y = data[sig_mask, obs_col]
    pyplot.scatter(x, y, color="black", edgecolors="black", zorder=3, s=25)

    # mark non-significant scales
    ns_mask = numpy.invert(sig_mask)
    # x = data[ns_mask, scale_col]
    x = scale[ns_mask]
    y = data[ns_mask, obs_col]
    pyplot.scatter(x, y, color="white", edgecolors="black", zorder=3, s=15)

    axs.set_xlabel("Scale")
    axs.set_ylabel(metric_title)
    if title != "":
        axs.set_title(title)
    pyplot.show()


def draw_bearing_correlogram(data: numpy.ndarray, title: str = "", symmetric: bool = True, alpha: float = 0.05):
    # column order is: min_scale, max_scale, bearing, # pairs, expected, observed, sd, z, prob
    mindist_col = 0
    b_col = 2
    exp_col = 4
    obs_col = 5
    p_col = 8
    dist_classes = sorted(set(data[:, mindist_col]))
    n_dists = len(dist_classes)
    deviation = data[:, obs_col] - data[:, exp_col]
    # one circle for each dist class, by ordinal rank, representing the expected value for that class
    base_circle = numpy.array([dist_classes.index(row[0])+1 for row in data])
    # the radius of each point is its base circle plus its deviation from its expectation
    r = base_circle + deviation
    # the angle (theta) is just the bearing that was tested
    theta = numpy.radians(data[:, b_col])
    drop_lines = [[(theta[i], base_circle[i]), (theta[i], r[i])] for i in range(len(r))]

    # mark positive and negative significant scales and angles
    sig_mask = [p <= alpha for p in data[:, p_col]]
    pos_mask = [i > 0 for i in data[:, obs_col]]
    neg_mask = numpy.invert(pos_mask)
    pos_mask = numpy.logical_and(sig_mask, pos_mask)  # combine to get positive significant
    neg_mask = numpy.logical_and(sig_mask, neg_mask)  # combine to get negative significant

    r_sig_pos = r[pos_mask]
    theta_sig_pos = theta[pos_mask]
    r_sig_neg = r[neg_mask]
    theta_sig_neg = theta[neg_mask]

    # mark non-significant scales and angles
    ns_mask = numpy.invert(sig_mask)
    r_ns = r[ns_mask]
    theta_ns = theta[ns_mask]

    if symmetric:
        # duplicate on opposite side of circle if drawing a full symmetric display
        r_sig_pos = numpy.append(r_sig_pos, r_sig_pos)
        theta_sig_pos = numpy.append(theta_sig_pos, theta_sig_pos + pi)
        r_sig_neg = numpy.append(r_sig_neg, r_sig_neg)
        theta_sig_neg = numpy.append(theta_sig_neg, theta_sig_neg + pi)
        r_ns = numpy.append(r_ns, r_ns)
        theta_ns = numpy.append(theta_ns, theta_ns + pi)
        for i in range(len(r)):
            drop_lines.append([(theta[i] + pi, base_circle[i]), (theta[i] + pi, r[i])])

    fig = pyplot.figure()
    axs = fig.add_subplot(projection="polar")

    drop_collection = collections.LineCollection(drop_lines, colors="silver", zorder=1)
    axs.add_collection(drop_collection)

    axs.scatter(theta_sig_pos, r_sig_pos, color="blue", edgecolors="black", zorder=3, s=15)
    axs.scatter(theta_sig_neg, r_sig_neg, color="red", edgecolors="black", zorder=3, s=15)
    axs.scatter(theta_ns, r_ns, color="white", edgecolors="black", zorder=3, s=5)
    pyplot.yticks(numpy.arange(1, n_dists+1))
    axs.set_yticklabels([])
    axs.set_ylim(0, n_dists+1)
    if not symmetric:
        axs.set_xlim(0, pi)

    if title != "":
        axs.set_title(title)
    pyplot.show()


def draw_windrose_correlogram(data: numpy.ndarray, low_data: numpy.ndarray, title: str = "", symmetric: bool = True,
                              alpha: float = 0.05):
    # column order is: min_scale, max_scale, min_angle, max_angle, # pairs, expected, observed, sd, z, prob
    mindist_col = 0
    b_col = 2
    exp_col = 4
    obs_col = 5
    p_col = 8
    annuli = set(data[:, mindist_col])
    annuli.add(low_data[:, mindist_col])
    annuli = sorted(annuli)
    n_annuli = len(annuli)

    # # one circle for each dist class, by ordinal rank, representing the expected value for that class
    # base_circle = numpy.array([dist_classes.index(row[0])+1 for row in data])
    # # the radius of each point is its base circle plus its deviation from its expectation
    # r = base_circle + deviation
    # # the angle (theta) is just the bearing that was tested
    # theta = numpy.radians(data[:, b_col])
    # drop_lines = [[(theta[i], base_circle[i]), (theta[i], r[i])] for i in range(len(r))]
    #
    # # mark positive and negative significant scales and angles
    # sig_mask = [p <= alpha for p in data[:, p_col]]
    # pos_mask = [i > 0 for i in data[:, obs_col]]
    # neg_mask = numpy.invert(pos_mask)
    # pos_mask = numpy.logical_and(sig_mask, pos_mask)  # combine to get positive significant
    # neg_mask = numpy.logical_and(sig_mask, neg_mask)  # combine to get negative significant
    #
    # r_sig_pos = r[pos_mask]
    # theta_sig_pos = theta[pos_mask]
    # r_sig_neg = r[neg_mask]
    # theta_sig_neg = theta[neg_mask]
    #
    # # mark non-significant scales and angles
    # ns_mask = numpy.invert(sig_mask)
    # r_ns = r[ns_mask]
    # theta_ns = theta[ns_mask]
    #
    # if symmetric:
    #     # duplicate on opposite side of circle if drawing a full symmetric display
    #     r_sig_pos = numpy.append(r_sig_pos, r_sig_pos)
    #     theta_sig_pos = numpy.append(theta_sig_pos, theta_sig_pos + pi)
    #     r_sig_neg = numpy.append(r_sig_neg, r_sig_neg)
    #     theta_sig_neg = numpy.append(theta_sig_neg, theta_sig_neg + pi)
    #     r_ns = numpy.append(r_ns, r_ns)
    #     theta_ns = numpy.append(theta_ns, theta_ns + pi)
    #     for i in range(len(r)):
    #         drop_lines.append([(theta[i] + pi, base_circle[i]), (theta[i] + pi, r[i])])
    #
    # fig = pyplot.figure()
    # axs = fig.add_subplot(projection="polar")
    #
    # drop_collection = collections.LineCollection(drop_lines, colors="silver", zorder=1)
    # axs.add_collection(drop_collection)
    #
    # axs.scatter(theta_sig_pos, r_sig_pos, color="blue", edgecolors="black", zorder=3, s=15)
    # axs.scatter(theta_sig_neg, r_sig_neg, color="red", edgecolors="black", zorder=3, s=15)
    # axs.scatter(theta_ns, r_ns, color="white", edgecolors="black", zorder=3, s=5)
    # pyplot.yticks(numpy.arange(1, n_dists+1))
    # axs.set_yticklabels([])
    # axs.set_ylim(0, n_dists+1)
    # if not symmetric:
    #     axs.set_xlim(0, pi)

    # if title != "":
    #     axs.set_title(title)
    pyplot.show()
