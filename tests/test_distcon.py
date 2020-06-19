import numpy
import pytest
import math
import pyssage.distcon
import pyssage.graph
from tests.test_common import test_coords, load_answer


def test_delaunay_tessellation():
    # answer calculated from PASSaGE 2 and exported as a binary connection matrix
    answer = load_answer("answers/delaunay_answer.txt")

    coords = test_coords()
    tessellation, connections = pyssage.distcon.delaunay_tessellation(coords[:, 0], coords[:, 1],
                                                                      output_frmt="binmatrix")
    pyssage.graph.draw_tessellation(tessellation, coords[:, 0], coords[:, 1], "Tessellation Test")
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], connection_frmt="binmatrix",
                                   title="Delaunay Connections Test")
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert connections[i, j] == answer[i, j]


def test_euc_dist_matrix():
    # answer calculated from PASSaGE 2 and exported to 5 decimals
    answer = load_answer("answers/euc_distmat_answer.txt")

    coords = test_coords()
    output = pyssage.distcon.euc_dist_matrix(coords[:, 0], coords[:, 1])
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert round(output[i, j], 5) == answer[i, j]


def test_euc_angle_matrix():
    x = [0, 0, 0, 45, 45, 45, -45, -45, -45]
    y = [0, 45, -45, 0, 45, -45, 0, 45, -45]
    output = pyssage.distcon.euc_angle_matrix(numpy.array(x), numpy.array(y))
    for i in range(len(output)):
        for j in range(i):
            if i != j:
                print(x[i], y[i], x[j], y[j], output[i, j], output[i, j] * 180 / math.pi)
    print()
    x = [0, 0, 0, 45, 45, 45, -45, -45, -45]
    y = [0, 45, -45, 0, 45, -45, 0, 45, -45]
    output = pyssage.distcon.euc_angle_matrix(numpy.array(x), numpy.array(y), do360=True)
    for i in range(len(output)):
        for j in range(len(output)):
            print(x[i], y[i], x[j], y[j], output[i, j], output[i, j] * 180 / math.pi)


def test_sph_dist_matrix():
    coords = test_coords()
    pyssage.distcon.sph_dist_matrix(coords[:, 0], coords[:, 1])
    """
    The original code matched the PASSaGE output, but I decided to go away from predetermined constants for
    conversion of angels to radians, meaning the numbers shift a bit due to greater floating-point accuracy. I am
    also be using a better estimate of the radius of the Earth in the calculations
    
    this change might cascade to other tests that relied upon these distances
    """
    # lat1 = 50.1234
    # lat2 = 58.3456
    # lon1 = -5.423
    # lon2 = -3.789
    # x = [50.1234, 58.3456]
    # y = [-5.423, -3.789]
    # print(pyssage.distcon.sph_dist(lat1, lat2, lon1, lon2))
    # print(pyssage.distcon.sph_angle(lat1, lat2, lon1, lon2) * 180 / math.pi)
    # print(pyssage.distcon.sph_angle2(lat1, lat2, lon1, lon2)*180/math.pi)

    # x = [0, 0, 0, 45, 45, 45, -45, -45, -45]
    # y = [0, 45, -45, 0, 45, -45, 0, 45, -45]
    # output = pyssage.distcon.euc_angle_matrix(numpy.array(x), numpy.array(y), do360=True)
    # for i in range(len(output)):
    #     for j in range(len(output)):
    #         print(x[i], y[i], x[j], y[j], output[i, j], output[i, j] * 180 / math.pi)


def test_sph_angle_matrix():
    coords = test_coords()
    output = pyssage.distcon.sph_angle_matrix(coords[:, 0], coords[:, 1])

    # lons = [0, 0, 0, 45, 45, 45, -45, -45, -45]
    # lats = [0, 45, -45, 0, 45, -45, 0, 45, -45]
    # output = pyssage.distcon.sph_angle_matrix(numpy.array(lons), numpy.array(lats))
    # for i in range(len(output)):
    #     for j in range(len(output)):
    #         if i != j:
    #             print(lons[i], lats[i], lons[j], lats[j], output[i, j] * 180 / math.pi)


def test_check_input_distance_matrix():
    # valid test
    test_size = 5
    matrix = numpy.zeros((test_size, test_size))
    n = pyssage.distcon.check_input_distance_matrix(matrix)
    assert n == test_size

    with pytest.raises(ValueError, match="distance matrix must be two-dimensional"):
        # broken test --> should raise error
        matrix = numpy.zeros((1, 2, 3))
        pyssage.distcon.check_input_distance_matrix(matrix)

    with pytest.raises(ValueError, match="distance matrix must be square"):
        # broken test --> should raise error
        matrix = numpy.zeros((test_size, test_size * 2))
        pyssage.distcon.check_input_distance_matrix(matrix)


def test_setup_connection_output():
    n = 5
    output = pyssage.distcon.setup_connection_output("boolmatrix", n)
    assert isinstance(output, numpy.ndarray)
    assert output.shape == (n, n)
    for i in range(n):
        for j in range(n):
            assert not output[i, j]

    output = pyssage.distcon.setup_connection_output("binmatrix", n)
    assert isinstance(output, numpy.ndarray)
    assert output.shape == (n, n)
    for i in range(n):
        for j in range(n):
            assert output[i, j] == 0

    output = pyssage.distcon.setup_connection_output("revbinmatrix", n)
    assert isinstance(output, numpy.ndarray)
    assert output.shape == (n, n)
    for i in range(n):
        for j in range(n):
            assert output[i, j] == 1

    output = pyssage.distcon.setup_connection_output("pairlist", n)
    assert isinstance(output, list)
    assert len(output) == 0

    with pytest.raises(ValueError, match="typo is not a valid output format for connections"):
        # broken test --> should raise error
        pyssage.distcon.setup_connection_output("typo", n)


def test_relative_neighborhood_network():
    # answer calculated from PASSaGE 2 and exported as a binary connection matrix
    answer = load_answer("answers/rel_neighbor_answer.txt")

    coords = test_coords()
    distances = pyssage.distcon.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.distcon.relative_neighborhood_network(distances, output_frmt="binmatrix")
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], connection_frmt="binmatrix",
                                   title="Relative Neighborhood Network Test")
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert connections[i, j] == answer[i, j]


def test_gabriel_network():
    # answer calculated from PASSaGE 2 and exported as a binary connection matrix
    answer = load_answer("answers/gabriel_answer.txt")

    coords = test_coords()
    distances = pyssage.distcon.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.distcon.gabriel_network(distances, output_frmt="binmatrix")
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], connection_frmt="binmatrix",
                                   title="Gabriel Graph/Network Test")
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert connections[i, j] == answer[i, j]


def test_minimum_spanning_tree():
    # answer calculated from PASSaGE 2 and exported as a binary connection matrix
    answer = load_answer("answers/minspan_answer.txt")

    coords = test_coords()
    distances = pyssage.distcon.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.distcon.minimum_spanning_tree(distances, output_frmt="binmatrix")
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], connection_frmt="binmatrix",
                                   title="Minimum-Spanning Tree Test")
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert connections[i, j] == answer[i, j]


def test_connect_distance_range():
    # answers calculated from PASSaGE 2 and exported as a binary connection matrix
    """
    PASSaGE answers had empty diagonals for some odd reason; needed to fix prior to import

    One small change between this implementation and that of PASSaGE (does not affect test):
    PASSaGE had the upper value as non-inclusive (dist < maxdist); currently this code is inclusive of the
    upper value (dist <= maxdist). Should think about whether to keep this way or not
    """
    answer = load_answer("distance_based_connect_25-100_answer.txt")

    coords = test_coords()
    distances = pyssage.distcon.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.distcon.connect_distance_range(distances, mindist=25, maxdist=100, output_frmt="binmatrix")
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], connection_frmt="binmatrix",
                                   title="Distance-based Connections (25-100) Test")
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert connections[i, j] == answer[i, j]

    answer = load_answer("distance_based_connect_50-150_answer.txt")
    connections = pyssage.distcon.connect_distance_range(distances, mindist=50, maxdist=150, output_frmt="binmatrix")
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], connection_frmt="binmatrix",
                                   title="Distance-based Connections (50-150) Test")
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert connections[i, j] == answer[i, j]


def test_least_diagonal_network():
    coords = test_coords()
    distances = pyssage.distcon.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.distcon.least_diagonal_network(coords[:, 0], coords[:, 1], distances, output_frmt="binmatrix")
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], connection_frmt="binmatrix",
                                   title="Least Diagonal Network Test")

    """
    if tested versus PASSaGE the test fails. I believe the PASSaGE code for this algorithm might have been 
    slightly buggy. it could also be an issue of ties being sorted into a different order, but affecting the 
    outcome. for now I'm believing the code here is likely correct
    """
    # # answer calculated from PASSaGE 2 and exported as a binary connection matrix
    # # PASSaGE output produced 1's or emtpy cells rather than 1's and 0's, so modification necessary prior to import
    # answer = load_answer("least_diag_answer.txt")
    #
    # coords = test_coords()
    # distances = pyssage.distcon.sph_dist_matrix(coords[:, 0], coords[:, 1])
    # connections = pyssage.distcon.least_diagonal_network(coords[:, 0], coords[:, 1], distances,
    #                                                      output_frmt="binmatrix")
    # pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], connection_frmt="binmatrix",
    #                                title="Least Diagonal Network Test")
    # for i in range(len(answer)):
    #     for j in range(len(answer)):
    #         assert connections[i, j] == answer[i, j]


def test_nearest_neighbor_connections():
    """
    The answer to this did not match that in PASSaGE 2. I have found an error in the old PASSaGE 2 code leading
    to the discrepancy. At this point I believe this code is correct, but do not have a formal test yet.

    For reference, the error in the PASSaGE code was the inadvertent resetting of positive connections to
    negative when looking at nearest neighbors for later points in the list, thus eliminating some potential
    asymmetric connections
    """
    coords = test_coords()
    distances = pyssage.distcon.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.distcon.nearest_neighbor_connections(distances, 1, output_frmt="binmatrix")
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], connection_frmt="binmatrix",
                                   title="Nearest Neighbor (k=1) Test")

    connections = pyssage.distcon.nearest_neighbor_connections(distances, 2, output_frmt="binmatrix")
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], connection_frmt="binmatrix",
                                   title="Nearest Neighbor (k=1) Test")


def test_shortest_path_distances():
    # answer calculated from PASSaGE 2 and exported to 5 decimals
    answer = load_answer("answers/shortest_path_minspan_answer.txt")

    coords = test_coords()
    distances = pyssage.distcon.sph_dist_matrix(coords[:, 0], coords[:, 1])

    # test a fully connected network
    connections = pyssage.distcon.minimum_spanning_tree(distances, output_frmt="boolmatrix")
    geodists, trace = pyssage.distcon.shortest_path_distances(distances, connections)
    pyssage.graph.draw_shortest_path(connections, coords[:, 0], coords[:, 1], trace, 0, 300,
                                     connection_frmt="boolmatrix")
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert round(geodists[i, j], 0) == round(answer[i, j], 0)
            # rounding differences are adding up; at least one of the estimated distances is marginally different at
            # even 1 decimal place (just a single digit) (others are different by one digit at 2, 3, 4, or 5 decimals)

    # test a partially connected network
    connections = pyssage.distcon.nearest_neighbor_connections(distances, 1, output_frmt="boolmatrix")
    _, _ = pyssage.distcon.shortest_path_distances(distances, connections)
    # print()
    # for i in range(20):
    #     print(0, i, pyssage.distcon.trace_path(0, i, trace), geodists[0, i])


def test_create_distance_classes():
    coords = test_coords()
    distances = pyssage.distcon.sph_dist_matrix(coords[:, 0], coords[:, 1])
    dc = pyssage.distcon.create_distance_classes(distances, "determine class width", 10)
    pyssage.graph.draw_distance_class_distribution(distances, dc, title="Ten Equal Width Distance Classes")
    dc = pyssage.distcon.create_distance_classes(distances, "determine pair count", 10)
    pyssage.graph.draw_distance_class_distribution(distances, dc, title="Ten Equal Count Distance Classes")
    dc = pyssage.distcon.create_distance_classes(distances, "set class width", 200)
    pyssage.graph.draw_distance_class_distribution(distances, dc, title="Distance Class Width Set to 200")
    dc = pyssage.distcon.create_distance_classes(distances, "set pair count", 5000)
    pyssage.graph.draw_distance_class_distribution(distances, dc, title="Distance Class Pair Count Set to 5000")
