import numpy
import pytest
import pyssage.distcon
import pyssage.graph
from tests.test_common import test_coords, load_answer


def test_delaunay_tessellation():
    # answer calculated from PASSaGE 2 and exported as a binary connection matrix
    answer = load_answer("delaunay_answer.txt")

    coords = test_coords()
    tessellation, connections = pyssage.distcon.delaunay_tessellation(coords[:, 0], coords[:, 1],
                                                                      output_frmt="binmatrix")
    pyssage.graph.draw_tessellation(tessellation, coords[:, 0], coords[:, 1], "Tessellation Test")
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], connection_frmt="binmatrix",
                                   title="Delaunay Connections Test")
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert answer[i, j] == connections[i, j]


def test_euc_dist_matrix():
    # answer calculated from PASSaGE 2 and exported to 5 decimals
    answer = load_answer("euc_distmat_answer.txt")

    coords = test_coords()
    output = pyssage.distcon.euc_dist_matrix(coords[:, 0], coords[:, 1])
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert answer[i, j] == round(output[i, j], 5)


def test_sph_dist_matrix():
    # answer calculated from PASSaGE 2 and exported to 5 decimals
    answer = load_answer("sph_distmat_answer.txt")

    coords = test_coords()
    output = pyssage.distcon.sph_dist_matrix(coords[:, 0], coords[:, 1])
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert answer[i, j] == round(output[i, j], 5)


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
    answer = load_answer("rel_neighbor_answer.txt")

    coords = test_coords()
    distances = pyssage.distcon.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.distcon.relative_neighborhood_network(distances, output_frmt="binmatrix")
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], connection_frmt="binmatrix",
                                   title="Relative Neighborhood Network Test")
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert answer[i, j] == connections[i, j]


def test_gabriel_network():
    # answer calculated from PASSaGE 2 and exported as a binary connection matrix
    answer = load_answer("gabriel_answer.txt")

    coords = test_coords()
    distances = pyssage.distcon.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.distcon.gabriel_network(distances, output_frmt="binmatrix")
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], connection_frmt="binmatrix",
                                   title="Gabriel Graph/Network Test")
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert answer[i, j] == connections[i, j]


def test_minimum_spanning_tree():
    # answer calculated from PASSaGE 2 and exported as a binary connection matrix
    answer = load_answer("minspan_answer.txt")

    coords = test_coords()
    distances = pyssage.distcon.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.distcon.minimum_spanning_tree(distances, output_frmt="binmatrix")
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], connection_frmt="binmatrix",
                                   title="Minimum-Spanning Tree Test")
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert answer[i, j] == connections[i, j]


def test_connect_distance_range():
    # # answer2 calculated from PASSaGE 2 and exported as a binary connection matrix
    # answer = load_answer("distance_based_connect_25-100_answer.txt")

    coords = test_coords()
    distances = pyssage.distcon.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.distcon.connect_distance_range(distances, mindist=25, maxdist=100, output_frmt="binmatrix")
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], connection_frmt="binmatrix",
                                   title="Distance-based Connections (25-100) Test")
    # for i in range(len(answer)):
    #     for j in range(len(answer)):
    #         assert answer[i, j] == connections[i, j]
    #
    # answer = load_answer("distance_based_connect_50-150_answer.txt")
    # connections = pyssage.distcon.connect_distance_range(distances, mindist=50, maxdist=150, output_frmt="binmatrix")
    # pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], connection_frmt="binmatrix",
    #                                title="Distance-based Connections (50-150) Test")
    # for i in range(len(answer)):
    #     for j in range(len(answer)):
    #         assert answer[i, j] == connections[i, j]


def test_least_diagonal_network():
    # answer calculated from PASSaGE 2 and exported as a binary connection matrix
    answer = load_answer("least_diag_answer.txt")

    coords = test_coords()
    distances = pyssage.distcon.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.distcon.least_diagonal_network(coords[:, 0], coords[:, 1], distances, output_frmt="binmatrix")
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], connection_frmt="binmatrix",
                                   title="Least Diagonal Network Test")
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert answer[i, j] == connections[i, j]


def test_nearest_neighbor_connections():
    # answer calculated from PASSaGE 2 and exported as a binary connection matrix
    answer = load_answer("nearest_neighbor_1_answer.txt")

    coords = test_coords()
    distances = pyssage.distcon.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.distcon.nearest_neighbor_connections(distances, 1, output_frmt="binmatrix")
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], connection_frmt="binmatrix",
                                   title="Nearest Neighbor (k=1) Test")
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert answer[i, j] == connections[i, j]

    answer = load_answer("nearest_neighbor_2_answer.txt")
    connections = pyssage.distcon.nearest_neighbor_connections(distances, 2, output_frmt="binmatrix")
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], connection_frmt="binmatrix",
                                   title="Nearest Neighbor (k=1) Test")
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert answer[i, j] == connections[i, j]


def test_shortest_path_distances():
    # answer calculated from PASSaGE 2 and exported to 5 decimals
    answer = load_answer("shortest_path_minspan_answer.txt")

    coords = test_coords()
    distances = pyssage.distcon.sph_dist_matrix(coords[:, 0], coords[:, 1])

    # test a fully connected network
    connections = pyssage.distcon.minimum_spanning_tree(distances, output_frmt="boolmatrix")
    geodists, trace = pyssage.distcon.shortest_path_distances(distances, connections)
    pyssage.graph.draw_shortest_path(connections, coords[:, 0], coords[:, 1], trace, 0, 300,
                                     connection_frmt="boolmatrix")
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert round(answer[i, j], 0) == round(geodists[i, j], 0)
            # rounding differences are adding up; a number of the estimated distances are marginally different at
            # even 1 decimal place (just a single digit) (others are different by one digit at 2, 3, 4, or 5 decimals)

    # test a partially connected network
    connections = pyssage.distcon.nearest_neighbor_connections(distances, 1, output_frmt="boolmatrix")
    _, _ = pyssage.distcon.shortest_path_distances(distances, connections)
    # print()
    # for i in range(20):
    #     print(0, i, pyssage.distcon.trace_path(0, i, trace), geodists[0, i])
