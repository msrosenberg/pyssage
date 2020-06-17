import numpy
import pytest
import pyssage.distcon
import pyssage.graph
from tests.test_common import test_coords, load_answer


def test_delaunay_tessellation():
    coords = test_coords()
    tessellation, connections = pyssage.distcon.delaunay_tessellation(coords[:, 0], coords[:, 1])
    pyssage.graph.draw_tessellation(tessellation, coords, "tessellation")
    pyssage.graph.draw_connections(connections, coords, "connections as boolean matrix")
    # _, connections = pyssage.distcon.delaunay_tessellation(coords[:, 0], coords[:, 1], output_frmt="binmatrix")
    # pyssage.graph.draw_connections(connections, coords, connection_frmt="binmatrix",
    #                                title="connections as binary matrix")
    # _, connections = pyssage.distcon.delaunay_tessellation(coords[:, 0], coords[:, 1], output_frmt="revbinmatrix")
    # pyssage.graph.draw_connections(connections, coords, connection_frmt="revbinmatrix",
    #                                title="connections as reverse binary matrix")
    # _, connections = pyssage.distcon.delaunay_tessellation(coords[:, 0], coords[:, 1], output_frmt="pairlist")
    # pyssage.graph.draw_connections(connections, coords, connection_frmt="pairlist", title="connections as pair list")


def test_euc_dist_matrix():
    answer = load_answer("euc_distmat_answer.txt")

    coords = test_coords()
    output = pyssage.distcon.euc_dist_matrix(coords[:, 0], coords[:, 1])
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert answer[i, j] == round(output[i, j], 5)


def test_sph_dist_matrix():
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
    coords = test_coords()
    distances = pyssage.distcon.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.distcon.relative_neighborhood_network(distances)
    pyssage.graph.draw_connections(connections, coords)
    # assert False


def test_gabriel_network():
    coords = test_coords()
    distances = pyssage.distcon.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.distcon.gabriel_network(distances)
    pyssage.graph.draw_connections(connections, coords)
    # assert False


def test_minimum_spanning_tree():
    coords = test_coords()
    distances = pyssage.distcon.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.distcon.minimum_spanning_tree(distances)
    pyssage.graph.draw_connections(connections, coords)
    # assert False


def test_connect_distance_range():
    coords = test_coords()
    distances = pyssage.distcon.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.distcon.connect_distance_range(distances, mindist=10, maxdist=50)
    pyssage.graph.draw_connections(connections, coords)
    # assert False


def test_least_diagonal_network():
    coords = test_coords()
    distances = pyssage.distcon.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.distcon.least_diagonal_network(coords[:, 0], coords[:, 1], distances)
    pyssage.graph.draw_connections(connections, coords)
    # assert False


def test_nearest_neighbor_connections():
    coords = test_coords()
    distances = pyssage.distcon.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.distcon.nearest_neighbor_connections(distances, 1)
    pyssage.graph.draw_connections(connections, coords)
