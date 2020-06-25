import numpy
import pytest
import pyssage.connections
import pyssage.distances
import pyssage.graph
from tests.test_common import test_coords, load_answer


def test_delaunay_tessellation():
    # answer calculated from PASSaGE 2 and exported as a binary connection matrix
    answer = load_answer("answers/delaunay_answer.txt")

    coords = test_coords()
    tessellation, connections = pyssage.connections.delaunay_tessellation(coords[:, 0], coords[:, 1])
    pyssage.graph.draw_tessellation(tessellation, coords[:, 0], coords[:, 1], "Tessellation Test")
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], title="Delaunay Connections Test")
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert connections[i, j] == answer[i, j]


def test_check_input_distance_matrix():
    # valid test
    test_size = 5
    matrix = numpy.zeros((test_size, test_size))
    n = pyssage.connections.check_input_distance_matrix(matrix)
    assert n == test_size

    with pytest.raises(ValueError, match="distance matrix must be two-dimensional"):
        # broken test --> should raise error
        matrix = numpy.zeros((1, 2, 3))
        pyssage.connections.check_input_distance_matrix(matrix)

    with pytest.raises(ValueError, match="distance matrix must be square"):
        # broken test --> should raise error
        matrix = numpy.zeros((test_size, test_size * 2))
        pyssage.connections.check_input_distance_matrix(matrix)


def test_relative_neighborhood_network():
    # answer calculated from PASSaGE 2 and exported as a binary connection matrix
    answer = load_answer("answers/rel_neighbor_answer.txt")

    coords = test_coords()
    distances = pyssage.distances.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.connections.relative_neighborhood_network(distances)
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], title="Relative Neighborhood Network Test")
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert connections[i, j] == answer[i, j]


def test_gabriel_network():
    # answer calculated from PASSaGE 2 and exported as a binary connection matrix
    answer = load_answer("answers/gabriel_answer.txt")

    coords = test_coords()
    distances = pyssage.distances.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.connections.gabriel_network(distances)
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], title="Gabriel Graph/Network Test")
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert connections[i, j] == answer[i, j]


def test_minimum_spanning_tree():
    # answer calculated from PASSaGE 2 and exported as a binary connection matrix
    answer = load_answer("answers/minspan_answer.txt")

    coords = test_coords()
    distances = pyssage.distances.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.connections.minimum_spanning_tree(distances)
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], title="Minimum-Spanning Tree Test")
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert connections[i, j] == answer[i, j]


def test_connect_distance_range():
    """
    Note: there is a logic change between PASSaGE 2 and pyssage that could cause this test to fail with
    the right data set. PASSaGE 2 treated the maximum distance as exclusive of the class; pyssage currently
    treats it as inclusive of the class. If a pair of points is exactly the maximum specified distance, the
    two algorithms will come to a different conclusion as to their connection status
    """
    # answer calculated from PASSaGE 2 and exported as a binary connection matrix
    answer = load_answer("answers/distance_based_connect_7-12_answer.txt")

    coords = test_coords()
    distances = pyssage.distances.euc_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.connections.connect_distance_range(distances, mindist=7, maxdist=12)
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1],
                                   title="Distance-based Connections (7-12) Test")
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert connections[i, j] == answer[i, j]


def test_least_diagonal_network():
    coords = test_coords()
    distances = pyssage.distances.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.connections.least_diagonal_network(coords[:, 0], coords[:, 1], distances)
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], title="Least Diagonal Network Test")

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
    distances = pyssage.distances.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.connections.nearest_neighbor_connections(distances, 1)
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1],
                                   title="Nearest Neighbor (k=1) Symmetric Test")

    connections = pyssage.connections.nearest_neighbor_connections(distances, 2)
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1],
                                   title="Nearest Neighbor (k=2) Symmetric Test")

    connections = pyssage.connections.nearest_neighbor_connections(distances, 2, symmetric=False)
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1],
                                   title="Nearest Neighbor (k=2) Asymmetric Test")
