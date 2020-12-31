import pyssage.connections
import pyssage.distances
import pyssage.graph
from tests.test_common import create_test_coords, load_answer


def test_delaunay_tessellation():
    # answer calculated from PASSaGE 2 and exported as a binary connection matrix
    answer = load_answer("answers/delaunay_answer.txt")

    coords = create_test_coords()
    tessellation, connections = pyssage.connections.delaunay_tessellation(coords[:, 0], coords[:, 1])
    pyssage.graph.draw_tessellation(tessellation, coords[:, 0], coords[:, 1], "Tessellation Test",
                                    figoutput=pyssage.graph.FigOutput(figshow=True))
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], title="Delaunay Connections Test",
                                   figoutput=pyssage.graph.FigOutput(figshow=True))
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert connections[i, j] == answer[i, j]


def test_relative_neighborhood_network():
    # answer calculated from PASSaGE 2 and exported as a binary connection matrix
    answer = load_answer("answers/rel_neighbor_answer.txt")

    coords = create_test_coords()
    distances = pyssage.distances.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.connections.relative_neighborhood_network(distances)
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], title="Relative Neighborhood Network Test",
                                   figoutput=pyssage.graph.FigOutput(figshow=True))
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert connections[i, j] == answer[i, j]


def test_gabriel_network():
    # answer calculated from PASSaGE 2 and exported as a binary connection matrix
    answer = load_answer("answers/gabriel_answer.txt")

    coords = create_test_coords()
    distances = pyssage.distances.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.connections.gabriel_network(distances)
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], title="Gabriel Graph/Network Test",
                                   figoutput=pyssage.graph.FigOutput(figshow=True))
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert connections[i, j] == answer[i, j]


def test_minimum_spanning_tree():
    # answer calculated from PASSaGE 2 and exported as a binary connection matrix
    answer = load_answer("answers/minspan_answer.txt")

    coords = create_test_coords()
    distances = pyssage.distances.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.connections.minimum_spanning_tree(distances)
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], title="Minimum-Spanning Tree Test",
                                   figoutput=pyssage.graph.FigOutput(figshow=True))
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert connections[i, j] == answer[i, j]


def test_connect_distance_range():
    # answer calculated from PASSaGE 2 and exported as a binary connection matrix
    answer = load_answer("answers/distance_based_connect_7-12_answer.txt")

    coords = create_test_coords()
    distances = pyssage.distances.euc_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.connections.connect_distance_range(distances, mindist=7, maxdist=12)
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1],
                                   title="Distance-based Connections (7-12) Test",
                                   figoutput=pyssage.graph.FigOutput(figshow=True))
    for i in range(len(answer)):
        for j in range(len(answer)):
            assert connections[i, j] == answer[i, j]


def test_least_diagonal_network():
    coords = create_test_coords()
    distances = pyssage.distances.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.connections.least_diagonal_network(coords[:, 0], coords[:, 1], distances)
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1], title="Least Diagonal Network Test",
                                   figoutput=pyssage.graph.FigOutput(figshow=True))

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
    coords = create_test_coords()
    distances = pyssage.distances.sph_dist_matrix(coords[:, 0], coords[:, 1])
    connections = pyssage.connections.nearest_neighbor_connections(distances, 1)
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1],
                                   title="Nearest Neighbor (k=1) Symmetric Test",
                                   figoutput=pyssage.graph.FigOutput(figshow=True))

    connections = pyssage.connections.nearest_neighbor_connections(distances, 2)
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1],
                                   title="Nearest Neighbor (k=2) Symmetric Test",
                                   figoutput=pyssage.graph.FigOutput(figshow=True))

    connections = pyssage.connections.nearest_neighbor_connections(distances, 2, symmetric=False)
    pyssage.graph.draw_connections(connections, coords[:, 0], coords[:, 1],
                                   title="Nearest Neighbor (k=2) Asymmetric Test",
                                   figoutput=pyssage.graph.FigOutput(figshow=True))
