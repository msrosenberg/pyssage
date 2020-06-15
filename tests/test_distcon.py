# import numpy
import pyssage.distcon
import pyssage.graph
from tests.test_common import test_coords, load_answer


def test_delaunay_tessellation():
    coords = test_coords()
    triangles, tessellation = pyssage.distcon.delaunay_tessellation(coords[:, 0], coords[:, 1])
    # pyssage.graph.draw_triangles(triangles, coords)
    pyssage.graph.draw_tessellation(tessellation, coords)

    print("# of polygons:", tessellation.npolygons())
    for i, p in enumerate(tessellation.polygons):
        print(p.name, p.nedges())


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
