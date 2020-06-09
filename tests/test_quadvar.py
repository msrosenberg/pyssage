import pyssage.quadvar


def test_wrap_transect():
    n = 100
    steps = 10
    # test wrapping positive
    answer = [95, 96, 97, 98, 99, 0, 1, 2, 3, 4]
    start = 95
    for i in range(steps):
        assert answer[i] == pyssage.quadvar.wrap_transect(start + i, n)
    # test wrapping negative
    answer = [4, 3, 2, 1, 0, 99, 98, 97, 96, 95]
    start = 4
    for i in range(steps):
        assert answer[i] == pyssage.quadvar.wrap_transect(start - i, n)
