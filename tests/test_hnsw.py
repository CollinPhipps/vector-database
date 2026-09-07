import numpy as np
import pytest

from vdb.store import HNSW, Node
from vdb.utils import MetricType


def _chain_graph():
    """Six points on a line, hand-wired as a chain 0-1-2-3-4-5 at layer 0.

    Used to exercise search_layer in isolation, independent of insert().
    """
    h = HNSW(dim=2, metric=MetricType.L2)
    for x in range(6):
        h.store.add(np.array([x, 0.0], dtype=np.float32))
        h.graph[x] = Node(vid=x, top_level=0)

    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]:
        h.graph[a].add_neighbor(b, 0)
        h.graph[b].add_neighbor(a, 0)
    return h


def test_search_layer_ef1_is_greedy_hill_climb():
    # ef=1 should walk the chain from vid 0 and settle on vid 4 (the nearest),
    # never getting stuck at an earlier node.
    h = _chain_graph()
    query = np.array([4.2, 0.0], dtype=np.float32)
    result = h.search_layer(query, entry_points=[0], ef=1, layer=0)
    assert result[0][1] == 4


def test_search_layer_ef3_returns_three_closest():
    h = _chain_graph()
    query = np.array([4.2, 0.0], dtype=np.float32)
    result = h.search_layer(query, entry_points=[0], ef=3, layer=0)
    assert sorted(vid for _, vid in result) == [3, 4, 5]


@pytest.fixture
def built_hnsw():
    np.random.seed(0)
    n, dim = 1000, 16
    h = HNSW(dim=dim, M=8, ef_construction=100, metric=MetricType.L2)
    data = np.random.randn(n, dim).astype(np.float32)
    h.build(data)
    return h, n


def test_build_structural_invariants(built_hnsw):
    h, n = built_hnsw
    assert len(h.graph) == n, "every inserted vector should be a node"

    for node in h.graph.values():
        assert len(node.neighbors[0]) <= h.M_max0, "layer-0 degree cap violated"

    isolated = [vid for vid, node in h.graph.items() if len(node.neighbors[0]) == 0]
    assert not isolated, "a working graph shouldn't leave nodes with zero links"


def test_recall_vs_brute_force(built_hnsw):
    h, n = built_hnsw
    k = 10

    np.random.seed(1)
    queries = np.random.randn(50, h.dim).astype(np.float32)

    total = 0.0
    for q in queries:
        truth = {vid for _, vid in h.store.flat_search(q, k, MetricType.L2)}
        approx = set(h.search(q, ef_search=50, k=k))
        total += len(truth & approx) / k

    recall = total / len(queries)
    assert recall > 0.85
