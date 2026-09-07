import numpy as np

from vdb.benchmark import brute_force_ground_truth, build_hnsw, sweep_hnsw
from vdb.utils import MetricType


def test_benchmark_harness_recall_rises_with_ef_search():
    """Synthetic smoke test for the harness itself — no h5py/faiss dependency,
    just confirms build_hnsw/sweep_hnsw/brute_force_ground_truth wire together
    correctly and that recall improves as ef_search grows.
    """
    np.random.seed(0)
    n_train, n_test, dim, k = 2000, 100, 32, 10
    metric = MetricType.L2

    train = np.random.randn(n_train, dim).astype(np.float32)
    test = np.random.randn(n_test, dim).astype(np.float32)

    gt = brute_force_ground_truth(train, test, k, metric)
    index, perm, _ = build_hnsw(train, M=16, ef_construction=100, metric=metric)

    rows = sweep_hnsw(index, perm, test, gt, k, ef_values=[10, 25, 50, 100, 200])

    assert rows[-1]["recall"] > rows[0]["recall"]
    assert rows[-1]["recall"] > 0.9
