"""
Benchmark harness for the HNSW index: load an ann-benchmarks dataset (or a subset),
recompute exact ground truth for the subset, build the index, and sweep ef_search to
trace a recall-vs-latency curve. FAISS's HNSW is included as the baseline to compare against.

Import these from a notebook; the __main__ block runs a synthetic smoke test that needs
neither h5py nor faiss, so the HNSW-side logic can be verified without the real dataset.
"""
import time
import numpy as np

from .store.hnsw import HNSW
from .utils.metrics import MetricType

# ann-benchmarks files tag their distance in an attribute; map it to our metric enum.
_DISTANCE_TO_METRIC = {
    "euclidean": MetricType.L2,
    "angular": MetricType.COSINE,
    "dot": MetricType.DOT,
}


def load_hdf5(path, n_train=None, n_test=None):
    """
    Load an ann-benchmarks HDF5 file. Returns (train, test, metric).

    n_train / n_test subset the arrays. NOTE: the file's precomputed `neighbors` are
    indices into the FULL train set, so they're invalid once you subset — recompute
    ground truth with brute_force_ground_truth() instead.
    """
    import h5py
    with h5py.File(path, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
        distance = f.attrs.get("distance", "euclidean")
    if n_train is not None:
        train = train[:n_train]
    if n_test is not None:
        test = test[:n_test]
    metric = _DISTANCE_TO_METRIC.get(distance, MetricType.L2)
    return train, test, metric


def _distances_to_all(train, query, metric):
    """Vectorized 'smaller = closer' distance from one query to every train row."""
    if metric == MetricType.L2:
        return np.sum((train - query) ** 2, axis=1)
    if metric == MetricType.COSINE:
        denom = np.linalg.norm(train, axis=1) * np.linalg.norm(query)
        denom = np.where(denom > 0, denom, 1.0)
        return 1.0 - (train @ query) / denom
    if metric == MetricType.DOT:
        return -(train @ query)
    raise ValueError(f"Unsupported metric: {metric}")


def brute_force_ground_truth(train, test, k, metric):
    """Exact top-k indices (into train) for each query. Use this after subsetting."""
    gt = np.empty((len(test), k), dtype=np.int64)
    for i, q in enumerate(test):
        gt[i] = np.argsort(_distances_to_all(train, q, metric))[:k]
    return gt


def recall_at_k(approx_ids_per_query, gt, k):
    """Mean fraction of the true top-k that was retrieved, averaged over queries."""
    total = 0.0
    for approx, truth in zip(approx_ids_per_query, gt):
        total += len(set(approx) & set(truth[:k].tolist())) / k
    return total / len(gt)


def build_hnsw(train, M, ef_construction, metric, seed=0):
    """
    Insert all train rows in shuffled order (HNSW is order-sensitive). Returns
    (index, perm, build_seconds), where perm maps HNSW vid -> original train index:
    the j-th inserted vector gets vid j, and we insert train[perm[j]], so vid j == perm[j].
    """
    dim = train.shape[1]
    perm = np.random.default_rng(seed).permutation(len(train))
    index = HNSW(dim=dim, M=M, ef_construction=ef_construction, metric=metric)
    t0 = time.perf_counter()
    for j in perm:
        index.insert(train[j])
    build_seconds = time.perf_counter() - t0
    return index, perm, build_seconds


def sweep_hnsw(index, perm, test, gt, k, ef_values):
    """For each ef_search: measure recall and average query latency (ms)."""
    results = []
    for ef in ef_values:
        approx_per_query = []
        t0 = time.perf_counter()
        for q in test:
            vids = index.search(q, ef_search=ef, k=k)
            approx_per_query.append([int(perm[v]) for v in vids])  # vid -> original index
        elapsed = time.perf_counter() - t0
        results.append({
            "ef_search": ef,
            "recall": recall_at_k(approx_per_query, gt, k),
            "ms_per_query": 1000 * elapsed / len(test),
        })
    return results


def build_faiss_hnsw(train, M, ef_construction):
    """Baseline: FAISS IndexHNSWFlat (L2). Returns (index, build_seconds)."""
    import faiss
    train = np.ascontiguousarray(train, dtype=np.float32)
    index = faiss.IndexHNSWFlat(train.shape[1], M)  # METRIC_L2 by default
    index.hnsw.efConstruction = ef_construction
    t0 = time.perf_counter()
    index.add(train)  # added in order, so faiss ids == original train indices
    build_seconds = time.perf_counter() - t0
    return index, build_seconds


def sweep_faiss(index, test, gt, k, ef_values):
    """Same sweep for the FAISS baseline. faiss ids are already original train indices."""
    import faiss  # noqa: F401  (ensures a clear error if called without faiss installed)
    test = np.ascontiguousarray(test, dtype=np.float32)
    results = []
    for ef in ef_values:
        index.hnsw.efSearch = ef
        t0 = time.perf_counter()
        _, ids = index.search(test, k)
        elapsed = time.perf_counter() - t0
        approx = [row.tolist() for row in ids]
        results.append({
            "ef_search": ef,
            "recall": recall_at_k(approx, gt, k),
            "ms_per_query": 1000 * elapsed / len(test),
        })
    return results
