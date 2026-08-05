import numpy as np
from .store import VectorStore
from ..utils.k_means import k_means
from ..utils.metrics import MetricType, Metrics

class IVF:
    def __init__(self, dim: int, k: int, nprobe: int, db: np.ndarray):
        if nprobe > k:
            raise ValueError(f"nprobe must be less than k")

        self.vector_store = VectorStore(dim, db)
        centroids, labels = k_means(db, k)
        self.centroids = VectorStore(dim, centroids)

        vid_by_row = np.array([self.vector_store.row_to_id[i] for i in range(db.shape[0])])
        self.vid_to_centroid = dict(zip(vid_by_row, labels.tolist()))

        sorted_indices = np.argsort(labels)
        diffs = np.diff(labels[sorted_indices])
        break_indices = np.where(diffs != 0)[0] + 1
        break_indices = np.concatenate(([0], break_indices, [len(labels)]))
        vid_by_row_sorted = vid_by_row[sorted_indices]

        for i in range(k):
            self.centroids.update_metadata(i, {"members" : vid_by_row_sorted[break_indices[i]:break_indices[i+1]].tolist()})

        self.nprobe = nprobe
        self.dim = dim

    def add(self, vector: np.ndarray, metadata=None):
        if vector.ndim != 1 or vector.shape[0] != self.dim:
            raise ValueError(f"Vector dim {vector.shape} does not match required shape ({self.dim},)")

        new_vid = self.vector_store.add(vector, metadata)

        centroid_search = self.centroids.flat_search(vector, k=1, metric=MetricType.L2)
        self.centroids.get_metadata(centroid_search[0][1])["members"].append(new_vid)
        self.vid_to_centroid[new_vid] = centroid_search[0][1]
        return new_vid

    def delete(self, vid: int):
        if not self.vector_store.valid_vid(vid):
            raise ValueError(f"{vid} not a valid vector id")

        centroid_vid = self.vid_to_centroid.pop(vid)
        self.centroids.get_metadata(centroid_vid)["members"].remove(vid)
        self.vector_store.delete(vid)

    def search(self, query: np.ndarray, k: int, metric: MetricType):
        if query.ndim != 1 or query.shape[0] != self.dim:
            raise ValueError(f"Vector dim {query.shape} does not match required shape ({self.dim},)")

        centroid_search = self.centroids.flat_search(query, self.nprobe, metric)
        vids = np.concatenate([self.centroids.get_metadata(vid)["members"] for _, vid in centroid_search])
        rows = [self.vector_store.id_to_row[vid] for vid in vids]

        candidate_matrix = self.vector_store.get_vectors()[rows]
        metric_fn = Metrics.get(metric)
        scores = metric_fn(candidate_matrix, query)
        results = [(score, vid) for vid, score in zip(vids, scores)]
        reverse_sort = (metric != MetricType.L2)
        results.sort(key=lambda x: x[0], reverse=reverse_sort)
        return results[:k]

if __name__ == "__main__":
    np.random.seed(42)
    n, dim, k, nprobe = 500, 16, 20, 20  # nprobe == k -> exhaustive, directly comparable to brute force

    db = np.random.randn(n, dim).astype(np.float32)
    ivf = IVF(dim=dim, k=k, nprobe=nprobe, db=db)

    # --- construction invariants: every original vid assigned to exactly one cluster ---
    all_members = [vid for c in range(k) for vid in ivf.centroids.get_metadata(c)["members"]]
    assert len(all_members) == n, f"expected {n} members total, got {len(all_members)}"
    assert set(all_members) == set(range(n)), "members should cover every original vid exactly once"
    assert len(ivf.vid_to_centroid) == n
    for vid in range(n):
        c = ivf.vid_to_centroid[vid]
        assert vid in ivf.centroids.get_metadata(c)["members"], f"vid {vid} missing from its own centroid's members"
    print("construction invariants OK")

    # --- search correctness: with nprobe == k, IVF should exactly match brute force ---
    baseline = VectorStore(dim, db)
    query = np.random.randn(dim).astype(np.float32)
    for metric in (MetricType.L2, MetricType.COSINE, MetricType.DOT):
        brute_vids = [vid for _, vid in baseline.flat_search(query, k=5, metric=metric)]
        ivf_vids = [vid for _, vid in ivf.search(query, k=5, metric=metric)]
        assert brute_vids == ivf_vids, f"{metric}: brute force {brute_vids} != ivf {ivf_vids}"
    print("search matches brute force when nprobe == k OK")

    # --- add: new vector is queryable, membership bookkeeping updated ---
    new_vec = np.random.randn(dim).astype(np.float32)
    new_vid = ivf.add(new_vec)
    assert new_vid == n
    assert new_vid in ivf.vid_to_centroid
    owning_centroid = ivf.vid_to_centroid[new_vid]
    assert new_vid in ivf.centroids.get_metadata(owning_centroid)["members"]

    results = ivf.search(new_vec, k=1, metric=MetricType.L2)
    assert results[0][1] == new_vid, "searching for the exact added vector should return itself as nearest"
    print("add OK")

    # --- delete: vector and bookkeeping removed, no longer searchable ---
    victim = 0
    victim_vector = db[victim].copy()
    victim_centroid = ivf.vid_to_centroid[victim]
    ivf.delete(victim)

    assert victim not in ivf.vid_to_centroid
    assert victim not in ivf.centroids.get_metadata(victim_centroid)["members"]
    assert not ivf.vector_store.valid_vid(victim)

    results = ivf.search(victim_vector, k=5, metric=MetricType.L2)
    assert victim not in [vid for _, vid in results], "deleted vid should not be returned by search"
    print("delete OK")

    print("all tests passed")