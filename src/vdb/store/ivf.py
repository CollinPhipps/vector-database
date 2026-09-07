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