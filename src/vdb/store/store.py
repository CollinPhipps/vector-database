import numpy as np
from ..utils.metrics import MetricType, Metrics

class VectorStore:
    def __init__(self, dim: int, db: np.ndarray = None):
        self.dim = dim
        self.id_to_row = {}
        self.row_to_id = {}
        self.metadata = {}
        self._next_id = 0

        if db is None: 
            self._vectors = np.empty((0, dim), dtype=np.float32)
        else:
            if db.size > 0 and db.shape[1] != dim:
                raise ValueError(f"Database dimension {db.shape[1]} does not match specified dimension {dim}")
             
            self._vectors = db.reshape(-1, dim).astype(np.float32) if db.size > 0 else np.empty((0, dim), dtype=np.float32)

            for i in range(db.shape[0]):
                self.id_to_row[i] = i
                self.row_to_id[i] = i
                self.metadata[i] = {}
            self._next_id = db.shape[0]

    def add(self, vector: np.ndarray, metadata: dict[any, any]=None):
        """
        adds a vector to the vector store using vstack. updates internal mappings
        of row to vid.

        Parameters:
        - vector: np.ndarry
            The vector to add to the database
        - metadata
        """
        vector = np.asarray(vector, dtype=np.float32)

        if vector.ndim != 1 or vector.shape[0] != self.dim:
            raise ValueError(f"Vector dim {vector.shape} does not match required shape ({self.dim},)")

        vid = self._next_id
        self._next_id += 1
        row = self._vectors.shape[0]

        self._vectors = np.vstack([self._vectors, vector.reshape(1, -1)])

        self.id_to_row[vid] = row
        self.row_to_id[row] = vid
        self.metadata[vid] = metadata or {}
        return vid

    def get(self, vid):
        """
        Retrieves the vector from the given vid.
        """
        if vid in self.id_to_row:
            row = self.id_to_row[vid]
            return self._vectors[row]
        
        raise ValueError(f"Unknown vector id: {vid}")

    def delete(self, vid: int):
        """
        Deletes the vector corresponding to the given vid by swapping the last vector in storage
        with the one being deleted.
        """
        if vid not in self.id_to_row:
            raise ValueError(f"Unknown vector id: {vid}")
        
        row = self.id_to_row[vid]
        last_row = self._vectors.shape[0] - 1
        last_vid = self.row_to_id[last_row]

        if row != last_row:
            self._vectors[row] = self._vectors[last_row]
            self.row_to_id[row] = last_vid
            self.id_to_row[last_vid] = row

        del self.row_to_id[last_row]
        del self.id_to_row[vid]
        del self.metadata[vid]
        self._vectors = self._vectors[:last_row]

    def get_vectors(self):
        """
        Returns the contiguous array of vectors.
        """
        return self._vectors

    def get_metadata(self, vid):
        """
        Retrieves the metadata associated with the given vid.
        """
        if vid in self.metadata:
            return self.metadata[vid]
        
        raise ValueError(f"Unknown vector id: {vid}")

    def update_metadata(self, vid: int, key: any, value: any = None):
        """
        Updates the metadata for the given vid. Pass a dict as `key` to merge
        multiple entries at once, or a single key/value pair to set one entry.
        """
        if vid not in self.id_to_row:
            raise ValueError(f"Unknown vector id: {vid}")

        if isinstance(key, dict):
            self.metadata[vid].update(key)
        else:
            self.metadata[vid][key] = value

    def update_vector(self, vid, vector):
        """
        Updates the vecotor for the given vid.
        """
        if vid not in self.id_to_row:
            raise ValueError(f"Unknown vector id: {vid}")

        vector = np.asarray(vector, dtype=np.float32)

        if vector.ndim != 1 or vector.shape[0] != self.dim:
            raise ValueError(f"Vector dim {vector.shape} does not match required shape ({self.dim},)")

        row = self.id_to_row[vid]
        self._vectors[row] = vector

    def flat_search(self, query: np.ndarray, k: int = 5, metric: MetricType = MetricType.L2):
        """
        Returns the scores and ids of the k closest vectors.
        """
        if query.ndim != 1 or query.shape[0] != self.dim:
            raise ValueError(f"Vector dim {query.shape} does not match required shape ({self.dim},)")

        metric_fn = Metrics.get(metric)
        scores = metric_fn(self._vectors, query)
        results = [(score, self.row_to_id[i]) for i, score in enumerate(scores)]

        reverse_sort = (metric != MetricType.L2)
        results.sort(key=lambda x: x[0], reverse=reverse_sort)

        return results[:k]

    def valid_vid(self, vid: int):
        return vid in self.id_to_row

if __name__ == "__main__":
    # Example usage
    db = VectorStore(dim=3)
    db.add(np.array([1, 2, 3]))
    db.add(np.array([4, 5, 6]))
    db.add(np.array([7, 8, 9]))

    query_vector = np.array([1, 0, 0])
    results = db.flat_search(query_vector, k=2, metric=MetricType.COSINE)
    print("Search Results (Cosine Similarity):", results)

    results = db.flat_search(query_vector, k=2, metric=MetricType.L2)
    print("Search Results (L2 Distance):", results)

    results = db.flat_search(query_vector, k=2, metric=MetricType.DOT)
    print("Search Results (Dot Product):", results)