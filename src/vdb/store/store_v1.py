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
            self.vectors = np.empty((0, dim), dtype=np.float32)
        else:
            if db.size > 0 and db.shape[1] != dim:
                raise ValueError(f"Database dimension {db.shape[1]} does not match specified dimension {dim}")
             
            self.vectors = db.reshape(-1, dim).astype(np.float32) if db.size > 0 else np.empty((0, dim), dtype=np.float32)

            for i in range(db.shape[0]):
                self.id_to_row[i] = i
                self.row_to_id[i] = i
            self._next_id = db.shape[0]

    def add(self, vector: np.ndarray, metadata=None):
        """
        adds a vector to the vector store using vstack. updates internal mappings
        of row to vid.
        """

        vector = np.asarray(vector, dtype=np.float32)

        if vector.ndim != 1 or vector.shape[0] != self.dim:
            raise ValueError(f"Vector dim {vector.shape} does not match required shape ({self.dim},)")

        vid = self._next_id
        self._next_id += 1
        row = self.vectors.shape[0]

        self.vectors = np.vstack([self.vectors, vector.reshape(1, -1)])

        self.id_to_row[vid] = row
        self.row_to_id[row] = vid
        self.metadata[vid] = metadata or {}
        return vid

    def get(self, vid):
        """
        Retrieves the vector and its corresponding metadata from the given vid.
        """
        if vid in self.id_to_row:
            row = self.id_to_row[vid]
            return self.vectors[row]
        
        raise ValueError(f"Unknown vector id: {vid}")

    def delete(self, vid: int):
        """
        Deletes the vector corresponding to the given vid by swapping the last vector in storage
        with the one being deleted.
        """
        if vid not in self.id_to_row:
            raise ValueError(f"Unknown vector id: {vid}")
        
        row = self.id_to_row[vid]
        last_row = self.vectors.shape[0] - 1
        last_vid = self.row_to_id[last_row]

        if row != last_row:
            self.vectors[row] = self.vectors[last_row]
            self.row_to_id[row] = last_vid
            self.id_to_row[last_vid] = row

        del self.row_to_id[last_row]
        del self.id_to_row[vid]
        self.vectors = self.vectors[:last_row]

    def get_vectors(self):
        """
        Returns the contiguous array of vectors.
        """
        return self.vectors

    def get_metadata(self, vid):
        """
        Retrieves the metadata associated with the given vid.
        """
        if vid in self.metadata:
            return self.metadata[vid]
        
        raise ValueError(f"Unknown vector id: {vid}")

    def update_metadata(self, vid, metadata):
        """
        Updates the metadata for the given vid.
        """
        if vid in self.metadata:
            self.metadata[vid] = metadata
        else:
            raise ValueError(f"Unknown vector id: {vid}")

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
        self.vectors[row] = vector

class Flat:

    @staticmethod
    def search(query: np.ndarray, database: VectorStore, k: int = 5, metric: MetricType = MetricType.L2):
        """
        Returns the scores and ids of the k closest vectors.
        """
        metric_fn = Metrics.get(metric)
        scores = metric_fn(database.vectors, query)
        results = [(score, database.row_to_id[i]) for i, score in enumerate(scores)]

        reverse_sort = (metric != MetricType.L2)
        results.sort(key=lambda x: x[0], reverse=reverse_sort)

        return results[:k]

if __name__ == "__main__":
    # Example usage
    db = VectorStore(dim=3)
    db.add(np.array([1, 2, 3]))
    db.add(np.array([4, 5, 6]))
    db.add(np.array([7, 8, 9]))

    query_vector = np.array([1, 0, 0])
    results = Flat.search(query_vector, db, k=2, metric=MetricType.COSINE)
    print("Search Results (Cosine Similarity):", results)

    results = Flat.search(query_vector, db, k=2, metric=MetricType.L2)
    print("Search Results (L2 Distance):", results)

    results = Flat.search(query_vector, db, k=2, metric=MetricType.DOT)
    print("Search Results (Dot Product):", results)
