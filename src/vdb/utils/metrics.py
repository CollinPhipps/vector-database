from enum import Enum
import numpy as np

class MetricType(Enum):
    COSINE = "cosine"
    L2 = "l2"
    DOT = "dot"

class Metrics:

    @staticmethod
    def cos_sim(database: np.ndarray, query: np.ndarray):
        db_norm = np.clip(np.linalg.norm(database, axis=1), 1e-10, None)
        query_norm = np.clip(np.linalg.norm(query), 1e-10, None)
        return np.dot(database, query) / (db_norm.ravel() * query_norm)

    @staticmethod
    def L2_dist(database: np.ndarray, query: np.ndarray):
        return np.sum((database - query) ** 2, axis=1)

    @staticmethod
    def dot(database: np.ndarray, query: np.ndarray):
        return np.dot(database, query)

    @classmethod
    def get(cls, metric: MetricType):
        mapping = {
            MetricType.COSINE: cls.cos_sim,
            MetricType.L2: cls.L2_dist,
            MetricType.DOT: cls.dot
        }
        if metric in mapping:
            return mapping[metric]
        raise ValueError(f"Unsupported metric type: {metric}")