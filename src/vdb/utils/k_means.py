import numpy as np

def k_means(X, k, max_iters=100, tol=1e-4):
    """
    Performs k-means clustering on the given data.

    Parameters:
    - X: np.ndarray, shape (n_samples, n_features)
        The input data to cluster.
    - k: int
        The number of clusters.
    - max_iters: int
        Maximum number of iterations for convergence.
    - tol: float
        Tolerance for convergence. If the change in centroids is less than this value, the algorithm stops.
    """
    n_samples, _ = X.shape

    random_indices = np.random.choice(n_samples, size=k, replace=False)
    centroids = X[random_indices]

    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids

    return centroids, labels