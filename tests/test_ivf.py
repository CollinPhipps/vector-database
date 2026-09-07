import numpy as np
import pytest

from vdb.store import VectorStore, IVF
from vdb.utils import MetricType


@pytest.fixture
def built_ivf():
    np.random.seed(42)
    n, dim, k, nprobe = 500, 16, 20, 20  # nprobe == k -> exhaustive, comparable to brute force
    db = np.random.randn(n, dim).astype(np.float32)
    ivf = IVF(dim=dim, k=k, nprobe=nprobe, db=db)
    return ivf, db, n, k


def test_construction_invariants(built_ivf):
    ivf, db, n, k = built_ivf

    all_members = [vid for c in range(k) for vid in ivf.centroids.get_metadata(c)["members"]]
    assert len(all_members) == n, "every original vid should be assigned to exactly one cluster"
    assert set(all_members) == set(range(n))

    assert len(ivf.vid_to_centroid) == n
    for vid in range(n):
        c = ivf.vid_to_centroid[vid]
        assert vid in ivf.centroids.get_metadata(c)["members"]


def test_search_matches_brute_force_when_nprobe_equals_k(built_ivf):
    ivf, db, n, k = built_ivf
    baseline = VectorStore(ivf.dim, db)

    np.random.seed(1)
    query = np.random.randn(ivf.dim).astype(np.float32)

    for metric in (MetricType.L2, MetricType.COSINE, MetricType.DOT):
        brute_vids = [vid for _, vid in baseline.flat_search(query, k=5, metric=metric)]
        ivf_vids = [vid for _, vid in ivf.search(query, k=5, metric=metric)]
        assert brute_vids == ivf_vids


def test_add_new_vector_is_searchable(built_ivf):
    ivf, db, n, k = built_ivf

    np.random.seed(2)
    new_vec = np.random.randn(ivf.dim).astype(np.float32)
    new_vid = ivf.add(new_vec)

    assert new_vid == n
    assert new_vid in ivf.vid_to_centroid
    owning_centroid = ivf.vid_to_centroid[new_vid]
    assert new_vid in ivf.centroids.get_metadata(owning_centroid)["members"]

    results = ivf.search(new_vec, k=1, metric=MetricType.L2)
    assert results[0][1] == new_vid, "searching for the exact added vector should return itself"


def test_delete_removes_vector_and_bookkeeping(built_ivf):
    ivf, db, n, k = built_ivf

    victim = 0
    victim_vector = db[victim].copy()
    victim_centroid = ivf.vid_to_centroid[victim]

    ivf.delete(victim)

    assert victim not in ivf.vid_to_centroid
    assert victim not in ivf.centroids.get_metadata(victim_centroid)["members"]
    assert not ivf.vector_store.valid_vid(victim)

    results = ivf.search(victim_vector, k=5, metric=MetricType.L2)
    assert victim not in [vid for _, vid in results]
