import numpy as np
import pytest

from vdb.store import VectorStore
from vdb.utils import MetricType


def test_add_and_get():
    store = VectorStore(dim=3)
    vid = store.add(np.array([1, 2, 3]))
    assert np.allclose(store.get(vid), [1, 2, 3])


def test_add_bulk_assigns_sequential_vids_and_metadata():
    store = VectorStore(dim=3)
    vectors = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    vids = store.add_bulk(vectors, metadata=[{"a": 1}, {"a": 2}, {"a": 3}])

    assert vids == [0, 1, 2]
    for vid, vec in zip(vids, vectors):
        assert np.allclose(store.get(vid), vec)
        assert store.get_metadata(vid) == {"a": vid + 1}


def test_delete_swaps_with_last_and_preserves_other_vectors():
    store = VectorStore(dim=3)
    vids = [store.add(np.array([i, i, i], dtype=np.float32)) for i in range(5)]

    store.delete(vids[1])  # delete a middle element, not the last

    assert not store.valid_vid(vids[1])
    with pytest.raises(ValueError):
        store.get(vids[1])

    for vid in vids:
        if vid == vids[1]:
            continue
        assert np.allclose(store.get(vid), [vid, vid, vid])


def test_delete_removes_metadata():
    store = VectorStore(dim=3)
    vid = store.add(np.array([1, 2, 3]), metadata={"a": 1})
    store.delete(vid)
    with pytest.raises(ValueError):
        store.get_metadata(vid)


def test_delete_unknown_vid_raises():
    store = VectorStore(dim=3)
    with pytest.raises(ValueError):
        store.delete(999)


def test_update_metadata_dict_merge_and_single_key():
    store = VectorStore(dim=3)
    vid = store.add(np.array([1, 2, 3]), metadata={"a": 1})

    store.update_metadata(vid, "b", 2)
    assert store.get_metadata(vid) == {"a": 1, "b": 2}

    store.update_metadata(vid, {"c": 3, "a": 10})
    assert store.get_metadata(vid) == {"a": 10, "b": 2, "c": 3}


def test_update_vector():
    store = VectorStore(dim=3)
    vid = store.add(np.array([1, 2, 3]))
    store.update_vector(vid, np.array([9, 9, 9]))
    assert np.allclose(store.get(vid), [9, 9, 9])


@pytest.mark.parametrize("metric", [MetricType.L2, MetricType.COSINE, MetricType.DOT])
def test_flat_search_finds_exact_match_first(metric):
    store = VectorStore(dim=3)
    for v in [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]:
        store.add(np.array(v, dtype=np.float32))

    results = store.flat_search(np.array([1, 0, 0], dtype=np.float32), k=2, metric=metric)
    assert len(results) == 2
    assert results[0][1] == 0  # vid 0 is an exact match, regardless of metric
