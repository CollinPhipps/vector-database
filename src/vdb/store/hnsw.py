import heapq
import numpy as np
from .store import VectorStore
from ..utils.metrics import MetricType

class Node:
    def __init__(self, vid: int, top_level: int):
        """
        level: the top layer the node reaches.
        """
        self.vid = vid
        self.top_level = top_level
        self.neighbors = [[] for _ in range(top_level + 1)] # neigbors[l] is the list of neighbor vids at layer l

    def add_neighbor(self, vid: int, level: int):
        if level > (self.top_level):
            raise ValueError(f"{level} must be less than top level: {self.top_level}")

        if vid in self.neighbors[level]:
            return
        self.neighbors[level].append(vid)

class HNSW:
    def __init__(self, dim, M=16, ef_construction=200, ef_search=50, metric=MetricType.L2):
        self.dim = dim
        self.M = M                     # target neighbors per node per layer
        self.M_max = M                 # degree cap on layers > 0
        self.M_max0 = 2 * M            # degree cap on layer 0 (kept denser)
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.metric = metric
        self.mL = 1.0 / np.log(M)      # level-generation normalization factor

        self.store = VectorStore(dim)  # vectors live here; the graph only holds vids
        self.graph = {}                # graph[vid] -> Node
        self.entry_point = None        # vid of the current top-of-graph node
        self.max_level = -1

    def _dist(self, query, vid):
        """
        Closeness where SMALLER = CLOSER, made uniform across metrics so the heaps
        and the greedy walk never have to special-case "similarity vs distance".
        """
        v = self.store.get(vid)
        if self.metric == MetricType.L2:
            return float(np.sum((v - query) ** 2))
        if self.metric == MetricType.COSINE:
            denom = np.linalg.norm(v) * np.linalg.norm(query)
            sim = float(np.dot(v, query) / denom) if denom > 0 else 0.0
            return 1.0 - sim
        if self.metric == MetricType.DOT:
            return -float(np.dot(v, query))
        raise ValueError(f"Unsupported metric: {self.metric}")

    def _reduce_candidate_list(self, sorted_candidates, M: int):
        results = [sorted_candidates[0]]
        for dist_can, neighbor_vid in sorted_candidates[1:]:
            if len(results) >= M:
                break
            add = True
            candidate_vector = self.store.get(neighbor_vid)
            for _, res_neigh_vid in results:
                # diversity rule: reject candidate e if an already-selected r is
                # closer to e than e is to the base -> measure dist(e, r), not dist(base, r)
                if self._dist(candidate_vector, res_neigh_vid) <= dist_can:
                    add = False
            if add:
                results.append((dist_can, neighbor_vid))

        return results

    def insert(self, vector: np.ndarray):
        # later add extendCandiates and keepPrunedConnections flags
        if vector.ndim != 1 or vector.shape[0] != self.dim:
            raise ValueError(f"Vector dim {vector.shape} does not match required shape ({self.dim},)")

        vid = self.store.add(vector)
        safe_zero = np.nextafter(0.0, 1.0)
        level = int(np.floor(-np.log(np.random.uniform(safe_zero, 1.0)) * self.mL))
        self.graph[vid] = Node(vid, level)

        if self.entry_point is None:
            self.entry_point = vid
            self.max_level = level
            return

        # descent
        entry = self.entry_point
        for layer in range(self.max_level, level, -1):
            W = self.search_layer(vector, [entry], ef=1, layer=layer)
            entry = min(W, key=lambda x: x[0])[1]

        entry_points = [entry]
        for layer in range(min(self.max_level, level), -1, -1):
            W = sorted(self.search_layer(vector, entry_points, self.ef_construction, layer=layer), key=lambda x: x[0])
            results = self._reduce_candidate_list(W, self.M)

            cap = self.M_max0 if layer == 0 else self.M_max
            for _, neighbor_vid in results:
                self.graph[vid].add_neighbor(neighbor_vid, layer)
                self.graph[neighbor_vid].add_neighbor(vid, layer)

                if len(self.graph[neighbor_vid].neighbors[layer]) > cap:
                    e_vector = self.store.get(neighbor_vid)
                    prune_candidates = [(self._dist(e_vector, n), n) for n in self.graph[neighbor_vid].neighbors[layer]]
                    prune_candidates.sort(key=lambda x: x[0])
                    pruned = self._reduce_candidate_list(prune_candidates, cap)
                    self.graph[neighbor_vid].neighbors[layer] = [n for _, n in pruned]

            entry_points = [vid for _, vid in W]

        if level > self.max_level:
            self.entry_point = vid
            self.max_level = level
                
    def search_layer(self, query, entry_points, ef, layer):
        """
        Best-first search over a single layer.

        - query:        the target vector we're routing toward
        - entry_points: iterable of vids to seed the search from
        - ef:           width of the result set to maintain (ef=1 -> greedy walk)
        - layer:        which layer's adjacency to traverse

        Returns a list of (closeness, vid) for the ef closest nodes found.
        """
        visited = set()
        candidates = []   # min-heap of (dist, vid): nearest-to-expand pops first
        found = []        # min-heap of (-dist, vid): the furthest keeper sits on top

        for ep in entry_points:
            dist = self._dist(query, ep)
            visited.add(ep)
            heapq.heappush(candidates, (dist, ep))
            heapq.heappush(found, (-dist, ep))

        while candidates:
            dist_vid, vid = heapq.heappop(candidates)
            furthest = -found[0][0]
            if dist_vid > furthest:
                break  # nearest remaining candidate is farther than our worst keeper -> done

            for neighbor_vid in self.graph[vid].neighbors[layer]:
                if neighbor_vid in visited:
                    continue
                visited.add(neighbor_vid)
                dist_to_node = self._dist(query, neighbor_vid)
                furthest = -found[0][0]
                if dist_to_node < furthest or len(found) < ef:
                    heapq.heappush(candidates, (dist_to_node, neighbor_vid))
                    heapq.heappush(found, (-dist_to_node, neighbor_vid))
                    if len(found) > ef:
                        heapq.heappop(found) # evict the furthest keeper

        return [(-nd, vid) for nd, vid in found]

    def search(self, query: np.ndarray, k: int):
        entry = self.entry_point
        for layer in range(self.max_level, 0, -1):
            W = self.search_layer(query, [entry], ef=1, layer=layer)
            entry = min(W, key=lambda x: x[0])[1]
        W = sorted(self.search_layer(query, [entry], ef=self.ef_search, layer=0))
        return [vid for _, vid in W[:k]]


if __name__ == "__main__":
    # Hand-wired toy graph to exercise search_layer in isolation (no insertion yet).
    # Six points on a line, chained as 0-1-2-3-4-5 at layer 0.
    h = HNSW(dim=2, metric=MetricType.L2)
    for x in range(6):
        h.store.add(np.array([x, 0.0], dtype=np.float32))
        h.graph[x] = Node(vid=x, top_level=0)

    edges = [(0,1),(1,2),(2,3),(3,4),(4,5)]
    for a, b in edges:
        h.graph[a].add_neighbor(b, 0)
        h.graph[b].add_neighbor(a, 0)

    query = np.array([4.2, 0.0], dtype=np.float32)

    # ef=1 is the greedy hill-climb: starting from vid 0, it should walk the chain
    # and settle on vid 4 (the nearest), never getting stuck earlier.
    res1 = h.search_layer(query, entry_points=[0], ef=1, layer=0)
    print("ef=1 from vid0:", res1)
    assert res1[0][1] == 4, f"expected nearest vid 4, got {res1}"

    # ef=3 should return the three closest overall: vids 3, 4, 5.
    res3 = h.search_layer(query, entry_points=[0], ef=3, layer=0)
    got = sorted(vid for _, vid in res3)
    print("ef=3 from vid0:", res3)
    assert got == [3, 4, 5], f"expected [3, 4, 5], got {got}"

    print("search_layer toy tests passed")

    # --- multi-layer HNSW built via insert(), compared to brute force ---
    np.random.seed(0)
    n, dim, k = 1000, 16, 10
    h = HNSW(dim=dim, M=8, ef_construction=100, ef_search=50, metric=MetricType.L2)

    data = np.random.randn(n, dim).astype(np.float32)
    for v in data:
        h.insert(v)

    # structural sanity: every vector is a node, and nobody exceeds the layer-0 cap
    assert len(h.graph) == n, f"expected {n} nodes, got {len(h.graph)}"
    for vid, node in h.graph.items():
        deg = len(node.neighbors[0])
        assert deg <= h.M_max0, f"node {vid} has degree {deg} > cap {h.M_max0}"
    # a working NSW shouldn't leave nodes stranded with zero links
    isolated = [vid for vid, node in h.graph.items() if len(node.neighbors[0]) == 0]
    assert not isolated, f"{len(isolated)} isolated nodes (graph not connected)"
    print("structural invariants OK")

    # the real query path: descend upper layers with ef=1, then search layer 0 with ef_search
    def hierarchical_search(q, k):
        entry = h.entry_point
        for layer in range(h.max_level, 0, -1):
            W = h.search_layer(q, [entry], ef=1, layer=layer)
            entry = min(W, key=lambda x: x[0])[1]
        W = sorted(h.search_layer(q, [entry], h.ef_search, 0))
        return [vid for _, vid in W[:k]]

    # count _dist calls to compare hierarchical descent vs a flat layer-0 search
    call_count = {"n": 0}
    base_dist = h._dist
    def counting_dist(query, vid):
        call_count["n"] += 1
        return base_dist(query, vid)
    h._dist = counting_dist

    queries = np.random.randn(50, dim).astype(np.float32)
    total, hier_calls, flat_calls = 0.0, 0, 0
    for q in queries:
        truth = {vid for _, vid in h.store.flat_search(q, k, MetricType.L2)}

        call_count["n"] = 0
        approx = set(hierarchical_search(q, k))
        hier_calls += call_count["n"]

        call_count["n"] = 0
        _ = sorted(h.search_layer(q, [h.entry_point], h.ef_search, 0))[:k]
        flat_calls += call_count["n"]

        total += len(truth & approx) / k

    recall = total / len(queries)
    print(f"max_level reached: {h.max_level}")
    print(f"hierarchical recall@{k}: {recall:.3f}")
    print(f"avg _dist calls/query   hierarchical: {hier_calls / len(queries):.1f}"
          f"   flat layer-0: {flat_calls / len(queries):.1f}")
    assert recall > 0.85, f"recall too low: {recall:.3f}"
    print("hierarchical HNSW recall test passed")
