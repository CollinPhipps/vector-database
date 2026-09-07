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
    def __init__(self, dim, M=16, ef_construction=200, metric=MetricType.L2):
        self.dim = dim
        self.M = M                     # target neighbors per node per layer
        self.M_max = M                 # degree cap on layers > 0
        self.M_max0 = 2 * M            # degree cap on layer 0 (kept denser)
        self.ef_construction = ef_construction
        self.metric = metric
        self.mL = 1.0 / np.log(M)      # level-generation normalization factor

        self.store = VectorStore(dim)
        self.graph = {}                # graph[vid] -> Node
        self.entry_point = None        # vid of the current top-of-graph node
        self.max_level = -1

    def build(self, db: np.ndarray):
        if db.size > 0 and db.shape[1] != self.dim:
            raise ValueError(f"Database dimension {db.shape[1]} does not match specified dimension {self.dim}")
        vectors = db.reshape(-1, self.dim).astype(np.float32)
        vectors = np.random.permutation(vectors)
        for vector in vectors:
            self.insert(vector)

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

    def search(self, query: np.ndarray, ef_search: int, k: int):
        entry = self.entry_point
        for layer in range(self.max_level, 0, -1):
            W = self.search_layer(query, [entry], ef=1, layer=layer)
            entry = min(W, key=lambda x: x[0])[1]
        W = sorted(self.search_layer(query, [entry], ef=ef_search, layer=0))
        return [vid for _, vid in W[:k]]
