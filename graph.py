import numpy as np
from binary_heap import BinaryHeap

from mylogger import logger


EPSILON = 1e-8


class Graph:
    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        self.distance = np.zeros((n, n), dtype=np.float32)
        self.dinic_vis = np.zeros(n, dtype=np.int32)

        self._count = 1
        self._h = np.ones(n, dtype=np.int32) * -1
        self._next = np.zeros(m, dtype=np.int32)
        self._to = np.zeros(m, dtype=np.int32)
        self._w = np.zeros(m, dtype=np.float32)
        self._s = 0
        self._t = 0

        self.visited = np.zeros(self.n, dtype=np.bool)  # cache for dijkstra

    def add_edge(self, u: int, v: int, w: float, bidirectional: bool = True):
        self._count += 1
        self._next[self._count] = self._h[u]
        self._to[self._count] = v
        self._w[self._count] = w
        self._h[u] = self._count

        if bidirectional:
            self.add_edge(u=v, v=u, w=w, bidirectional=False)

    def calculate_distance(self, st: int, d: np.ndarray):
        """
        Calculate the distance from st to all other points
        :param st: start point
        :param d: distances from point st to all other points
        """
        heap = BinaryHeap()
        visited = self.visited
        for i in range(self.n):
            d[i] = np.inf
            visited[i] = False
        heap.insert(st)
        d[st] = 0
        visited[st] = True
        while not heap.empty():
            u = heap.top()
            i = self._h[u]
            while i != -1:
                v = self._to[i]
                if d[v] > d[u] + self._w[i]:
                    d[v] = d[u] + self._w[i]
                    if not visited[v]:
                        heap.insert(v)
                        visited[v] = True
                i = self._next[i]
            heap.pop()
            visited[u] = False

    def calculate_all_distance(self):
        for i in range(self.n):
            self.calculate_distance(i, self.distance[i])

    def dinic_bfs(self):
        heap = BinaryHeap()
        for i in range(self.n):
            self.dinic_vis[i] = -1
        heap.insert(self._s)
        self.dinic_vis[self._s] = 0
        while not heap.empty():
            now = heap.pop()
            i = self._h[now]
            while i != -1:
                v = self._to[i]
                if self._w[i] > 0 and self.dinic_vis[v] == -1:
                    self.dinic_vis[v] = self.dinic_vis[now] + 1
                    heap.insert(v)
                i = self._next[i]
        if self.dinic_vis[self._t] == -1:
            return False
        return True

    def dinic_dfs(self, x: int, f: float):
        if x == self._t:
            return f
        used = 0.
        i = self._h[x]
        while i != -1:
            v = self._to[i]
            if self._w[i] > 0 and self.dinic_vis[v] == self.dinic_vis[x] + 1:
                cap = self.dinic_dfs(v, min(f - used, self._w[i]))
                self._w[i] -= cap
                self._w[i ^ 1] += cap
                used += cap
                if abs(f - used) < EPSILON:
                    return f
            i = self._next[i]
        if abs(used) < EPSILON:
            self.dinic_vis[x] = -1
        return used

    def dinic(self, s: int, t: int):
        self._s = s
        self._t = t
        ans = 0.
        while self.dinic_bfs():
            ans += self.dinic_dfs(s, np.inf)
        return ans
