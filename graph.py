import numpy as np
from binary_heap import BinaryHeap

from mylogger import logger
from func import *

EPSILON = 1e-8


class Graph:
    def __init__(self, n: int, m: int):
        self.n = n  # vertexes number
        self.m = m  # edges number

        # edge
        self._count = 1  # count of edge
        self._h = np.ones(n, dtype=np.int32) * -1  # index of list header of edges
        self._next = np.zeros(m, dtype=np.int32)  # next edge
        self._to = np.zeros(m, dtype=np.int32)  # index of edge to
        self._w = np.zeros(m, dtype=np.float32)  # weight of edge

        # distance
        self.distance = np.zeros((n, n), dtype=np.float32)  # distance matrix
        self.visited = np.zeros(self.n, dtype=np.bool)  # cache for dijkstra

        # dinic
        self.dinic_vis = np.zeros(n, dtype=np.int32)  # dinic visit
        self._s = 0  # source
        self._t = 0  # target

    def clear(self):
        self._count = 1
        for i in range(self.n):
            self._h[i] = -1

    def add_edge(self, u: int, v: int, weight: float, bidirectional: bool = True):
        # point A, B, C, D, ...
        # edge AB, AC, AD,...
        # before:
        # B -> C -> D
        # after:
        # E -> B -> C -> D
        self._count += 1
        self._next[self._count] = self._h[u]
        self._to[self._count] = v
        self._w[self._count] = weight
        self._h[u] = self._count

        if bidirectional:
            self.add_edge(u=v, v=u, weight=weight, bidirectional=False)

    def calculate_distance(self, src: int, d: np.ndarray):
        """
        Calculate the distance from st to all other points
        :param src: start point
        :param d: distances from point st to all other points
        """
        heap = BinaryHeap()
        visited = self.visited
        for i in range(self.n):
            d[i] = np.inf
            visited[i] = False
        heap.insert(src)
        d[src] = 0.
        visited[src] = True
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
        @time_it
        def calculate_all_distance():
            for i in range(self.n):
                self.calculate_distance(i, self.distance[i])

        @time_it
        def test_distance_symmetric():
            for i in range(self.n):
                for j in range(self.n):
                    logger.assert_true(
                        abs(self.distance[i][j] - self.distance[j][i]) < EPSILON,
                        f'd[{i}][{j}]={self.distance[i][j]}, d[{j}][{i}]={self.distance[j][i]}: not symmetric')

        calculate_all_distance()
        test_distance_symmetric()

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
