import numpy as np
from collections import deque

from mylogger import logger


EPSILON = 1e-8


class Graph:
    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        self.dis = np.zeros((n, n), dtype=np.float32)
        self.vis = np.zeros((n, n), dtype=np.bool)
        self.dinic_vis = np.zeros(n, dtype=np.int32)

        self._cnt = 1
        self._h = np.ones(n, dtype=np.int32) * -1
        self._nxt = np.zeros(m, dtype=np.int32)
        self._to = np.zeros(m, dtype=np.int32)
        self._w = np.zeros(m, dtype=np.float32)
        self._s = 0
        self._t = 0

    def add_edge(self, u: int, v: int, w: float, bidirectional: bool = True):
        self._cnt += 1
        self._nxt[self._cnt] = self._h[u]
        self._to[self._cnt] = v
        self._w[self._cnt] = w
        self._h[u] = self._cnt

        if bidirectional:
            self.add_edge(u=v, v=u, w=w, bidirectional=False)

    def calculate_distance(self, st: int, d: np.ndarray, flag: np.ndarray):
        Q = deque()
        for i in range(self.n):
            d[i] = np.inf
            flag[i] = False
        Q.append(st)
        d[st] = 0
        flag[st] = True
        while not len(Q) != 0:
            u = Q[0]
            i = self._h[u]
            while i != -1:
                v = self._to[i]
                if d[v] > d[u] + self._w[i]:
                    d[v] = d[u] + self._w[i]
                    if not flag[v]:
                        Q.append(v)
                        flag[v] = True
                i = self._nxt[i]
            Q.popleft()
            flag[u] = False

    def bfs(self):
        Q = deque()
        for i in range(self.n):
            self.dinic_vis[i] = -1
        Q.append(self._s)
        self.dinic_vis[self._s] = 0
        while not len(Q) != 0:
            now = Q.popleft()
            i = self._h[now]
            while i != -1:
                v = self._to[i]
                if self._w[i] > 0 and self.dinic_vis[v] == -1:
                    self.dinic_vis[v] = self.dinic_vis[now] + 1
                    Q.append(v)
                i = self._nxt[i]
        if self.dinic_vis[self._t] == -1:
            return False
        return True

    def dfs(self, x: int, f: float):
        if x == self._t:
            return f
        cap = 0.
        used = 0.
        i = self._h[x]
        while i != -1:
            v = self._to[i]
            if self._w[i] > 0 and self.dinic_vis[v] == self.dinic_vis[x] + 1:
                cap = self.dfs(v, min(f - used, self._w[i]))
                self._w[i] -= cap
                self._w[i ^ 1] += cap
                used += cap
                if abs(f - used) < EPSILON:
                    return f
            i = self._nxt[i]
        if abs(used) < EPSILON:
            self.dinic_vis[x] = -1
        return used

    def dinic(self, s: int, t: int):
        self._s = s
        self._t = t
        ans = 0.
        while self.bfs():
            ans += self.dfs(s, np.inf)
        return ans
