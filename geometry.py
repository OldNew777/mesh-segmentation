import math
import os
import sys
from typing import Tuple, List

from plyfile import PlyData
import numpy as np
import random

from mylogger import logger
from graph import Graph
from func import *

palette = []
default_color = np.array([1.0, 1.0, 1.0])


def init_palette(k: int):
    for i in range(k):
        palette.append(np.array([random.random(),
                                 random.random(),
                                 random.random()]))


class Vertex:
    def __init__(self, pos: np.ndarray, normal: np.ndarray, color=default_color):
        self.pos = pos
        self.normal = normal
        self.color = color

    def __str__(self):
        return f"Vertex pos: {self.pos}, normal: {self.normal}, color: {self.color}"

    def __repr__(self):
        return self.__str__()


class Mesh:
    def __init__(self, indexes: np.ndarray):
        self.indexes = indexes

    def __str__(self):
        return f"Mesh indexes: {self.indexes}"

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, item):
        return self.indexes[item]


class Geometry:
    def __init__(self, vertices: List[Vertex], meshes: List[Mesh]):
        self.v = vertices
        self.f = meshes
        self.edge: List[Tuple[int, int]] = []
        self.edge_w: List[Tuple[float, float]] = []

        self.eta = 0.2
        self.delta = 0.8
        self.epsilon = 1e-8
        self.fuzzy_epsilon = 0.075
        self.k_limit = 20
        self.iter_max = 8

        logger.debug(self.v)
        logger.debug(self.f)

    @classmethod
    def from_ply(cls, ply_path: os.path) -> 'Geometry':
        logger.assert_true(ply_path.endswith('.ply'), 'File must be a .ply file')
        scene = PlyData.read(ply_path)
        logger.info('Loaded mesh from file:', ply_path)
        logger.info(scene)

        vertices = []
        meshes = []
        for v in scene['vertex']:
            pos = np.array([v['x'], v['y'], v['z']])
            normal = np.array([v['nx'], v['ny'], v['nz']])
            vertices.append(Vertex(pos, normal))
        for f in scene['face']:
            meshes.append(Mesh(f['vertex_indices']))

        return cls(vertices, meshes)

    def split_mesh(self, k: int = -1) -> 'Geometry':
        logger.info(f'Splitting mesh into {k} segments')

        n_node = len(self.f) // 3
        m_edge = len(self.f) << 2

        graph = Graph(n_node, m_edge)
        self.build_graph(graph, n_node)
        graph.calculate_all_distance()

        repetition, k_suggest = self.choose_k(graph=graph, k=k)
        if k <= 0:
            k = k_suggest

        self.cluster_k(graph=graph, repetition=repetition, k=k)
        belong = self.final_decomposition(graph=graph, repetition=repetition, k=k)

        geometry_split = Geometry(vertices=self.v.copy(), meshes=self.f.copy())
        for i in range(n_node):
            for j in range(3):
                if belong[i] == -1:
                    geometry_split.v[i * 3 + j].color = default_color
                else:
                    geometry_split.v[i * 3 + j].color = palette[belong[i]]

        del repetition
        del belong
        del graph
        return geometry_split

    def build_graph(self, graph: Graph, n_face: int):
        edge_map = {}
        center = []
        normal = []

        self.edge.clear()
        self.edge_w.clear()

        for i in range(n_face):
            base = i * 3

            center.append(
                (self.v[base + 0].pos +
                 self.v[base + 1].pos +
                 self.v[base + 2].pos) / 3)
            average_normal = normalize(
                (self.v[base + 0].normal +
                 self.v[base + 1].normal +
                 self.v[base + 2].normal))
            face_normal = normalize(
                np.cross(self.v[self.f[i][0]].pos - self.v[self.f[i][1]].pos,
                         self.v[self.f[i][1]].pos - self.v[self.f[i][2]].pos)
            )
            if np.dot(face_normal, average_normal) < 0:
                face_normal = -face_normal
            normal.append(face_normal)

            for j in range(3):
                edge_temp = (self.f[i][j], self.f[i][(j + 1) % 3])
                if edge_temp[0] < edge_temp[1]:
                    edge_temp = (edge_temp[1], edge_temp[0])
                if edge_temp not in edge_map:
                    edge_map[edge_temp] = i
                else:
                    u = edge_map[edge_temp]
                    self.edge.append((u, i))
                    edge_weight = self.calculate_weight(
                        line=edge_temp,
                        normal_u=normal[u],
                        normal_v=normal[i],
                        center_u=center[u],
                        center_v=center[i]
                    )
                    self.edge_w.append(edge_weight)

        geo_dis_average = 0.
        angle_dis_average = 0.
        for dis_tuple in self.edge_w:
            geo_dis_average += dis_tuple[0]
            angle_dis_average += dis_tuple[1]
        edge_num = len(self.edge)
        geo_dis_average /= edge_num
        angle_dis_average /= edge_num
        for i in range(edge_num):
            self.edge_w[i][0] /= geo_dis_average
            self.edge_w[i][1] /= angle_dis_average
            # edge weight = delta * geo_dis + (1 - alpha) * angle_dis
            cur_edge_weight = self.delta * self.edge_w[i][0] + \
                              (1 - self.delta) * self.edge_w[i][1]
            graph.add_edge(u=self.edge[i][0], v=self.edge[i][1], weight=cur_edge_weight)

        del edge_map
        del center
        del normal

    def calculate_weight(
            self,
            line: Tuple[int, int],
            normal_u: np.ndarray,
            normal_v: np.ndarray,
            center_u: np.ndarray,
            center_v: np.ndarray) -> Tuple[float, float]:
        vertex_u = self.v[line[0]].pos
        vertex_v = self.v[line[1]].pos

        dis_u = distance(center_u, vertex_u)
        dis_v = distance(center_v, vertex_u)
        dis_line = distance(vertex_v, vertex_u)

        dis_cos_u = np.dot(center_u - vertex_u, vertex_v - vertex_u) / dis_line
        dis_cos_v = np.dot(center_v - vertex_u, vertex_v - vertex_u) / dis_line

        dis_sin_u = math.sqrt(dis_u * dis_u - dis_cos_u * dis_cos_u)
        dis_sin_v = math.sqrt(dis_v * dis_v - dis_cos_v * dis_cos_v)

        verticle = abs(dis_cos_u) + abs(dis_cos_v)
        if dis_cos_u * dis_cos_v > 0:
            horizon = abs(dis_sin_u - dis_sin_v)
        else:
            horizon = dis_sin_u + dis_sin_v

        geo_dis = math.sqrt(horizon * horizon + verticle * verticle)

        center_uv = center_v - center_u
        cos_angle = np.dot(normal_u, normal_v)
        if np.dot(center_uv, normal_u) < 0:
            angle_dis = self.eta * (1 - cos_angle)
        else:
            angle_dis = 1 - cos_angle

        return geo_dis, angle_dis

    def choose_k(self, graph: Graph, k: int) -> Tuple[np.ndarray, int]:
        lim = k
        if lim <= 0:
            lim = self.k_limit
        repetition = np.zeros(lim, dtype=np.int)
        Gk = []

        choose_min = sys.float_info.max
        for i in range(graph.n):
            dis_sum = 0.
            for j in range(graph.n):
                dis_sum += graph.distance[i][j]
            if dis_sum < choose_min:
                choose_min = dis_sum
                repetition[0] = i
        for cur in range(1, lim):
            choose_max = -sys.float_info.max
            for i in range(graph.n):
                dis_min = sys.float_info.max
                for j in range(cur):
                    dis_min = min(dis_min, graph.distance[i][repetition[j]])
                if dis_min > choose_max:
                    choose_max = dis_min
                    repetition[cur] = i
            Gk.append(choose_max)

        if k == 2:
            dis_max = -sys.float_info.max
            for i in range(graph.n):
                for j in range(i + 1, graph.n):
                    if graph.distance[i][j] > dis_max:
                        repetition[0] = i
                        repetition[1] = j
                        dis_max = graph.distance[i][j]

        k_suggest = -1
        grad_max = -sys.float_info.max
        for i in range(1, lim - 1):
            if Gk[i - 1] - Gk[i] > grad_max:
                grad_max = Gk[i - 1] - Gk[i]
                k_suggest = i + 2

        del Gk
        return repetition, k_suggest

    def cluster_k(self, graph: Graph, repetition: np.ndarray, k: int):
        repetition_backup = np.zeros(k, dtype=np.int)
        belong = np.zeros(graph.n, dtype=np.int)
        converge = False
        p = np.zeros(shape=(k, graph.n), dtype=np.float)
        has: List[List[int]] = [[] for _ in range(k)]
        iter = 0

        while not converge:
            iter += 1
            for i in range(k):
                repetition_backup[i] = repetition[i]
                has[i].clear()
            logger.info(' '.join(repetition))

            for j in range(graph.n):
                belong[j] = -1
                dis_inv_sum = 0.
                for i in range(k):
                    if graph.distance[repetition[i]][j] < self.epsilon:
                        belong[j] = i
                        break
                    dis_inv_sum += 1. / graph.distance[repetition[i]][j]
                if belong[j] != -1:
                    for i in range(k):
                        p[i][j] = 0.
                    p[belong[j]][j] = 1.
                else:
                    p_max = -sys.float_info.max
                    for i in range(k):
                        p[i][j] = (1. / graph.distance[repetition[i]][j]) / dis_inv_sum
                        if p[i][j] > p_max:
                            p_max = p[i][j]
                            belong[j] = i
                has[belong[j]].append(j)

            # recalculate p
            for j in range(graph.n):
                belong[j] = -1
                dis_inv_sum = 0.
                for i in range(k):
                    if graph.distance[repetition[i]][j] < self.epsilon:
                        belong[j] = i
                        break
                    p[i][j] = 0.
                    node_belong = len(has[i])
                    for node in has[i]:
                        p[i][j] += graph.distance[node][j]
                    p[i][j] = node_belong / p[i][j]
                    dis_inv_sum += p[i][j]
                if belong[j] != -1:
                    for i in range(k):
                        p[i][j] = 0.
                    p[belong[j]][j] = 1.
                else:
                    for i in range(k):
                        p[i][j] /= dis_inv_sum

            for i in range(k):
                dis_min = sys.float_info.max
                for cur in range(graph.n):
                    dis_sum = 0.
                    for j in range(graph.n):
                        dis_sum += p[i][j] * graph.distance[cur][j]
                    if dis_sum < dis_min:
                        dis_min = dis_sum
                        repetition[i] = cur

            check_equal = False
            for i in range(k):
                for j in range(i):
                    if repetition[i] == repetition[j]:
                        check_equal = True
                        break
                if check_equal:
                    break
            if check_equal:
                for i in range(k):
                    repetition[i] = repetition_backup[i]
            converge = True
            for i in range(k):
                if repetition[i] != repetition_backup[i]:
                    converge = False
                    break
            if iter == self.iter_max:
                break

        del belong
        del repetition_backup
        del p
        del has

    def final_decomposition(self, graph: Graph, repetition: np.ndarray, k: int, n_node: int) -> np.ndarray:
        belong = np.zeros(n_node, dtype=np.int)
        fuzzy_type = k * (k - 1) >> 1
        fuzzy_v: List[List[int]] = [[] for _ in range(fuzzy_type)]
        for j in range(graph.n):
            dis_min = dis_cmin = sys.float_info.max
            belong_min = belong_cmin = -1
            for i in range(k):
                if graph.distance[repetition[i]][j] < dis_min:
                    dis_cmin = dis_min
                    belong_cmin = belong_min
                    dis_min = graph.distance[repetition[i]][j]
                    belong_min = i
                elif graph.distance[repetition[i]][j] < dis_cmin:
                    dis_cmin = graph.distance[repetition[i]][j]
                    belong_cmin = i
            belong[j] = -1
            if dis_min < self.epsilon:
                belong[j] = belong_min
            else:
                p = dis_cmin / (dis_min + dis_cmin)
                if p > 0.5 + self.fuzzy_epsilon:
                    belong[j] = belong_min
                else:
                    if belong_min < belong_cmin:
                        belong_min, belong_cmin = belong_cmin, belong_min
                    ind = ((belong_min * (belong_min - 1)) >> 1) + belong_cmin
                    fuzzy_v[ind].append(j)

        flow_graph = Graph(n=graph.n + 2, m=graph.m)
        flag = np.zeros(graph.n, dtype=np.bool)
        edge_num = len(self.edge)
        cur_ind = 0
        for s in range(k):
            for t in range(s):
                if len(fuzzy_v[cur_ind]) == 0:
                    cur_ind += 1
                    continue
                flow_graph.clear()
                for i in range(graph.n):
                    flag[i] = False
                for node in fuzzy_v[cur_ind]:
                    flag[node] = True
                for i in range(edge_num):
                    if not flag[self.edge[i][0]] and not flag[self.edge[i][1]]:
                        continue
                    u = self.edge[i][0]
                    v = self.edge[i][1]
                    weight = 1. / (1. + self.edge_w[i][1])
                    if flag[u] and flag[v]:
                        flow_graph.add_edge(u=u, v=v, weight=weight)
                    elif flag[u]:
                        if belong[v] == s:
                            flow_graph.add_edge(u=v, v=u, weight=weight, bidirectional=False)
                            flow_graph.add_edge(u=u, v=v, weight=0., bidirectional=False)
                            flow_graph.add_edge(u=graph.n, v=v, weight=sys.float_info.max, bidirectional=False)
                            flow_graph.add_edge(u=v, v=graph.n, weight=0., bidirectional=False)
                        elif belong[v] == t:
                            flow_graph.add_edge(u=u, v=v, weight=weight, bidirectional=False)
                            flow_graph.add_edge(u=v, v=u, weight=0., bidirectional=False)
                            flow_graph.add_edge(u=v, v=graph.n + 1, weight=sys.float_info.max, bidirectional=False)
                            flow_graph.add_edge(u=graph.n + 1, v=v, weight=0., bidirectional=False)
                    else:
                        if belong[u] == s:
                            flow_graph.add_edge(u=u, v=v, weight=weight, bidirectional=False)
                            flow_graph.add_edge(u=v, v=u, weight=0., bidirectional=False)
                            flow_graph.add_edge(u=graph.n, v=u, weight=sys.float_info.max, bidirectional=False)
                            flow_graph.add_edge(u=u, v=graph.n, weight=0., bidirectional=False)
                        elif belong[u] == t:
                            flow_graph.add_edge(u=v, v=u, weight=weight, bidirectional=False)
                            flow_graph.add_edge(u=u, v=v, weight=0., bidirectional=False)
                            flow_graph.add_edge(u=u, v=graph.n + 1, weight=sys.float_info.max, bidirectional=False)
                            flow_graph.add_edge(u=graph.n + 1, v=u, weight=0., bidirectional=False)
                cost = flow_graph.dinic(s=graph.n, t=graph.n + 1)
                for node in fuzzy_v[cur_ind]:
                    if flow_graph.dinic_vis[node] != -1:
                        belong[node] = s
                    else:
                        belong[node] = t
                cur_ind += 1

        del flag
        del flow_graph
        del fuzzy_v

        return belong
