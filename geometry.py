import math
import os
import shutil
import sys
import json
from typing import Tuple, List

from plyfile import PlyData
import numpy as np
import random

import numba

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
    def __init__(self, pos: np.ndarray, normal: np.ndarray):
        self.pos = pos.copy()
        self.normal = normal.copy()

    def __str__(self):
        return f"Vertex pos: {self.pos}, normal: {self.normal}"

    def __repr__(self):
        return self.__str__()


class Mesh:
    def __init__(self, indexes: np.ndarray, color=default_color):
        self.indexes = indexes.copy()
        self.color = color.copy()

    def __str__(self):
        return f"Mesh color: {self.color}, indexes: {self.indexes}"

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, item):
        return self.indexes[item]

    def __setitem__(self, key, value):
        self.indexes[key] = value


def merge_obj(src_list: list, dst_path: os.path) -> None:
    # merge obj files
    with open(dst_path, 'w') as f_out:
        for src in src_list:
            with open(src, 'r') as f_src:
                for line in f_src:
                    f_out.write(line)


class Geometry:
    eta = 0.2
    delta = 0.8
    epsilon = 1e-8
    fuzzy_epsilon = 0.075
    k_limit = 20
    iter_max = 8

    def __init__(self, vertices: List[Vertex], meshes: List[Mesh]):
        self.v = vertices
        self.f = meshes
        self.edge: List[Tuple[int, int]] = []
        self.edge_w: List[Tuple[float, float]] = []

    def __add__(self, other):
        len_v = len(self.v)
        len_f = len(self.f)

        self.v.extend(other.v.copy())

        f_other = other.f.copy()
        for f in f_other:
            f.indexes += len_v
        self.f.extend(f_other)

        edge_other = other.edge.copy()
        for i in range(len(edge_other)):
            edge = edge_other[i]
            edge_other[i] = (edge[0] + len_f, edge[1] + len_f)
        self.edge.extend(edge_other)

        self.edge_w.extend(other.edge_w.copy())
        return self

    @classmethod
    def from_list(cls, v_list: List[np.ndarray], vn_list: List[np.ndarray], f_list: List[np.ndarray],
                  remove_duplicated: bool, color: np.ndarray = default_color) -> 'Geometry':
        vertices = []
        meshes = []
        index = 0
        vertice_index = {}
        for f in f_list:
            mesh = Mesh(f, color)
            for i in range(3):
                pos = v_list[mesh[i]]
                pos_tuple = tuple(pos)
                if not remove_duplicated or (pos_tuple not in vertice_index):
                    vertice_index[pos_tuple] = index
                    normal = vn_list[mesh[i]]
                    vertices.append(Vertex(pos, normal))
                    index += 1
                mesh[i] = vertice_index[pos_tuple]
            meshes.append(mesh)
        return cls(vertices, meshes)

    @classmethod
    def from_ply(cls, ply_path: os.path, remove_duplicated: bool, color: np.ndarray = default_color) -> 'Geometry':
        logger.assert_true(ply_path.endswith('.ply'), 'File must be a .ply file')
        scene = PlyData.read(ply_path)
        logger.info('Loaded mesh from file:', ply_path)
        logger.info(scene)

        v_list = []
        vn_list = []
        f_list = []
        for i, v in enumerate(scene['vertex']):
            pos = np.array([v['x'], v['y'], v['z']])
            normal = np.array([v['nx'], v['ny'], v['nz']])
            v_list.append(pos)
            vn_list.append(normal)
        for f in scene['face']:
            f_list.append(np.array(f['vertex_indices']))

        return cls.from_list(v_list, vn_list, f_list, remove_duplicated, color)

    @classmethod
    def from_obj(cls, obj_path: os.path, remove_duplicated: bool, color: np.ndarray = default_color) -> 'Geometry':
        logger.assert_true(obj_path.endswith('.obj'), 'File must be a .obj file')
        v_list = []
        vn_list = []
        f_list = []
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    pos = np.array([float(x) for x in line.split()[1:]])
                    v_list.append(pos)
                elif line.startswith('vn '):
                    normal = np.array([float(x) for x in line.split()[1:]])
                    vn_list.append(normal)
                elif line.startswith('f '):
                    vertices = line.split()[1:]
                    face = []
                    for vertice in vertices:
                        v_vt_vn = vertice.split('/')
                        v = int(v_vt_vn[0])
                        vn = int(v_vt_vn[2])
                        logger.assert_true(v == vn, 'v and vn must be the same')
                        face.append(v)
                    f_list.append(np.array(face))
        for face in f_list:
            face += len(v_list)

        return cls.from_list(v_list, vn_list, f_list, remove_duplicated, color)

    @classmethod
    @time_it
    def from_file(cls, file_path: os.path, remove_duplicated: bool, color: np.ndarray = default_color) -> 'Geometry':
        if file_path.endswith('.ply'):
            return cls.from_ply(file_path, remove_duplicated, color)
        elif file_path.endswith('.obj'):
            return cls.from_obj(file_path, remove_duplicated, color)
        else:
            raise Exception('File must be a .ply or .obj file')

    @classmethod
    @time_it
    def from_files(cls, file_paths: list, remove_duplicated: bool) -> 'Geometry':
        geometry = cls([], [])
        for i, file_path in enumerate(file_paths):
            geometry += cls.from_file(file_path, remove_duplicated, palette[i])
        return geometry

    def __str__(self):
        return f"Geometry vertices: {self.v}, meshes: {self.f}, edge: {self.edge}, edge_w: {self.edge_w}"

    def __repr__(self):
        return self.__str__()

    @time_it
    def split_mesh(self, k_max: int = -1) -> 'Geometry':
        logger.info(f'Splitting mesh geometry into k_max={k_max} segments')

        n_node = len(self.f)
        m_edge = len(self.f) * 4
        logger.info(f'n_node = {n_node}, m_edge = {m_edge}')

        graph = Graph(n_node, m_edge)
        self.create_graph(graph, n_node)
        graph.calculate_all_distance()

        rep, k_suggest = self.choose_k(graph=graph, k_max=k_max)
        k = k_suggest
        rep = rep[:k]
        logger.info(f'k = {k} after choose_k')

        self.cluster_k(graph=graph, rep=rep, k=k)
        belong = self.final_decomposition(graph=graph, rep=rep, k=k, n_node=n_node)

        geometry_split = Geometry(vertices=[], meshes=[])
        for i in range(n_node):
            for j in range(3):
                geometry_split.v.append(self.v[self.f[i][j]])
            if belong[i] == -1:
                color = default_color
            else:
                color = palette[belong[i]]
            geometry_split.f.append(Mesh(
                indexes=np.array([i * 3 + j for j in range(3)], dtype=np.int),
                color=color
            ))

        del rep
        del belong
        del graph

        logger.debug(geometry_split)
        return geometry_split

    @time_it
    def create_graph(self, graph: Graph, n_face: int):
        edge_map = {}
        center = []
        normal = []

        self.edge.clear()
        self.edge_w.clear()

        for i in range(n_face):
            vertexes = [self.v[self.f[i][j]] for j in range(3)]
            center.append(
                (vertexes[0].pos +
                 vertexes[1].pos +
                 vertexes[2].pos) / 3)  # center of triangle mesh
            average_normal = normalize(
                (vertexes[0].normal +
                 vertexes[1].normal +
                 vertexes[2].normal))
            face_normal = normalize(
                np.cross(vertexes[0].pos - vertexes[1].pos,
                         vertexes[1].pos - vertexes[2].pos)
            )
            if np.dot(face_normal, average_normal) < 0:
                face_normal = -face_normal
            normal.append(face_normal)

            for j in range(3):
                edge_temp = (self.f[i][j], self.f[i][(j + 1) % 3])
                # ensure edge_temp[0] >= edge_temp[1] for mapping
                if edge_temp[0] < edge_temp[1]:
                    edge_temp = (edge_temp[1], edge_temp[0])
                if edge_temp not in edge_map:
                    edge_map[edge_temp] = i
                else:
                    # overlapping edge
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

        # calculate weight of edges
        geo_dis_average = 0.
        angle_dis_average = 0.
        for dis_tuple in self.edge_w:
            geo_dis_average += dis_tuple[0]
            angle_dis_average += dis_tuple[1]
        edge_num = len(self.edge)
        geo_dis_average /= edge_num
        angle_dis_average /= edge_num
        for i in range(edge_num):
            self.edge_w[i] = (self.edge_w[i][0] / geo_dis_average,
                              self.edge_w[i][1] / angle_dis_average)
            # edge weight = alpha * geo_dis + (1 - alpha) * angle_dis
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
        cos_alpha_uv = np.dot(normal_u, normal_v)
        if np.dot(center_uv, normal_u) < 0:  # Convex
            angle_dis = self.eta * (1 - cos_alpha_uv)
        else:  # concave
            angle_dis = 1 - cos_alpha_uv

        return geo_dis, angle_dis

    @time_it
    def choose_k(self, graph: Graph, k_max: int) -> Tuple[np.ndarray, int]:
        lim = k_max
        if lim <= 0:
            lim = self.k_limit
        rep = np.zeros(lim, dtype=np.int)
        Gk = []

        choose_min = sys.float_info.max
        # First, choose the first representative
        for i in range(graph.n):
            dis_sum = graph.distance[i].sum()
            if dis_sum < choose_min:
                choose_min = dis_sum
                rep[0] = i
        # Add other representatives 1 by 1
        for cur in range(1, lim):
            choose_max = -sys.float_info.max
            for i in range(graph.n):
                dis_min = sys.float_info.max
                for j in range(cur):
                    dis_min = min(dis_min, graph.distance[i][rep[j]])
                if dis_min > choose_max:
                    choose_max = dis_min
                    rep[cur] = i
            Gk.append(choose_max)

        if k_max == 2:
            # Find the most distant pair of representatives
            dis_max = -sys.float_info.max
            for i in range(graph.n):
                for j in range(i + 1, graph.n):
                    if graph.distance[i][j] > dis_max:
                        rep[0] = i
                        rep[1] = j
                        dis_max = graph.distance[i][j]

        # Choose k_suggest by the greatest 1st derivative of G
        k_suggest = 2
        grad_max = -sys.float_info.max
        for i in range(1, lim - 1):
            grad = Gk[i - 1] - Gk[i]
            if grad > grad_max:
                grad_max = grad
                k_suggest = i + 2

        logger.info(' '.join(map(
            lambda x: f'G({x[0] + 2})={x[1]}',
            enumerate(Gk)
        )))

        del Gk
        return rep, k_suggest

    @time_it
    def cluster_k(self, graph: Graph, rep: np.ndarray, k: int):
        rep_bak = np.zeros(k, dtype=np.int)
        belong = np.zeros(graph.n, dtype=np.int)
        converge = False
        p = np.zeros(shape=(k, graph.n), dtype=np.float)  # probability of each face belonging to each cluster
        has: List[List[int]] = [[] for _ in range(k)]  # has[i] = [j, ...] means rep[i] has j
        iter = 0

        while not converge:
            iter += 1
            for i in range(k):
                rep_bak[i] = rep[i]
                has[i].clear()
            logger.info(f'Iter {iter}: REP = ' + ' '.join(map(str, rep)))

            for j in range(graph.n):
                belong[j] = -1
                dis_inv_sum = 0.  # denominator of P_pj(fi)
                for i in range(k):
                    if graph.distance[rep[i]][j] < self.epsilon:
                        belong[j] = i
                        break
                    dis_inv_sum += 1. / graph.distance[rep[i]][j]
                if belong[j] != -1:
                    # if j is a representative, then P_pj(fi) = 1
                    for i in range(k):
                        p[i][j] = 0.
                    p[belong[j]][j] = 1.
                else:
                    # if j is not a representative, then P_pj(fi) = dist / dis_inv_sum
                    p_max = -sys.float_info.max
                    for i in range(k):
                        # set p_max as belong
                        p[i][j] = (1. / graph.distance[rep[i]][j]) / dis_inv_sum
                        if p[i][j] > p_max:
                            p_max = p[i][j]
                            belong[j] = i
                has[belong[j]].append(j)

            # recalculate p
            for j in range(graph.n):
                belong[j] = -1
                dis_inv_sum = 0.
                for i in range(k):
                    if graph.distance[rep[i]][j] < self.epsilon:
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
                        rep[i] = cur

            check_equal = False
            for i in range(k):
                for j in range(i):
                    if rep[i] == rep[j]:
                        check_equal = True
            if check_equal:
                for i in range(k):
                    rep[i] = rep_bak[i]
            # check if rep changes during this iter
            converge = True
            for i in range(k):
                if rep[i] != rep_bak[i]:
                    converge = False
                    break
            # break if iter reaches limit
            if iter == self.iter_max:
                break

        del belong
        del rep_bak
        del p
        del has

    @time_it
    def final_decomposition(self, graph: Graph, rep: np.ndarray, k: int, n_node: int) -> np.ndarray:
        belong = np.zeros(n_node, dtype=np.int)
        fuzzy_type = k * (k - 1) >> 1
        fuzzy_v: List[List[int]] = [[] for _ in range(fuzzy_type)]
        for j in range(graph.n):
            dis_min = dis_cmin = sys.float_info.max
            belong_min = belong_cmin = -1
            for i in range(k):
                if graph.distance[rep[i]][j] < dis_min:
                    dis_cmin = dis_min
                    belong_cmin = belong_min
                    dis_min = graph.distance[rep[i]][j]
                    belong_min = i
                elif graph.distance[rep[i]][j] < dis_cmin:
                    dis_cmin = graph.distance[rep[i]][j]
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
                for i in fuzzy_v[cur_ind]:
                    flag[i] = True
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

    @time_it
    def separate_face_by_color(self) -> dict:
        color_map = {}
        for mesh in self.f:
            key_color = (mesh.color[0], mesh.color[1], mesh.color[2])
            if key_color not in color_map:
                color_map[key_color] = []
            color_map[key_color].append(mesh)

        logger.debug(f'len(self.f) = {len(self.f)}')
        logger.debug(color_map)
        color_mesh_num = {}
        for color, mesh_list in color_map.items():
            color_mesh_num[color] = len(mesh_list)
        logger.info(f'color_map: {color_mesh_num}')

        return color_map

    @time_it
    def export_obj(self, root_dir: os.path) -> list:
        model_dir = os.path.join(root_dir, 'obj')
        os.makedirs(model_dir, exist_ok=True)
        mtl_filename = 'material.mtl'
        model_filename = os.path.join(model_dir, f'model.obj')

        colors = {}
        index = 0
        color_map = self.separate_face_by_color()
        with open(model_filename, 'w') as file:
            file.write(f'mtllib {mtl_filename}\n')
            for color, faces in color_map.items():
                obj = [
                    '\n',
                    f'g obj_part_{index}\n',
                    f'usemtl color_{index}\n',
                ]
                colors[f'color_{index}'] = color

                f_num = len(faces)
                pos = []
                normal = []
                for f in faces:
                    for i in range(3):
                        pos.append(self.v[f[i]].pos)
                        normal.append(self.v[f[i]].normal)
                assert (f_num * 3) == len(pos) == len(normal)
                for x in pos:
                    obj.append(f'v {x[0]} {x[1]} {x[2]}\n')
                for n in normal:
                    obj.append(f'vn {n[0]} {n[1]} {n[2]}\n')
                for i in range(-f_num, 0):
                    obj.append(f'f {3 * i}//{3 * i} {3 * i + 1}//{3 * i + 1} {3 * i + 2}//{3 * i + 2}\n')

                file.writelines(obj)
                index += 1
        with open(os.path.join(model_dir, mtl_filename), 'w') as f:
            for name, color in colors.items():
                f.writelines([
                    f'newmtl {name}\n',
                    f'\tKa 1.0 1.0 1.0\n',
                    f'\tKd {color[0]} {color[1]} {color[2]}\n\n',
                ])

        return [model_filename]

    @time_it
    def export_ply(self, root_dir: os.path) -> list:
        model_dir = os.path.join(root_dir, 'ply')
        os.makedirs(model_dir, exist_ok=True)
        model_filename = os.path.join(model_dir, f'model.ply')

        with open(model_filename, 'w') as file:
            file.writelines([
                'ply\n',
                'format ascii 1.0\n',
                'comment author: Xin Chen\n',
                f'element vertex {len(self.v)}\n',
                'property float x\n',
                'property float y\n',
                'property float z\n',
                'property float nx\n',
                'property float ny\n',
                'property float nz\n',
                'property uchar red\n',
                'property uchar green\n',
                'property uchar blue\n',
                f'element face {len(self.f)}\n',
                'property list uchar int vertex_indices\n',
                'end_header\n',
            ])

            for index, v in enumerate(self.v):
                color = np.array(self.f[index // 3].color * 255, dtype=np.uint8)
                file.write(
                    f'{v.pos[0]} {v.pos[1]} {v.pos[2]} '
                    f'{v.normal[0]} {v.normal[1]} {v.normal[2]} '
                    f'{color[0]} {color[1]} {color[2]}\n')

            for face in self.f:
                file.write(f'3 {face[0]} {face[1]} {face[2]}\n')

        return [model_filename]

    @time_it
    def export_opengl_render(self, root_dir: os.path) -> list:
        model_dir = os.path.join(root_dir, 'opengl-render')
        os.makedirs(model_dir, exist_ok=True)
        scene_filename = os.path.join(model_dir, f'scene.json')

        model_filenames = []
        colors = {}
        color2mesh = {}
        index = 0
        color_map = self.separate_face_by_color()
        aabb = np.array([
            [sys.float_info.max, sys.float_info.max, sys.float_info.max],
            [-sys.float_info.max, -sys.float_info.max, -sys.float_info.max],
        ])
        for color, faces in color_map.items():
            mesh_filename = os.path.join(model_dir, f'obj_part_{index}.obj')
            while os.path.exists(mesh_filename):
                index += 1
                mesh_filename = os.path.join(model_dir, f'obj_part_{index}.obj')
            color_name = f'color_{index}'
            colors[color_name] = color
            color2mesh[color_name] = mesh_filename
            model_filenames.append(mesh_filename)

            f_num = len(faces)
            v_list = []
            vn_list = []
            for f in faces:
                for i in range(3):
                    pos = self.v[f[i]].pos
                    aabb[0] = np.minimum(aabb[0], pos)
                    aabb[1] = np.maximum(aabb[1], pos)
                    normal = self.v[f[i]].normal
                    v_list.append(f'v {pos[0]} {pos[1]} {pos[2]}\n')
                    vn_list.append(f'vn {normal[0]} {normal[1]} {normal[2]}\n')
            obj = v_list + vn_list

            for i in range(-f_num, 0):
                obj.append(f'f {3 * i}//{3 * i} {3 * i + 1}//{3 * i + 1} {3 * i + 2}//{3 * i + 2}\n')

            with open(color2mesh[color_name], 'w') as file:
                file.writelines(obj)

        materials = []
        meshes = []
        lights = []

        center = (aabb[0] + aabb[1]) / 2
        size = aabb[1] - aabb[0]
        half_size = size / 2
        logger.info(f'aabb: {aabb}, center: {center}, size: {size}')
        light_dis = 5
        light_template = {
            "emission": [1.0, 1.0, 1.0],
            "scale": (light_dis * half_size.mean().item()) ** 2,
            "position": [0, 0, 0],
        }
        for i in [-1, 1]:
            light = light_template.copy()
            light['position'] = list(center)
            light['position'][0] += i * half_size[0] * light_dis
            light['position'][1] += half_size[1] * light_dis
            lights.append(light)

            light = light_template.copy()
            light['position'] = list(center)
            light['position'][1] += half_size[1] * light_dis
            light['position'][2] += i * half_size[2] * light_dis
            lights.append(light)

        camera_pos = center + half_size[0] * 2
        camera = {
            "resolution": [1024, 1024],
            "position": list(camera_pos),
            "front": list(normalize(center - camera_pos)),
            "up": [0, 1, 0],
            "fov": 60,
        }
        for name, color in colors.items():
            materials.append({
                'name': name,
                'type': 'phong',
                'diffuse': list(color),
            })
            meshes.append({
                "transform": {},
                "file": os.path.relpath(color2mesh[name], os.path.dirname(scene_filename)),
                "material": name,
            })
        scene = {
            'materials': materials,
            'meshes': meshes,
            'lights': lights,
            'camera': camera,
            'renderer': {
                "enable_vsync": True,
                "enable_shadow": True,
                "output_file": "output.png"
            }
        }

        with open(scene_filename, 'w') as f:
            json.dump(scene, f, indent=4)

        return model_filenames
