import os
from plyfile import PlyData
import numpy as np
import random

from mylogger import logger


palette = []


def init_palette(k: int):
    for i in range(k):
        palette.append(np.array([random.random(),
                                 random.random(),
                                 random.random()]))


class Vertex:
    def __init__(self, pos, normal, color=np.array([0.0, 0.0, 0.0])):
        self.pos = pos
        self.normal = normal
        self.color = color

    def __str__(self):
        return f"Vertex pos: {self.pos}, normal: {self.normal}, color: {self.color}"

    def __repr__(self):
        return self.__str__()


class Mesh:
    def __init__(self, indexes):
        self.indexes = indexes

    def __str__(self):
        return f"Mesh indexes: {self.indexes}"

    def __repr__(self):
        return self.__str__()


class Geometry:
    def __init__(self, vertices, meshes):
        self.v = vertices
        self.f = meshes
        self.edge = None
        self.edge_w = None

        self.eta = 0.2
        self.delta = 0.8
        self.epsilon = 1e-8
        self.fuzzy_epsilon = 0.075
        self.k_limit = 20
        self.iter_max = 8

        logger.debug(self.v)
        logger.debug(self.f)

    @classmethod
    def from_ply(cls, ply_path: os.path):
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

    def split_mesh(self, k: int = -1):
        logger.info(f'Splitting mesh into {k} segments')

        n_node = len(self.f) / 3
        m_edge = len(self.f) << 2

        return 0
