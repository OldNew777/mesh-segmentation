import os
import sys
from plyfile import PlyData, PlyElement
from mylogger import logger


def load_mesh(filename: os.path):
    logger.assert_true(filename.endswith('.ply'), 'File must be a .ply file')
    scene = PlyData.read(filename)
    logger.debug(scene)
    logger.debug(type(scene['vertex'].data[0]))
    logger.debug(scene['face'].data)
    return scene


def mesh_segmentation(filename: os.path):
    meshes = load_mesh(filename)


if __name__ == "__main__":
    mesh_segmentation(sys.argv[1])
