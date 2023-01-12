import os
import sys
import random
from argparse import ArgumentParser
import shutil

from mylogger import logger
from geometry import Geometry, init_palette


def mesh_segmentation(filename: os.path, k_max: int) -> Geometry:
    if k_max <= 0:
        logger.error('k must be greater than 0')
        sys.exit(1)

    meshes = Geometry.from_ply(filename)
    return meshes.split_mesh(k_max=k_max)


def parse_args(argv):
    parser = ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, help='The path to the .ply file', default='./data/horse.ply')
    parser.add_argument('-k', '--k_max', type=int, help='The max number of clusters', default=20)
    parser.add_argument('-s', '--seed', type=int, help='The seed of random', default=3984572)
    return parser.parse_known_args(argv)[0]


if __name__ == "__main__":
    logger.set_level(logger.INFO)
    args = parse_args(sys.argv[1:])
    logger.info(f'Init with random seed {args.seed}')
    random.seed(args.seed)
    init_palette(max(args.k_max, Geometry.k_limit))
    geometry = mesh_segmentation(filename=os.path.relpath(args.filename), k_max=args.k_max)

    root_dir = os.path.relpath('outputs')
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    geometry.export_ply(root_dir=root_dir)
    geometry.export_obj(root_dir=root_dir)
    geometry.export_opengl_render(root_dir=root_dir)
