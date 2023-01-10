import os
import sys
import numpy as np
import random
from argparse import ArgumentParser

from mylogger import logger
from geometry import Geometry, init_palette
from graph import Graph


def mesh_segmentation(filename: os.path, k: int) -> Geometry:
    if k <= 0:
        logger.error('k must be greater than 0')
        sys.exit(1)

    meshes = Geometry.from_ply(filename)
    return meshes.split_mesh(k=k)


def parse_args(argv):
    parser = ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, help='The path to the .ply file', default='./data/horse.ply')
    parser.add_argument('-k', '--k', type=int, help='The number of clusters', default=4)
    return parser.parse_known_args(argv)[0]


if __name__ == "__main__":
    logger.set_level(logger.INFO)
    seed = 3984572
    logger.info(f'Init with random seed {seed}')
    random.seed(seed)
    args = parse_args(sys.argv[1:])
    init_palette(max(args.k, Geometry.k_limit))
    geometry = mesh_segmentation(filename=os.path.relpath(args.filename), k=args.k)
    geometry.dump_obj(root_dir=os.path.relpath('./outputs'))
