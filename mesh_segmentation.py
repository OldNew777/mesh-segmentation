import argparse
import os
import sys
import random
from argparse import ArgumentParser
import shutil

from mylogger import logger
from geometry import *


def parse_args(argv):
    parser = ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, help='The path to the .ply file', default='./data/horse.ply')
    parser.add_argument('-k', '--k_max', type=int, help='The max number of clusters', default=20)
    parser.add_argument('-s', '--seed', type=int, help='The seed of random', default=3984572)
    parser.add_argument('--n_hierarchy', type=int, help='The layers of hierarchy', default=2)
    return parser.parse_known_args(argv)[0]


def mesh_segmentation(filename: os.path, k_max: int) -> Geometry:
    if k_max <= 0:
        logger.error('k must be greater than 0')
        sys.exit(1)

    meshes = Geometry.from_file(filename, remove_duplicated=True)
    return meshes.split_mesh(k_max=k_max)


def hierarchical_mesh_segmentation(args: argparse.Namespace, output_dir: os.path):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    model_filenames = [os.path.relpath(args.filename)]
    temp_dir = os.path.join(output_dir, 'temp')

    # model_filenames = [os.path.relpath(f'./outputs/level-2/opengl-render/obj_part_{j}.obj') for j in range(4)]
    # geometry = Geometry.from_files(model_filenames, remove_duplicated=False)
    # geometry.export_opengl_render(root_dir=temp_dir)
    # sys.exit(-1)

    for i in range(1, args.n_hierarchy + 1):
        logger.info('Start to segment the meshes in No.{} hierarchy'.format(i))
        output_dir_i = os.path.join(output_dir, f'level-{i}')
        models_temp = []
        for model_filename in model_filenames:
            geometry = mesh_segmentation(model_filename, args.k_max)
            models_temp += geometry.export_opengl_render(root_dir=temp_dir)
        # reload and export to get the right output model
        geometry = Geometry.from_files(models_temp, remove_duplicated=False)
        shutil.rmtree(temp_dir)
        geometry.export_ply(root_dir=output_dir_i)
        geometry.export_obj(root_dir=output_dir_i)
        model_filenames = geometry.export_opengl_render(root_dir=output_dir_i)
        logger.info('Finish to segment the meshes in the No.{} hierarchy'.format(i))


if __name__ == "__main__":
    logger.set_level(logger.INFO)
    args = parse_args(sys.argv[1:])
    logger.info(f'Init with random seed {args.seed}')
    random.seed(args.seed)
    init_palette(max(args.k_max, Geometry.k_limit) ** args.n_hierarchy)

    output_dir = os.path.relpath('outputs')
    hierarchical_mesh_segmentation(args=args, output_dir=output_dir)
