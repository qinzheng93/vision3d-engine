import os
import os.path as osp
import pickle
from typing import List

from numpy import ndarray
from tqdm import tqdm

from .distributed import master_only

# pickle utilities


def load_pickle(filename: str):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def dump_pickle(data, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


# reading / writing


def readlines(filename: str) -> List[str]:
    with open(filename, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


def writelines(lines: List[str], filename: str, mode: str = "w"):
    lines = [line + "\n" if not line.endswith("\n") else line for line in lines]
    with open(filename, mode) as f:
        f.writelines(lines)


# directory utilities


@master_only
def ensure_dir(path: str):
    if not osp.exists(path):
        os.makedirs(path)
    else:
        assert osp.isdir(path), f"'{path}' already exists but is not a directory."


# special functions


def write_correspondences(file_name: str, src_corr_points: ndarray, tgt_corr_points: ndarray):
    if not file_name.endswith(".obj"):
        file_name += ".obj"

    v_lines = []
    l_lines = []

    num_corr = src_corr_points.shape[0]
    for i in tqdm(range(num_corr)):
        n = i * 2

        src_point = src_corr_points[i]
        tgt_point = tgt_corr_points[i]

        line = "v {:.6f} {:.6f} {:.6f}\n".format(src_point[0], src_point[1], src_point[2])
        v_lines.append(line)

        line = "v {:.6f} {:.6f} {:.6f}\n".format(tgt_point[0], tgt_point[1], tgt_point[2])
        v_lines.append(line)

        line = "l {} {}\n".format(n + 1, n + 2)
        l_lines.append(line)

    with open(file_name, "w") as f:
        f.writelines(v_lines)
        f.writelines(l_lines)
