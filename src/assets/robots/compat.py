"""Asset loading compatibility for Unitree's mjlab 1.2 model factories."""
from pathlib import Path


def update_assets(assets, path, meshdir=None, glob='*', recursive=False):
    for entry in Path(path).glob(glob):
        if entry.is_file():
            key = f'{meshdir}/{entry.name}' if meshdir else entry.name
            assets[key] = entry.read_bytes()
        elif recursive and entry.is_dir():
            update_assets(assets, entry, meshdir, glob, recursive)
