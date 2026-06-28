"""Deterministic SHA-1 helpers for input CSVs and map occupancy files."""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


def sha1_file(path: str | Path) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def sha1_map(map_yaml_path: str | Path) -> str:
    """Hash the .pgm referenced by the map YAML (the actual occupancy data)."""
    map_yaml_path = Path(map_yaml_path)
    with open(map_yaml_path) as f:
        cfg = yaml.safe_load(f)
    image_rel = cfg.get('image', '')
    if not image_rel:
        raise ValueError(f"No 'image:' key in {map_yaml_path}")
    image_path = (map_yaml_path.parent / image_rel).resolve()
    return sha1_file(image_path)
