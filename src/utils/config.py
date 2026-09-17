"""YAML config loader."""

import os
import re
import yaml


def load_config(path):
    """Load YAML config, resolving ${paths.xxx} references."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # First pass: extract all paths.xxx values
    cfg = yaml.safe_load(raw)

    # Resolve ${...} references within paths section
    paths = cfg.get("paths", {})
    for key, value in list(paths.items()):
        if isinstance(value, str):
            for pk, pv in paths.items():
                value = value.replace(f"${{paths.{pk}}}", str(pv))
            paths[key] = value

    # Resolve references in other sections
    def resolve(obj):
        if isinstance(obj, dict):
            return {k: resolve(v) for k, v in obj.items()}
        elif isinstance(obj, str):
            for pk, pv in paths.items():
                obj = obj.replace(f"${{paths.{pk}}}", str(pv))
            return obj
        return obj

    for section in cfg:
        if section != "paths":
            cfg[section] = resolve(cfg[section])

    cfg["paths"] = paths
    return cfg


def get_task_config(cfg):
    return cfg.get("tasks", {})
