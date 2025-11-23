from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """
    Load the YAML config and apply CLI overrides (key.sub=value).
    The returned dict includes extra fields:
      - project_root: repository root path
      - config_path: absolute path to the loaded config
    All paths under `paths.*` plus `training.load_path` (when present) are resolved relative to the repo root.
    """
    overrides = overrides or []
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')

    with path.open('r', encoding='utf-8') as fp:
        data = yaml.safe_load(fp) or {}

    _apply_overrides(data, overrides)

    data['project_root'] = PROJECT_ROOT
    data['config_path'] = path
    paths_cfg = data.get('paths', {})
    data['paths'] = {k: _resolve_path(v) for k, v in paths_cfg.items()}

    training_cfg = data.get('training', {})
    load_path = training_cfg.get('load_path')
    if load_path:
        training_cfg['load_path'] = _resolve_path(load_path)
    data['training'] = training_cfg

    return data


def _apply_overrides(config: MutableMapping[str, Any], overrides: Iterable[str]) -> None:
    for override in overrides:
        if '=' not in override:
            raise ValueError(f'Override format must be key=value, got: {override}')
        key, raw_value = override.split('=', 1)
        keys = key.split('.')
        value = yaml.safe_load(raw_value)
        _set_by_path(config, keys, value)


def _set_by_path(config: MutableMapping[str, Any], keys: List[str], value: Any) -> None:
    curr: MutableMapping[str, Any] = config
    for key in keys[:-1]:
        if key not in curr or not isinstance(curr[key], MutableMapping):
            curr[key] = {}
        curr = curr[key]
    curr[keys[-1]] = value


def _resolve_path(path_like: Any) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


__all__ = ['load_config', 'PROJECT_ROOT']

