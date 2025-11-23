from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from s2c.config import load_config, PROJECT_ROOT
from s2c.engine import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Unified S2C training entry point')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/clcd.yaml',
        help='Path to the config file (relative to repo root by default)',
    )
    parser.add_argument(
        '--options',
        nargs='*',
        default=None,
        help='Override config fields via key=value pairs, e.g. training.lr=0.001',
    )
    return parser.parse_args()


def run_with_config(config_path: str | Path, overrides: Optional[Iterable[str]] = None) -> None:
    cfg = load_config(config_path, overrides)
    run_training(cfg)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    run_with_config(config_path, args.options)


if __name__ == '__main__':
    main()

