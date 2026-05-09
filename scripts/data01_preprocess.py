#!/usr/bin/env python3
from __future__ import annotations

from _bootstrap import bootstrap_repo_root

bootstrap_repo_root()
from src.datasets.data01_preprocess import main


if __name__ == "__main__":
    main()
