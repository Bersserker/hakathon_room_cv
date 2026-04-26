from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_repo_root() -> Path:
    """Добавляет корень репозитория в `sys.path`.

    Пояснение: нужен, чтобы wrapper из `scripts/` мог импортировать код из `src/`.
    """
    root_dir = Path(__file__).resolve().parents[1]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    return root_dir
