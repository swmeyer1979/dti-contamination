from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def _sentinel_path(name: str) -> Path:
    return CHECKPOINTS_DIR / f"{name}.done"


def sentinel_exists(name: str) -> bool:
    return _sentinel_path(name).exists()


def require_sentinel(name: str) -> None:
    path = _sentinel_path(name)
    if not path.exists():
        raise RuntimeError(
            f"Required sentinel '{name}' not found at {path}. "
            f"Run the prerequisite phase to create checkpoints/{name}.done."
        )


def write_sentinel(name: str) -> Path:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _sentinel_path(name)
    ts = datetime.now(timezone.utc).isoformat()
    path.write_text(f"{ts}\n")
    return path

