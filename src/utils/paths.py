from datetime import datetime
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamped_path(root: Path, stem: str, suffix: str) -> Path:
    ensure_dir(root)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    return root / f"{stem}_{ts}{suffix}"
