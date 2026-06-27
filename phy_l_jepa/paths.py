from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_root() -> Path:
    env_value = os.environ.get("PHY_L_JEPA_DATA_ROOT")
    candidates = []
    if env_value:
        candidates.append(Path(env_value).expanduser())

    root = project_root()
    candidates.extend(
        [
            root / "data" / "GIGADATASET_COLAB_NPZ",
            Path.cwd() / "data" / "GIGADATASET_COLAB_NPZ",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[1]
