import datetime as dt
import os
from typing import Any

import yaml


NETWORK_THROTTLE_MARKERS = ("akshare", "cache_incremental", "cache_stale")


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def today_str() -> str:
    return dt.datetime.now().strftime("%Y%m%d")


def resolve_template(template: str, **kwargs: str) -> str:
    return template.format(**kwargs)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def should_throttle_after_source(source: str | None, had_error: bool = False) -> bool:
    if had_error:
        return True
    source_text = str(source or "").lower()
    return any(marker in source_text for marker in NETWORK_THROTTLE_MARKERS)
