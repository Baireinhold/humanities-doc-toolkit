from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件格式必须是 YAML dict: {path}")
    return data


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _normalize_ai_services(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """兼容映射：旧版 Classifier 使用 ai.services；工具链统一为 ai_services.services [2]."""
    if "ai_services" in cfg:
        return cfg

    if "ai" in cfg and isinstance(cfg.get("ai"), dict):
        ai = cfg.get("ai") or {}
        services = ai.get("services") if isinstance(ai.get("services"), dict) else None
        if services is not None:
            cfg = dict(cfg)
            cfg["ai_services"] = {"services": services}
    return cfg


def load_merged_config(
    global_path: str = "global.yaml",
    tool_path: Optional[str] = None,
) -> Dict[str, Any]:
    global_cfg = _normalize_ai_services(_load_yaml(Path(global_path)))
    tool_cfg = _normalize_ai_services(_load_yaml(Path(tool_path))) if tool_path else {}
    merged = _deep_merge(global_cfg, tool_cfg)

    ai_services = merged.get("ai_services", {})
    if not isinstance(ai_services, dict) or not isinstance(ai_services.get("services"), dict):
        raise ValueError("配置缺少 ai_services.services（请检查 global.yaml）")

    return merged