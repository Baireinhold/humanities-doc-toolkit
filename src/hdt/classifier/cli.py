# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from .engine import ClassifierEngine


def main():
    p = argparse.ArgumentParser(
        prog="hdt-classifier",
        description="人文学科文献归类(CLI) - 默认 dry-run：只生成 planned_to 清单与日志，不移动文件"
    )
    p.add_argument("--global-config", default="global.yaml", help="全局配置(共享 ai_services)")
    p.add_argument("--config", default="classifier.yaml", help="Classifier 专用配置")
    args = p.parse_args()

    engine = ClassifierEngine(args.global_config, args.config)
    raise SystemExit(engine.run_interactive(dry_run=True))


def apply_main():
    p = argparse.ArgumentParser(
        prog="hdt-classifier-apply",
        description="按上次 JSON 日志执行移动(apply) - 读取 planned_to 并落盘"
    )
    p.add_argument("--global-config", default="global.yaml", help="全局配置(共享 ai_services)")
    p.add_argument("--config", default="classifier.yaml", help="Classifier 专用配置")
    p.add_argument("--log", required=True, help="dry-run 生成的 classification_details_*.json 路径")
    args = p.parse_args()

    engine = ClassifierEngine(args.global_config, args.config)
    raise SystemExit(engine.apply_moves_from_log(args.log))