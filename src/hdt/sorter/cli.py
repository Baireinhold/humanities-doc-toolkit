# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from .engine import SorterEngine

def main():
    p = argparse.ArgumentParser(prog="hdt-sorter", description="智能文档分拣（两轮筛选）- CLI")
    p.add_argument("--global-config", default="global.yaml", help="全局配置（共享 ai_services）")
    p.add_argument("--config", default="sorter.yaml", help="Sorter 专用配置")
    args = p.parse_args()

    engine = SorterEngine(args.global_config, args.config)
    raise SystemExit(engine.run_interactive())