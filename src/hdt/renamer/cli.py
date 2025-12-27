# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from .engine import RenamerEngine

def main():
    parser = argparse.ArgumentParser(
        prog="hdt-renamer",
        description="人文学科文献重命名（AI辅助）- CLI"
    )
    parser.add_argument("--global-config", default="global.yaml", help="全局配置（共享 ai_services）")
    parser.add_argument("--config", default="renamer.yaml", help="Renamer 专用配置")
    parser.add_argument("-i", "--input", dest="input_dir", help="输入目录（包含PDF）")
    parser.add_argument("-o", "--output", dest="output_dir", help="输出目录（可选；分类模式下会创建子文件夹）")
    args = parser.parse_args()

    engine = RenamerEngine(args.global_config, args.config)
    raise SystemExit(engine.run_interactive(input_dir=args.input_dir, output_dir=args.output_dir))