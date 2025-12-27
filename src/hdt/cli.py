# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from .renamer.engine import RenamerEngine
from .classifier.engine import ClassifierEngine
from .sorter.engine import SorterEngine
from .__about__ import __title__, __version__, __author__, __email__, __github__


def _banner():
    print("=" * 72)
    print(f"{__title__}  v{__version__}")
    print(f"Author: {__author__} | {__email__}")
    print(__github__)
    print("=" * 72)
    print("提示:首次运行请编辑 global.yaml 填入 API key")
    print()


def _ensure(dst: str, src: str):
    d = Path(dst)
    if d.exists():
        return
    s = Path(src)
    if s.exists():
        d.write_text(s.read_text(encoding="utf-8"), encoding="utf-8")


def _bootstrap_configs():
    # README 里说明示例配置从 configs/ 复制到根目录 [1]
    _ensure("global.yaml", "configs/global.example.yaml")
    _ensure("renamer.yaml", "configs/renamer.example.yaml")
    _ensure("classifier.yaml", "configs/classifier.example.yaml")
    _ensure("sorter.yaml", "configs/sorter.example.yaml")


def main():
    _banner()
    _bootstrap_configs()

    while True:
        print("请选择功能:")
        print("1) Renamer    批量重命名/标准化")
        print("2) Classifier 先 dry-run 生成 JSON,并默认打开 JSON 核对")
        print("3) Sorter     按研究需求智能分拣(默认 copy)")
        print("4) Classifier-Apply  从 JSON 清单执行移动(单独任务)")
        print("0) 退出")
        c = input("输入选项(0-4): ").strip()

        if c == "1":
            code = RenamerEngine("global.yaml", "renamer.yaml").run_interactive()
            print(f"\n(已返回主菜单，退出码={code})\n")
            continue
        elif c == "2":
            code = ClassifierEngine("global.yaml", "classifier.yaml").run_interactive(dry_run=True)
            print(f"\n(已返回主菜单，退出码={code})\n")
            continue
        elif c == "3":
            code = SorterEngine("global.yaml", "sorter.yaml").run_interactive()
            print(f"\n(已返回主菜单，退出码={code})\n")
            continue
        elif c == "4":
            log_path = input("请输入 classification_details_*.json 路径: ").strip().strip('"')
            if not log_path:
                print("路径不能为空。\n")
                continue
            engine = ClassifierEngine("global.yaml", "classifier.yaml")
            code = engine.apply_moves_from_log(log_path)
            print(f"\n(已返回主菜单，退出码={code})\n")
            continue
        elif c == "0":
            return 0
        else:
            print("无效输入,请重试。\n")
if __name__ == "__main__":
    main()