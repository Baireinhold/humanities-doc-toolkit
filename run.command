#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

LOG="$ROOT/run_macos_launch.log"
exec > >(tee -a "$LOG") 2>&1

echo "================================================"
echo "Humanities Doc Toolkit (v0.1.0) - macOS Launcher"
echo "Workdir: $ROOT"
echo "Log: $LOG"
echo "================================================"
# 1) 强制优先用 Homebrew python3.12（你机器上已存在）：
# command -v python3.12 输出为 /opt/homebrew/bin/python3.12 [user_query]
PY=""

candidates=(python3.13 python3.12 python3.11 python3.10 python3 python)

for c in "${candidates[@]}"; do
  if command -v "$c" >/dev/null 2>&1; then
    if "$c" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' >/dev/null 2>&1; then
      PY="$c"
      break
    fi
  fi
done

if [[ -z "$PY" ]]; then
  echo "[ERROR] 未找到 Python>=3.10。本项目要求 >=3.10。" 
  read -p "按回车退出..."
  exit 1
fi

# 2) 版本检查（必须>=3.10）[13]
"$PY" -c 'import sys; assert sys.version_info >= (3,10), sys.version'
echo "[INFO] Using: $PY ($($PY --version))"

# 3) 创建/复用 venv + 安装
if [[ ! -x ".venv/bin/python" ]]; then
  echo "[INFO] Creating venv: .venv"
  "$PY" -m venv .venv
fi

echo "[INFO] Installing deps (first time may be slow)..."
.venv/bin/python -m pip install -U pip setuptools wheel
.venv/bin/python -m pip install -e .   # 依赖来自 pyproject.toml [13]

# 4) 启动主菜单（你的统一入口是 hdt = hdt.cli:main）[13]
echo "[INFO] Launching: python -m hdt.cli"
.venv/bin/python -m hdt.cli

read -p "按回车退出..."