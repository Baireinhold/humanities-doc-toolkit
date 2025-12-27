#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

LOG="$ROOT_DIR/run_unix.log"
SENTINEL="$ROOT_DIR/.venv/.hdt_installed"

echo "================================================" | tee "$LOG"
echo "Humanities Doc Toolkit (v0.1.0) - Launcher" | tee -a "$LOG"
echo "Workdir: $ROOT_DIR" | tee -a "$LOG"
echo "================================================" | tee -a "$LOG"

# 0) 选择 Python：优先 python3，其次 python；并检查 >=3.10（与 requires-python 对齐）[13]
PYBIN=""
if command -v python3 >/dev/null 2>&1; then PYBIN="python3"; fi
if [[ -z "$PYBIN" ]] && command -v python >/dev/null 2>&1; then PYBIN="python"; fi
if [[ -z "$PYBIN" ]]; then
  echo "[ERROR] 未检测到 python3/python。请安装 Python >= 3.10。" | tee -a "$LOG"
  exit 1
fi

"$PYBIN" -c 'import sys; assert sys.version_info >= (3,10), sys.version' >/dev/null

# 1) 创建 venv
if [[ ! -x ".venv/bin/python" ]]; then
  echo "[INFO] 创建虚拟环境 .venv ..." | tee -a "$LOG"
  "$PYBIN" -m venv .venv
fi

# 2) 首次安装/后续跳过（哨兵）
if [[ ! -f "$SENTINEL" ]]; then
  echo "[INFO] 首次安装依赖(可能较慢)..." | tee -a "$LOG"
  .venv/bin/python -m pip install -U pip setuptools wheel | tee -a "$LOG"
  .venv/bin/python -m pip install -e . | tee -a "$LOG"
  echo "installed" > "$SENTINEL"
else
  echo "[INFO] 检测到已安装哨兵,跳过依赖安装。" | tee -a "$LOG"
fi

# 3) 启动统一入口（hdt.cli:main）[10][13]
echo "[INFO] 启动: .venv/bin/python -m hdt.cli" | tee -a "$LOG"
.venv/bin/python -m hdt.cli