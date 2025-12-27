from __future__ import annotations
from tqdm import tqdm

def make_progress(total: int, desc: str):
    return tqdm(
        total=total,
        desc=desc,
        leave=True,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )

def progress_log(progress: tqdm, msg: str):
    # 不破坏进度条的输出方式
    try:
        progress.write(msg)
    except Exception:
        print(msg)