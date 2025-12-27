# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Humanities Doc Toolkit - Classifier engine (v0.1)

目标：
- 尽量复刻 Document Classifier v3.1 的“用户体验与可核对性”：
  * 扫描 -> PDF提取(按页范围/最小长度) -> AI分类(多key轮询) -> 置信度过滤 -> 安全移动(可选备份)
  * tqdm 进度条 + 静默/错误可见
  * 摘要报告（各文件夹计数、失败/低置信度统计）
  * 详细日志：JSON（全量字段） + TXT（移动清单） + 运行日志文件
- 与工具链规范对齐：共享 global.yaml 的 ai_services；classifier.yaml 放工具差异化配置
- 兼容旧 v3.1 配置结构 ai.services：由 loader 进行映射（ai -> ai_services）[2][1]
"""

import os
import re
import json
import time
import shutil
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tqdm import tqdm

from ..config.loader import load_merged_config  # loader 已提供 ai->ai_services 兼容映射
from ..common.logging_utils import setup_logger


PROJECT_NAME = "Humanities Doc Toolkit - Classifier"
VERSION = "0.1"


class ClassificationMode(Enum):
    PREDEFINED = "predefined"
    DYNAMIC = "dynamic"
    HYBRID = "hybrid"


class ProcessingStatus(Enum):
    PENDING = "待处理"
    SCANNING = "扫描中"
    EXTRACTING = "提取中"
    ANALYZING = "分析中"
    CLASSIFYING = "分类中"
    MOVING = "移动中"
    COMPLETED = "已完成"
    FAILED = "失败"
    SKIPPED = "跳过"
    LOW_CONFIDENCE = "置信度过低"


@dataclass
class ClassificationResult:
    folder: str
    confidence: float
    reasoning: str
    source: str  # ai / keyword / predefined


@dataclass
class DocumentInfo:
    path: str
    filename: str
    size_mb: float
    pages: int
    content_length: int
    status: ProcessingStatus
    classification: Optional[ClassificationResult] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    moved_to: Optional[str] = None
    planned_to: Optional[str] = None  # ✅ dry-run 计划目标路径


# ----------------------------
# Logging (对齐 v3.1：文件+控制台，且支持轮转配置) [2][1]
# ----------------------------
def _setup_run_logging(cfg: Dict[str, Any]) -> Tuple[logging.Logger, Path]:
    log_cfg = cfg.get("logging", {}) if isinstance(cfg.get("logging", {}), dict) else {}
    level = str(log_cfg.get("level", "INFO")).upper()

    # v3.1: paths.log_folder [1]
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    log_dir = Path(paths.get("log_folder", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"classifier_{ts}.log"

    logger = logging.getLogger("hdt-classifier")
    logger.setLevel(getattr(logging, level, logging.INFO))
    logger.handlers.clear()

    # 控制台输出（可关）
    enable_console = bool(log_cfg.get("enable_console_logging", True))
    fmt = log_cfg.get("log_format", "%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s")
    formatter = logging.Formatter(fmt)

    if enable_console:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    # 文件输出（可关）
    enable_file = bool(log_cfg.get("enable_file_logging", True) or log_cfg.get("enable_file_logging", False) or log_cfg.get("enable_file_logging", True))
    if enable_file:
        # 简化：不引入 RotatingFileHandler 也可；但 v3.1 配置提供轮转参数 [1]。
        # 这里直接写一个会话一个文件，配合 max_log_file_size_mb 可后续升级到 RotatingFileHandler。
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.info(f"📊 {PROJECT_NAME} v{VERSION} - 日志启动")
    return logger, log_dir


# ----------------------------
# PDF extract (复刻 v3.1：PyPDF2 页码范围、最小内容长度、元数据可选) [2][1]
# ----------------------------
class PDFExtractor:
    """
    提取策略（auto）：
    1) 优先 PyMuPDF（fitz）：速度快、兼容性通常更好（与你 renamer 一致）
    2) 回退 PyPDF2：保持 v3.1 的逐页 extract_text 逻辑与页范围控制[2]

    同时：完全屏蔽 PyPDF2 在某些PDF字体异常时产生的“unknown widths”噪声输出。
    """

    def __init__(self, cfg: Dict[str, Any], logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.pdf_cfg = cfg.get("pdf_processing", {}) if isinstance(cfg.get("pdf_processing", {}), dict) else {}

        self.page_range_start = int(self.pdf_cfg.get("page_range_start", 1))
        self.max_pages = int(self.pdf_cfg.get("max_pages_per_file", 20))
        self.min_len = int(self.pdf_cfg.get("min_content_length", 100))

        extractor_cfg = self.pdf_cfg.get("extractor", {}) if isinstance(self.pdf_cfg.get("extractor", {}), dict) else {}
        self.engine = str(extractor_cfg.get("engine", "auto")).lower()  # auto|pymupdf|pypdf2
        self.suppress_warnings = bool(extractor_cfg.get("suppress_warnings", True))
        self.silence_stderr = bool(extractor_cfg.get("silence_stderr", True))
        self.fallback = bool(extractor_cfg.get("fallback_to_other_engine", True))

    def extract(self, doc: DocumentInfo) -> Tuple[str, bool]:
        doc.status = ProcessingStatus.EXTRACTING

        # engine 决策
        engines: List[str]
        if self.engine == "pymupdf":
            engines = ["pymupdf"]
        elif self.engine == "pypdf2":
            engines = ["pypdf2"]
        else:
            engines = ["pymupdf", "pypdf2"]  # auto：优先 pymupdf，失败再回退

        last_err: Optional[str] = None

        for eng in engines:
            content = ""
            ok = False
            try:
                if eng == "pymupdf":
                    content, ok = self._extract_with_pymupdf(doc)
                else:
                    content, ok = self._extract_with_pypdf2(doc)

                if ok and len(content) >= self.min_len:
                    doc.content_length = len(content)
                    return content, True

                last_err = doc.error or f"{eng} 提取失败/内容过短"
                if not self.fallback:
                    break

            except Exception as e:
                last_err = f"{eng} 提取异常: {e}"
                doc.error = last_err
                if not self.fallback:
                    break

        doc.error = last_err or "提取失败"
        return "", False

    def _extract_with_pymupdf(self, doc: DocumentInfo) -> Tuple[str, bool]:
        try:
            import fitz  # PyMuPDF
        except Exception as e:
            doc.error = "缺少依赖 PyMuPDF（请 pip install PyMuPDF）"
            self.logger.error(f"缺少 PyMuPDF: {e}")
            return "", False

        try:
            parts: List[str] = []
            pdf = fitz.open(doc.path)
            total_pages = pdf.page_count
            doc.pages = total_pages

            start = max(1, self.page_range_start) - 1
            if self.max_pages == -1:
                end = total_pages
            else:
                end = min(start + max(1, self.max_pages), total_pages)

            self.logger.debug(f"{doc.filename}: [PyMuPDF] 提取页码 {start+1}-{end}/{total_pages}")

            for i in range(start, end):
                try:
                    t = pdf[i].get_text() or ""
                    if t.strip():
                        parts.append(t.strip())
                except Exception as pe:
                    self.logger.warning(f"{doc.filename}: [PyMuPDF] 第{i+1}页提取失败: {pe}")

            pdf.close()
            content = self._clean(" ".join(parts))
            if len(content) < self.min_len:
                doc.error = f"[PyMuPDF] 内容过短: {len(content)} < {self.min_len}"
                return "", False
            return content, True

        except Exception as e:
            doc.error = f"[PyMuPDF] 提取失败: {e}"
            self.logger.error(f"{doc.filename}: [PyMuPDF] 内容提取失败: {e}")
            return "", False

    def _extract_with_pypdf2(self, doc: DocumentInfo) -> Tuple[str, bool]:
        try:
            import PyPDF2  # 与 v3.1 一致[2]
        except Exception as e:
            doc.error = "缺少依赖 PyPDF2（请 pip install PyPDF2）"
            self.logger.error(f"缺少 PyPDF2: {e}")
            return "", False

        # 两层降噪：logging + stderr（v3.1 中也有“静默处理”理念，但这里仅对提取环节生效）[2]
        import logging as _logging
        from contextlib import redirect_stderr
        from io import StringIO

        if self.suppress_warnings:
            _logging.getLogger("PyPDF2").setLevel(_logging.ERROR)

        stderr_buf = StringIO()
        stderr_ctx = redirect_stderr(stderr_buf) if self.silence_stderr else None

        try:
            content_parts: List[str] = []

            if stderr_ctx:
                with stderr_ctx:
                    content_parts = self._pypdf2_read_pages(PyPDF2, doc)
            else:
                content_parts = self._pypdf2_read_pages(PyPDF2, doc)

            content = self._clean(" ".join(content_parts))
            doc.content_length = len(content)

            if len(content) < self.min_len:
                doc.error = f"[PyPDF2] 内容过短: {len(content)} < {self.min_len}"
                return "", False

            return content, True

        except Exception as e:
            doc.error = f"[PyPDF2] 提取失败: {e}"
            self.logger.error(f"{doc.filename}: [PyPDF2] 内容提取失败: {e}")
            return "", False

    def _pypdf2_read_pages(self, PyPDF2, doc: DocumentInfo) -> List[str]:
        parts: List[str] = []
        with open(doc.path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            total_pages = len(reader.pages)
            doc.pages = total_pages

            start = max(1, self.page_range_start) - 1
            if self.max_pages == -1:
                end = total_pages
            else:
                end = min(start + max(1, self.max_pages), total_pages)

            self.logger.debug(f"{doc.filename}: [PyPDF2] 提取页码 {start+1}-{end}/{total_pages}")

            for i in range(start, end):
                try:
                    t = reader.pages[i].extract_text() or ""
                    if t.strip():
                        parts.append(t.strip())
                except Exception as pe:
                    # v3.1 中对单页失败是 warning 并继续[2]
                    self.logger.warning(f"{doc.filename}: [PyPDF2] 第{i+1}页提取失败: {pe}")
        return parts

    def _clean(self, content: str) -> str:
        # v3.1 有清理空白与特殊符号的处理，这里先做轻量清洗[2]
        content = re.sub(r"\s+", " ", content)
        return content.strip()


# ----------------------------
# Folder selection (复刻 v3.1：预设/动态/混合) [2][1]
# ----------------------------
class FolderPolicy:
    def __init__(self, cfg: Dict[str, Any], logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.mode = ClassificationMode.HYBRID

    def set_mode(self, mode: ClassificationMode):
        self.mode = mode

    def scan_target_folders(self, output_path: str) -> List[str]:
        p = Path(output_path)
        p.mkdir(parents=True, exist_ok=True)
        return [x.name for x in p.iterdir() if x.is_dir()]

    def predefined_folders(self) -> List[str]:
        predefined = self.cfg.get("classification", {}).get("predefined_categories", {})
        return list(predefined.keys()) if isinstance(predefined, dict) else []

    def available(self, existing: List[str]) -> List[str]:
        if self.mode == ClassificationMode.PREDEFINED:
            return self.predefined_folders()
        if self.mode == ClassificationMode.DYNAMIC:
            return existing
        return sorted(set(existing) | set(self.predefined_folders()))


# ----------------------------
# Safe move + optional backup (复刻 v3.1 SafeFileManager + backup 开关) [2][1]
# ----------------------------
class SafeFileManager:
    def __init__(self, cfg: Dict[str, Any], logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        fm = cfg.get("file_management", {}) if isinstance(cfg.get("file_management", {}), dict) else {}
        self.safety = fm.get("safety", {}) if isinstance(fm.get("safety", {}), dict) else {}
        self.ops = fm.get("operations", {}) if isinstance(fm.get("operations", {}), dict) else {}

        # 注意：你旧 config 有 flase 拼写错误 [1]，这里做健壮解析
        self.enable_backup = self._as_bool(self.safety.get("enable_backup", False))
        self.backup_folder = Path(self.safety.get("backup_folder", "./backups"))
        self.backup_folder.mkdir(parents=True, exist_ok=True)

        self.conflict = str(self.ops.get("conflict_resolution", "rename")).lower()

    def _as_bool(self, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in {"true", "1", "yes", "y"}
        return bool(v)

    def _resolve_conflict(self, target: Path) -> Path:
        if self.conflict != "rename":
            return target
        base = target.with_suffix("")
        ext = target.suffix
        c = 1
        out = target
        while out.exists():
            out = Path(f"{base}_{c:03d}{ext}")
            c += 1
        return out

    def _backup(self, src: Path) -> Optional[Path]:
        if not self.enable_backup:
            return None
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dst = self.backup_folder / f"{ts}_{src.name}"
            shutil.copy2(src, dst)
            self.logger.info(f"备份创建: {dst}")
            return dst
        except Exception as e:
            self.logger.warning(f"备份失败: {src}: {e}")
            return None

    def move_to_exact(self, src: str, dst: str) -> Tuple[bool, str]:
        """
        严格按指定目标路径移动，用于 apply 阶段复现 dry-run 的 planned_to。
        注意：如果 dst 已存在，将视为失败（或你也可以在这里再做二次 resolve）。
        """
        try:
            src_p = Path(src)
            dst_p = Path(dst)
            dst_p.parent.mkdir(parents=True, exist_ok=True)

            if not src_p.exists():
                return False, f"源文件不存在: {src_p}"
            if dst_p.exists():
                return False, f"目标已存在(拒绝覆盖): {dst_p}"

            self._backup(src_p)
            shutil.move(str(src_p), str(dst_p))
            return True, str(dst_p)
        except Exception as e:
            return False, str(e)

    def move(self, src: str, folder: str, base_output: str) -> Tuple[bool, str]:
        try:
            src_p = Path(src)
            dst_dir = Path(base_output) / folder
            dst_dir.mkdir(parents=True, exist_ok=True)

            dst_p = dst_dir / src_p.name
            if dst_p.exists():
                dst_p = self._resolve_conflict(dst_p)

            self._backup(src_p)
            shutil.move(str(src_p), str(dst_p))
            return True, str(dst_p)
        except Exception as e:
            return False, str(e)

    def plan_target(self, src: str, folder: str, base_output: str) -> Tuple[bool, str]:
        """
        仅计算最终目标路径(含冲突改名策略)，不执行移动。
        用于 dry-run 阶段生成 planned_to，保证 apply 可复现。
        """
        try:
            src_p = Path(src)
            dst_dir = Path(base_output) / folder
            dst_dir.mkdir(parents=True, exist_ok=True)

            dst_p = dst_dir / src_p.name
            if dst_p.exists():
                dst_p = self._resolve_conflict(dst_p)

            return True, str(dst_p)
        except Exception as e:
            return False, str(e)

# ----------------------------
# AI Classifier (多key轮询；Claude/Gemini 特殊协议；其余 OpenAI-compatible)
# 与你 v3.1 的 multi-api 轮询思想一致 [2]，并与工具链的 ai_services 结构对齐 [1]
# ----------------------------
class AIClassifier:
    def __init__(self, cfg: Dict[str, Any], logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.timeout = int(cfg.get("ai_services", {}).get("api_request_timeout", 30))

        proc = cfg.get("processing", {}) if isinstance(cfg.get("processing", {}), dict) else {}
        conf = proc.get("confidence", {}) if isinstance(proc.get("confidence", {}), dict) else {}
        self.min_conf = float(conf.get("min_threshold", 0.3))
        self.skip_low = bool(conf.get("skip_low_confidence", True))
        err = proc.get("error_handling", {}) if isinstance(proc.get("error_handling", {}), dict) else {}
        self.retry_attempts = int(err.get("retry_attempts", 2))

        self.session = self._session()

        services = cfg.get("ai_services", {}).get("services", {})
        self.pool: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        for name, scfg in services.items():
            if not isinstance(scfg, dict) or not scfg.get("enabled", False):
                continue
            for k in scfg.get("api_keys", []):
                if isinstance(k, dict) and k.get("enabled", False) and str(k.get("key", "")).strip():
                    self.pool.append((name, scfg, k))

        self._idx = 0
        if not self.pool:
            raise ValueError("未配置任何可用AI服务（ai_services.services.*.enabled + api_keys[].enabled）")

        self.usage_stats: Dict[str, int] = {}

    def _session(self) -> requests.Session:
        s = requests.Session()
        retry = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20))
        return s

    def _next_api(self) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        item = self.pool[self._idx % len(self.pool)]
        self._idx += 1
        return item

    def classify(self, content: str, folders: List[str]) -> Optional[ClassificationResult]:
        preview = content[:4000] if len(content) > 4000 else content

        # ✅ 修复：prompt 中不能出现未转义的双引号（你当前 engine.py 的错误就在这里）[2]
        prompt = (
            "你是专门处理人文学科文献的分类专家。\n"
            "请从下列【可选分类文件夹】中选择最合适的一个，并给出置信度(0-1)与简短理由。\n\n"
            "【可选分类文件夹】:\n"
            + "\n".join([f"- {f}" for f in folders])
            + "\n\n"
            f"【文档内容摘要】:\n{preview}\n\n"
            "只输出JSON，不要输出其他文本：\n"
            '{"folder":"...", "confidence":0.0, "reasoning":"..."}'
        )

        for _ in range(max(1, self.retry_attempts)):
            service_name, scfg, kcfg = self._next_api()
            api_id = f"{service_name}_{kcfg.get('name','key')}"
            self.usage_stats[api_id] = self.usage_stats.get(api_id, 0) + 1

            try:
                data = self._call(service_name, scfg, kcfg, prompt)
                if not data:
                    continue

                folder_raw = str(data.get("folder", "")).strip()
                confidence = float(data.get("confidence", 0.0) or 0.0)
                reasoning = str(data.get("reasoning", "")).strip()

                matched = self._match_folder(folder_raw, folders)
                if not matched:
                    continue

                if self.skip_low and confidence < self.min_conf:
                    return ClassificationResult(matched, confidence, reasoning, "ai_low_confidence")
                return ClassificationResult(matched, confidence, reasoning, "ai")

            except Exception as e:
                self.logger.warning(f"AI调用失败({api_id}): {e}")
                continue

        return None

    def _match_folder(self, result: str, folders: List[str]) -> Optional[str]:
        r = result.strip().strip("\"'")
        rl = r.lower()
        for f in folders:
            if r == f or rl == f.lower():
                return f
        for f in folders:
            fl = f.lower()
            if rl in fl or fl in rl:
                return f
        return None

    def _safe_json(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        t = text.strip()
        if t.startswith("```"):
            t = re.sub(r"^```[a-zA-Z]*\n", "", t).strip()
            t = t.rstrip("`").strip()
        if not (t.startswith("{") and t.endswith("}")):
            m = re.search(r"\{.*\}", t, re.DOTALL)
            if m:
                t = m.group(0)
        try:
            obj = json.loads(t)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _call(self, service_name: str, scfg: Dict[str, Any], kcfg: Dict[str, Any], prompt: str) -> Optional[Dict[str, Any]]:
        key = str(kcfg.get("key", "")).strip()
        base_url = str(scfg.get("base_url", "")).rstrip("/")
        model = scfg.get("model", "")
        max_tokens = int(scfg.get("max_tokens", 300))
        temperature = float(scfg.get("temperature", 0.1))

        # Claude
        if service_name == "claude" or "anthropic" in base_url:
            url = base_url if base_url.endswith("/v1/messages") else base_url + "/v1/messages"
            headers = {
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload = {
                "model": model or "claude-3-haiku-20240307",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            r = self.session.post(url, headers=headers, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            text = ""
            if isinstance(data.get("content"), list) and data["content"]:
                text = data["content"][0].get("text", "")
            return self._safe_json(text)

        # Gemini
        if service_name == "gemini" or "generativelanguage.googleapis.com" in base_url:
            url = base_url
            if "key=" not in url:
                url = url + ("&" if "?" in url else "?") + f"key={key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
            }
            r = self.session.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            text = ""
            try:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                text = ""
            return self._safe_json(text)

        # OpenAI-compatible
        url = base_url if base_url.endswith("/chat/completions") else base_url + "/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "你是专业的人文学科文档分类专家。"},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        r = self.session.post(url, headers=headers, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        return self._safe_json(text)


# ----------------------------
# Report Generator (复刻 v3.1：摘要 + JSON详细日志 + TXT移动清单) [2]
# ----------------------------
class ReportGenerator:
    def __init__(self):
        self.start_time = datetime.now()
        self.docs: List[DocumentInfo] = []
        self.folder_stats: Dict[str, int] = {}
        self.error_summary: Dict[str, int] = {}
        self.api_usage_stats: Dict[str, int] = {}

    def add(self, doc: DocumentInfo):
        self.docs.append(doc)
        if doc.status == ProcessingStatus.COMPLETED and doc.classification:
            f = doc.classification.folder
            self.folder_stats[f] = self.folder_stats.get(f, 0) + 1
        if doc.error:
            et = doc.error.split(":")[0]
            self.error_summary[et] = self.error_summary.get(et, 0) + 1

    def set_api_usage(self, stats: Dict[str, int]):
        self.api_usage_stats = dict(stats or {})

    def summary_text(self) -> str:
        total = len(self.docs)
        ok = len([d for d in self.docs if d.status == ProcessingStatus.COMPLETED])
        fail = len([d for d in self.docs if d.status == ProcessingStatus.FAILED])
        low = len([d for d in self.docs if d.status == ProcessingStatus.LOW_CONFIDENCE])
        dur = datetime.now() - self.start_time
        avg = sum(d.processing_time for d in self.docs) / max(1, total)

        lines = []
        lines.append("=" * 70)
        lines.append(f"📊 {PROJECT_NAME} v{VERSION} 处理报告")
        lines.append("=" * 70)
        lines.append(f"🕒 用时: {dur}")
        lines.append(f"📁 总文档数: {total}")
        lines.append(f"✅ 完成(已生成 planned_to): {ok}")
        lines.append(f"❌ 失败: {fail}")
        lines.append(f"⚠️ 低置信度/未移动: {low}")
        lines.append(f"⏱️ 平均处理时间: {avg:.2f} 秒/文档")

        if self.folder_stats:
            lines.append("")
            lines.append("📂 分类分布:")
            for folder, count in sorted(self.folder_stats.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"   📂 {folder}: {count} 个文档")

        if self.api_usage_stats:
            lines.append("")
            lines.append("🤖 API使用统计:")
            for api_id, count in sorted(self.api_usage_stats.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"   🤖 {api_id}: {count} 次调用")

        if self.error_summary:
            lines.append("")
            lines.append("❌ 错误类型统计:")
            for et, c in sorted(self.error_summary.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"   ❌ {et}: {c} 次")

        lines.append("=" * 70)
        return "\n".join(lines)

    def save_json(self, log_dir: Path) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = log_dir / f"classification_details_{ts}.json"

        details = {
            "session_info": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
                "version": VERSION,
            },
            "statistics": {
                "total": len(self.docs),
                "successful": len([d for d in self.docs if d.status == ProcessingStatus.COMPLETED]),
                "failed": len([d for d in self.docs if d.status == ProcessingStatus.FAILED]),
                "low_confidence": len([d for d in self.docs if d.status == ProcessingStatus.LOW_CONFIDENCE]),
                "folder_stats": self.folder_stats,
                "api_usage_stats": self.api_usage_stats,
                "error_summary": self.error_summary,
            },
            "documents": [],
        }

        for d in self.docs:
            row = {
                "filename": d.filename,
                "path": d.path,
                "size_mb": d.size_mb,
                "pages": d.pages,
                "content_length": d.content_length,
                "status": d.status.value,
                "processing_time": d.processing_time,
                "planned_to": d.planned_to,   # ✅ 新增
                "moved_to": d.moved_to,
                "error": d.error,
            }
            if d.classification:
                row["classification"] = {
                    "folder": d.classification.folder,
                    "confidence": d.classification.confidence,
                    "reasoning": d.classification.reasoning,
                    "source": d.classification.source,
                }
            details["documents"].append(row)

        with out.open("w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)
        return out

    def save_txt_moves(self, log_dir: Path) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = log_dir / f"move_list_{ts}.txt"

        ok_docs = [d for d in self.docs if d.status == ProcessingStatus.COMPLETED and d.classification]
        low_docs = [d for d in self.docs if d.status == ProcessingStatus.LOW_CONFIDENCE]
        fail_docs = [d for d in self.docs if d.status == ProcessingStatus.FAILED]

        with out.open("w", encoding="utf-8") as f:
            f.write(f"{PROJECT_NAME} v{VERSION} 清单\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("说明:\n")
            f.write("- planned_to 存在 => 本次为 dry-run 计划移动(未执行移动)\n")
            f.write("- moved_to 存在   => 已执行移动(来自 apply 或旧版直接移动)\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"✅ 已完成分类/计划或已移动 ({len(ok_docs)}):\n")
            f.write("-" * 60 + "\n")
            for d in ok_docs:
                f.write(f"📄 {d.filename}\n")
                f.write(f"   📂 分类: {d.classification.folder}\n")
                f.write(f"   🎯 置信度: {d.classification.confidence:.2f}\n")
                f.write(f"   💭 理由: {d.classification.reasoning}\n")
                if d.planned_to:
                    f.write(f"   🧾 planned_to: {d.planned_to}\n")
                if d.moved_to:
                    f.write(f"   ✅ moved_to: {d.moved_to}\n")
                f.write("\n")

            if low_docs:
                f.write(f"\n⚠️ 低置信度/未纳入清单 ({len(low_docs)}):\n")
                f.write("-" * 60 + "\n")
                for d in low_docs:
                    f.write(f"📄 {d.filename} - {d.error}\n")

            if fail_docs:
                f.write(f"\n❌ 失败 ({len(fail_docs)}):\n")
                f.write("-" * 60 + "\n")
                for d in fail_docs:
                    f.write(f"📄 {d.filename} - {d.error}\n")

        return out


# ----------------------------
# Scanner (复刻 v3.1：按大小排序，小文件优先) [2]
# ----------------------------
def scan_pdfs(cfg: Dict[str, Any], input_folder: str) -> List[DocumentInfo]:
    pdf_cfg = cfg.get("pdf_processing", {}) if isinstance(cfg.get("pdf_processing", {}), dict) else {}
    max_mb = float(pdf_cfg.get("max_file_size_mb", 1000))
    docs: List[DocumentInfo] = []

    for root, _, files in os.walk(input_folder):
        for fn in files:
            if not fn.lower().endswith(".pdf"):
                continue
            fp = Path(root) / fn
            try:
                size_mb = fp.stat().st_size / (1024 * 1024)
                if max_mb > 0 and size_mb > max_mb:
                    continue
                docs.append(DocumentInfo(
                    path=str(fp),
                    filename=fn,
                    size_mb=size_mb,
                    pages=0,
                    content_length=0,
                    status=ProcessingStatus.PENDING
                ))
            except Exception:
                continue

    docs.sort(key=lambda d: d.size_mb)
    return docs


# ----------------------------
# Engine entry
# ----------------------------
class ClassifierEngine:
    def __init__(self, global_config: str, tool_config: str):
        self.cfg = load_merged_config(global_config, tool_config)
        self.logger, self.log_dir = _setup_run_logging(self.cfg)

    def _select_mode(self) -> ClassificationMode:
        default = self.cfg.get("classification", {}).get("modes", {}).get("default_mode", "hybrid")
        mapping = {"predefined": "1", "dynamic": "2", "hybrid": "3"}
        default_choice = mapping.get(str(default).lower(), "3")

        print("\n📂 选择分类模式:")
        print("1. 预设分类模式（配置预定义分类规则）")
        print("2. 动态检测模式（扫描目标目录现有文件夹）")
        print("3. 混合模式（预设 + 动态合并，推荐）")
        while True:
            c = input(f"请选择 (1-3, 默认={default_choice}): ").strip() or default_choice
            if c == "1":
                return ClassificationMode.PREDEFINED
            if c == "2":
                return ClassificationMode.DYNAMIC
            if c == "3":
                return ClassificationMode.HYBRID
            print("❌ 请输入 1-3")

    def _get_paths(self) -> Tuple[str, str]:
        paths = self.cfg.get("paths", {}) if isinstance(self.cfg.get("paths", {}), dict) else {}
        default_input = paths.get("input_folder", "")
        default_output = paths.get("output_folder", "")

        print("\n📁 路径配置:")
        while True:
            ip = input(f"PDF文件夹路径{f' (默认: {default_input})' if default_input else ''}: ").strip().strip('"') or default_input
            if ip and os.path.exists(ip):
                break
            print("❌ 路径不存在，请重新输入。")

        op = input(f"分类目标路径{f' (默认: {default_output})' if default_output else ' (回车=./classified)'}: ").strip().strip('"') or default_output or "./classified"
        return ip, op

    def display_header(self):
        from ..__about__ import __title__, __version__, __author__, __email__, __github__

        print("=" * 70)
        print(f"🎯 {__title__} - Classifier v{__version__}")
        print(f"👨‍💻 作者: {__author__} | {__email__}")
        print(f"🔗 {__github__}")
        print("-" * 70)
        print("📝 专门针对人文学科的智能文献分类与归档工具(可复核/低风险)")
        print("核心流程: 扫描 → 提取 → AI分类 → 置信度过滤 → 生成清单/日志 [12]")
        print("安全策略:")
        print("  1) 默认 dry-run：只生成 JSON/TXT 清单，不移动文件（更安全）")
        print("  2) 需要移动时：运行 hdt-classifier-apply --log <classification_details_*.json>")
        print("配置提示: 复制 configs/*.example.yaml 到根目录生成 global.yaml/classifier.yaml [2]")
        print("安全提示: 请勿提交真实 API Key；仓库只放 example 配置 [2]")
        print("=" * 70)

    def _get_services_cfg(self) -> Dict[str, Any]:
        """
        获取服务配置：工具链规范是 ai_services.services；
        兼容 v3.1 的 ai.services 结构 [2][1]。
        """
        if isinstance(self.cfg.get("ai_services"), dict) and isinstance(self.cfg["ai_services"].get("services"), dict):
            return self.cfg["ai_services"]["services"]
        if isinstance(self.cfg.get("ai"), dict) and isinstance(self.cfg["ai"].get("services"), dict):
            return self.cfg["ai"]["services"]
        return {}

    def display_simplified_api_status(self) -> bool:
        """
        显示可用AI服务概览（复刻 v3.1 的简化状态显示）[2]。
        """
        services = self._get_services_cfg()
        if not services:
            print("❌ 配置中未找到 AI 服务配置（ai_services.services）")
            return False

        available = []
        print("\n🤖 可用AI服务:")
        for name, scfg in services.items():
            if not isinstance(scfg, dict) or not scfg.get("enabled", False):
                continue

            keys = scfg.get("api_keys", []) if isinstance(scfg.get("api_keys", []), list) else []
            enabled_keys = [k for k in keys if isinstance(k, dict) and k.get("enabled", False) and str(k.get("key", "")).strip()]
            if not enabled_keys:
                continue

            model = scfg.get("model", "default")
            print(f"   ✅ {name.upper():<10} 模型: {model:<18} 密钥: {len(enabled_keys)}")
            available.append(name)

        if not available:
            print("❌ 没有可用的AI服务（请检查 enabled 与 api_keys）")
            return False

        return True

    def apply_moves_from_log(self, log_path: str) -> int:
        """
        根据 dry-run 生成的 classification_details_*.json 执行移动。
        严格按 planned_to 落盘（可复核/可复现）。
        """
        p = Path(log_path)
        if not p.exists():
            print(f"❌ 日志不存在: {p}")
            return 1

        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"❌ 日志JSON解析失败: {e}")
            return 1

        docs = data.get("documents", [])
        if not isinstance(docs, list) or not docs:
            print("❌ 日志中没有 documents")
            return 1

        mover = SafeFileManager(self.cfg, self.logger)

        ok, fail, skipped = 0, 0, 0
        for item in docs:
            try:
                status = item.get("status")
                src = item.get("path")
                planned_to = item.get("planned_to")

                if status != ProcessingStatus.COMPLETED.value:
                    skipped += 1
                    continue
                if not (src and planned_to):
                    skipped += 1
                    continue
                if not Path(src).exists():
                    fail += 1
                    self.logger.warning(f"源文件不存在: {src}")
                    continue

                ok2, msg = mover.move_to_exact(src, planned_to)
                if ok2:
                    ok += 1
                else:
                    fail += 1
                    self.logger.warning(f"移动失败: {src} -> {planned_to} | {msg}")

            except Exception as e:
                fail += 1
                self.logger.warning(f"移动异常: {e}")

        print(f"✅ apply 完成: 成功移动={ok}, 失败={fail}, 跳过={skipped}")
        return 0 if fail == 0 else 1

    def display_ai_service_selection(self) -> Optional[str]:
        """
        选择使用哪个服务：
        - 选择单一服务：将其它服务 enabled=False
        - 或“全部服务自动轮询”：保持原状
        返回：选中的服务名（单一）或 None（表示全部轮询）
        （复刻 v3.1 的 service selection 交互）[2]。
        """
        services = self._get_services_cfg()

        available_services = []
        for name, scfg in services.items():
            if not isinstance(scfg, dict) or not scfg.get("enabled", False):
                continue
            keys = scfg.get("api_keys", []) if isinstance(scfg.get("api_keys", []), list) else []
            enabled_keys = [k for k in keys if isinstance(k, dict) and k.get("enabled", False) and str(k.get("key", "")).strip()]
            if enabled_keys:
                available_services.append({
                    "name": name,
                    "model": scfg.get("model", "default"),
                    "key_count": len(enabled_keys),
                })

        if not available_services:
            print("❌ 没有可用AI服务")
            return None

        # 只有一个服务时默认直接用它
        if len(available_services) == 1:
            s = available_services[0]["name"]
            print(f"🤖 将使用: {s.upper()}")
            return s

        print("\n🤖 选择AI服务:")
        for i, s in enumerate(available_services, 1):
            print(f"{i}. {s['name'].upper()} - 模型: {s['model']} ({s['key_count']}个密钥)")
        print(f"{len(available_services) + 1}. 使用全部服务（自动轮询，推荐）")

        default_choice = str(len(available_services) + 1)
        while True:
            choice = input(f"\n请选择 (1-{len(available_services)+1}, 默认={default_choice}): ").strip() or default_choice

            if choice == str(len(available_services) + 1):
                print("✅ 将使用所有可用AI服务（自动轮询）")
                return None

            try:
                idx = int(choice)
                if 1 <= idx <= len(available_services):
                    selected = available_services[idx - 1]["name"]
                    for name in services.keys():
                        services[name]["enabled"] = (name == selected)
                    print(f"✅ 已选择: {selected.upper()}（其它服务已临时禁用）")
                    return selected
            except ValueError:
                pass

            print("❌ 请输入有效选项")

    def _open_file(self, p: Path):
        """跨平台打开文件(用于打开 dry-run 生成的 JSON 清单)"""
        try:
            import subprocess, platform
            sysname = platform.system()
            if sysname == "Windows":
                os.startfile(str(p))  # type: ignore[attr-defined]
            elif sysname == "Darwin":
                subprocess.run(["open", str(p)], check=False)
            else:
                subprocess.run(["xdg-open", str(p)], check=False)
        except Exception as e:
            self.logger.warning(f"打开文件失败: {p} | {e}")

    def run_interactive(self, dry_run: bool = True) -> int:
        self.display_header()

        if not self.display_simplified_api_status():
            return 1
        self.display_ai_service_selection()

        mode = self._select_mode()
        input_path, output_path = self._get_paths()
        # 2) 扫描
        docs = scan_pdfs(self.cfg, input_path)
        if not docs:
            print("❌ 未找到可处理的PDF")
            return 1

        policy = FolderPolicy(self.cfg, self.logger)
        policy.set_mode(mode)
        existing = policy.scan_target_folders(output_path)
        folders = policy.available(existing)
        if not folders:
            print("❌ 没有可用分类文件夹：请在目标目录创建文件夹或配置 predefined_categories")
            return 1

        extractor = PDFExtractor(self.cfg, self.logger)
        mover = SafeFileManager(self.cfg, self.logger)
        ai = AIClassifier(self.cfg, self.logger)
        report = ReportGenerator()

        # 3) 多线程设置（沿用 v3.1 processing.multithreading.max_workers）[1][2]
        proc = self.cfg.get("processing", {}) if isinstance(self.cfg.get("processing", {}), dict) else {}
        mt = proc.get("multithreading", {}) if isinstance(proc.get("multithreading", {}), dict) else {}
        enable_mt = bool(mt.get("enabled", True))
        max_workers = int(mt.get("max_workers", 8))
        if not enable_mt:
            max_workers = 1

        print(f"\n🚀 开始处理 {len(docs)} 个文档 (线程数: {max_workers}) ...")
        print(f"📂 可选分类文件夹数: {len(folders)}")
        start = time.time()

        def handle_one(d: DocumentInfo) -> DocumentInfo:
            t0 = time.time()
            try:
                # 1) 提取
                d.status = ProcessingStatus.EXTRACTING
                content, ok = extractor.extract(d)
                if not ok:
                    d.status = ProcessingStatus.FAILED
                    return d

                # 2) 分类
                d.status = ProcessingStatus.CLASSIFYING
                res = ai.classify(content, folders)
                if not res:
                    d.status = ProcessingStatus.LOW_CONFIDENCE
                    d.error = "AI未返回可匹配分类或置信度过低"
                    return d

                d.classification = res

                if res.source == "ai_low_confidence":
                    d.status = ProcessingStatus.LOW_CONFIDENCE
                    d.error = f"置信度 {res.confidence:.2f} 低于阈值，保持原位"
                    return d

                # 3) 只支持 dry-run 生成 planned_to（推荐的安全模式）[2]
                if dry_run:
                    ok2, planned = mover.plan_target(d.path, res.folder, output_path)
                    if ok2:
                        d.status = ProcessingStatus.COMPLETED
                        d.planned_to = planned
                        d.moved_to = None
                    else:
                        d.status = ProcessingStatus.FAILED
                        d.error = planned
                    return d

                # 如果你真的想支持“直接移动”，请不要在这里做；
                # 统一走 apply_moves_from_log()，保证可复核、可回滚。
                d.status = ProcessingStatus.SKIPPED
                d.error = "当前交互模式不支持直接移动；请使用 apply（从 JSON 清单执行移动）"
                return d

            except Exception as e:
                d.status = ProcessingStatus.FAILED
                d.error = str(e)
                return d
            finally:
                d.processing_time = time.time() - t0

        def _plan_summary(docs_list: List[DocumentInfo]) -> Dict[str, int]:
            from collections import defaultdict
            from pathlib import Path as _Path
            dd = defaultdict(int)
            for d in docs_list:
                if d.status == ProcessingStatus.COMPLETED and d.planned_to:
                    dd[_Path(d.planned_to).parent.name] += 1
            return dict(dd)

        # 线程池执行
        results: List[DocumentInfo] = []
        if max_workers == 1:
            for d in tqdm(docs, desc="处理进度"):
                r = handle_one(d)
                report.add(r)
                results.append(r)
        else:
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ClassifierWorker") as ex:
                futs = {ex.submit(handle_one, d): d for d in docs}
                for fut in tqdm(as_completed(futs), total=len(futs), desc="处理进度"):
                    r = fut.result()
                    report.add(r)
                    results.append(r)

        dur = time.time() - start
        completed = [d for d in results if d.status == ProcessingStatus.COMPLETED]

        # 写入 API 使用统计与日志文件
        report.set_api_usage(ai.usage_stats)
        json_path = report.save_json(self.log_dir)
        txt_path = report.save_txt_moves(self.log_dir)

        print("\n" + report.summary_text())
        print(f"\n🧾 JSON 详情: {json_path}")
        print(f"📄 TXT 清单: {txt_path}")

        if dry_run:
            print("\n🧾 本次为 dry-run（未移动文件）。将自动打开 JSON 供你核对。")
            self._open_file(Path(json_path))

            while True:
                print("\n下一步：")
                print("  y) 我已核对，立即按该 JSON 执行移动")
                print('  a) 在终端显示“计划移动摘要”(每个分类文件夹多少个)')
                print("  n) 退出（仅保留清单，不移动）")
                ans = input("请选择 (y/a/n) [默认 n]: ").strip().lower() or "n"

                if ans == "a":
                    summ = _plan_summary(completed)
                    if not summ:
                        print("（无 planned_to 记录：可能全部低置信度/失败）")
                    else:
                        print("\n计划移动摘要：")
                        for folder, cnt in sorted(summ.items(), key=lambda x: x[1], reverse=True):
                            print(f"  📂 {folder}: {cnt} 个")
                    continue

                if ans == "y":
                    # 复用 apply：按本次 json 直接执行，不重跑AI
                    code = self.apply_moves_from_log(str(json_path))
                    return code

                if ans == "n":
                    break

                print("❌ 无效输入，请重试。")

        self.logger.info(f"任务完成: total={len(docs)} seconds={dur:.1f} json={json_path} txt={txt_path}")
        return 0