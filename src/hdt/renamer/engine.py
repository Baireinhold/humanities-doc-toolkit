# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Renamer Engine (Humanities Doc Toolkit)

迁移与增强说明：
- 核心流程来自你稳定的 DocumentRenamerEnhanced：扫描PDF -> 抽取文本 -> AI提取元信息(JSON) -> 按模板生成文件名 -> 分类移动/原地重命名 [3]。
- 保留中文交互与美化输出（colorama）、多线程+多key轮询（APIKeyManager队列）[3]。
- 扩展AI提供商兼容：Claude / OpenAI / Gemini / DeepSeek / Kimi / GLM。
- 仍支持“分类模式(category)”与“原地重命名(rename_only)”两种模式，并创建 处理成功/处理失败/问题文件 子目录 [3]。
"""

import os
import sys
import re
import json
import time
import yaml
import gc
import shutil
import queue
import logging
import argparse
import threading
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tqdm import tqdm
import fitz  # PyMuPDF
from colorama import init, Fore, Style
import psutil

init(autoreset=True)

# ---------------------------
# API Key Manager (稳定路径)
# ---------------------------
class APIKeyManager:
    """API密钥管理器 - 支持多线程轮询（队列取用/归还）[3]"""

    def __init__(self, service_config: Dict[str, Any]):
        self.service_config = service_config
        self.available_keys = [k for k in service_config.get("api_keys", []) if k.get("enabled", False)]
        self.key_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.key_stats = defaultdict(lambda: {"calls": 0, "errors": 0, "last_used": None})
        self.token_stats = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        for k in self.available_keys:
            self.key_queue.put(k)

    def get_key(self) -> Optional[Dict[str, Any]]:
        try:
            return self.key_queue.get_nowait()
        except queue.Empty:
            return None

    def return_key(self, key_config: Dict[str, Any], success: bool = True):
        key_name = key_config.get("name", "unknown")
        self.key_stats[key_name]["calls"] += 1
        self.key_stats[key_name]["last_used"] = datetime.now()
        if not success:
            self.key_stats[key_name]["errors"] += 1
        self.key_queue.put(key_config)

    def record_token_usage(self, key_name: str, input_tokens: int, output_tokens: int, total_tokens: int):
        self.token_stats[key_name]["input_tokens"] += input_tokens or 0
        self.token_stats[key_name]["output_tokens"] += output_tokens or 0
        self.token_stats[key_name]["total_tokens"] += total_tokens or 0

    def get_stats(self) -> Dict[str, Any]:
        return dict(self.key_stats)

    def get_token_stats(self) -> Dict[str, Any]:
        return dict(self.token_stats)


# ---------------------------
# Renamer Engine
# ---------------------------
class DocumentRenamerEnhanced:
    """智能PDF重命名工具（迁移增强版）[3]"""

    def __init__(self, global_config_path: str = "global.yaml", renamer_config_path: str = "renamer.yaml"):
        self.version = "0.1"
        self.author = "Baireinhold"
        self.global_config_path = global_config_path
        self.renamer_config_path = renamer_config_path

        self.config = self.load_and_merge_config()
        self.logger = self.setup_logging()

        self.stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "problem_docs": 0,
            "start_time": None,
            "api_calls": 0,
            "thread_stats": defaultdict(int),
        }

        self.duplicate_tracker = defaultdict(int)
        self.api_managers: Dict[str, APIKeyManager] = {}
        self.processing_log: List[Dict[str, Any]] = []

        self.session = self._build_requests_session()

        self.ensure_directories()
        self.show_banner()
        self.show_config_info()

    # ---------------------------
    # Config & Logging
    # ---------------------------
    def load_yaml(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"配置文件格式必须为dict: {path}")
        return data

    def deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(base)
        for k, v in override.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = self.deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    def load_and_merge_config(self) -> Dict[str, Any]:
        g = self.load_yaml(self.global_config_path)
        r = self.load_yaml(self.renamer_config_path)
        merged = self.deep_merge(g, r)

        # 最小字段校验：ai_services.services 必须存在 [2]
        ai_services = merged.get("ai_services", {})
        if not isinstance(ai_services, dict) or not isinstance(ai_services.get("services", {}), dict):
            raise ValueError("global.yaml 缺少 ai_services.services 配置 [2]")

        return merged

    def setup_logging(self) -> logging.Logger:
        log_level = "INFO"
        logger = logging.getLogger("hdt-renamer")
        logger.setLevel(getattr(logging, log_level, logging.INFO))
        logger.handlers.clear()
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        return logger

    def _build_requests_session(self) -> requests.Session:
        s = requests.Session()
        retry = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        return s

    # ---------------------------
    # UI / Banner
    # ---------------------------
    def show_banner(self):
        from ..__about__ import __title__, __version__, __author__, __email__, __github__
        banner = f"""
    {Fore.CYAN}{'='*72}
    {Fore.YELLOW}📚 {__title__} - Renamer v{__version__}
    {Fore.GREEN}🤖 AI辅助 PDF 重命名/标准化 (人文学科友好命名规范)
    {Fore.BLUE}👨‍💻 作者: {__author__} | {__email__}
    {Fore.MAGENTA}🔗 {__github__}
    {Fore.WHITE}提示:
    - 默认会读取 global.yaml + renamer.yaml (共享 ai_services; 工具差异配置) [2]
    - 请勿提交真实 API Key；仅提交 example 配置 [2]
    {'='*72}{Style.RESET_ALL}
    """
        print(banner)

    def show_config_info(self):
        try:
            perf = self.config.get("processing", {}).get("performance", {})
            max_workers = perf.get("max_workers", 8)
            print("🔧 配置信息:")
            print(f"📋 最大工作线程: {max_workers}")

            active_service = self.config.get("ai_services", {}).get("active_service", "deepseek")
            services = self.config.get("ai_services", {}).get("services", {})
            if active_service in services:
                api_keys = services[active_service].get("api_keys", [])
                enabled_count = len([k for k in api_keys if k.get("enabled", False)])
                print(f"🤖 当前AI服务: {active_service}")
                print(f"🔑 可用API密钥: {enabled_count}/{len(api_keys)}")
            print("✅ 配置加载完成")
        except Exception as e:
            print(f"⚠️ 配置信息显示失败: {e}")

    # ---------------------------
    # Directories
    # ---------------------------
    def ensure_directories(self):
        # Renamer 本身只需要 logs/output/backup；你在旧版 config.yaml 中有 directories.paths [2]
        paths = self.config.get("directories", {}).get("paths", {})
        for _, p in paths.items():
            try:
                if p:
                    Path(p).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    # ---------------------------
    # Interactive choices (中文)
    # ---------------------------
    def select_ai_service(self) -> str:
        services = self.config["ai_services"]["services"]
        available = []

        print(f"\n{Fore.CYAN}🤖 可用AI服务:")
        for i, (name, cfg) in enumerate(services.items(), 1):
            api_keys = cfg.get("api_keys", [])
            enabled_keys = [k for k in api_keys if k.get("enabled", False)]
            enabled_count = len(enabled_keys)

            if cfg.get("enabled", False) and enabled_count > 0:
                status = f"{Fore.GREEN}✅ 可用"
                available.append(name)
                self.api_managers[name] = APIKeyManager(cfg)  # 使用稳定的队列轮询 [3]
            else:
                status = f"{Fore.RED}❌ 不可用"

            emoji = {
                "deepseek": "🔍",
                "openai": "🧠",
                "gemini": "🎯",
                "claude": "💡",
                "kimi": "🌙",
                "glm": "🧬",
            }.get(name, "🤖")

            print(f"  {i}. {emoji} {name} - {status} ({enabled_count}个密钥)")

        if not available:
            print(f"{Fore.RED}❌ 没有可用AI服务，请检查 global.yaml 的 ai_services.services 配置 [2]")
            sys.exit(1)

        # 默认使用 active_service
        default_service = self.config.get("ai_services", {}).get("active_service", available[0])
        if default_service in available:
            default_idx = list(services.keys()).index(default_service) + 1
        else:
            default_idx = 1

        while True:
            choice = input(f"\n{Fore.YELLOW}🎯 请选择AI服务 (1-{len(services)}) [默认{default_idx}]: {Style.RESET_ALL}").strip()
            if not choice:
                picked = list(services.keys())[default_idx - 1]
                if picked in available:
                    print(f"{Fore.GREEN}✅ 已选择: {picked}{Style.RESET_ALL}")
                    return picked
                picked = available[0]
                print(f"{Fore.GREEN}✅ 已选择: {picked}{Style.RESET_ALL}")
                return picked
            try:
                idx = int(choice) - 1
                picked = list(services.keys())[idx]
                if picked in available:
                    print(f"{Fore.GREEN}✅ 已选择: {picked}{Style.RESET_ALL}")
                    return picked
                print(f"{Fore.RED}❌ 该服务不可用，请重新选择。{Style.RESET_ALL}")
            except Exception:
                print(f"{Fore.RED}❌ 无效输入，请输入数字。{Style.RESET_ALL}")

    def get_processing_pages(self) -> Tuple[int, int]:
        extraction = self.config.get("processing", {}).get("extraction", {})
        default_start = int(extraction.get("start_page", 1))
        default_end = int(extraction.get("end_page", 10))

        print(f"\n{Fore.CYAN}📄 页面范围设置:")
        print(f"📋 默认范围: {default_start}-{default_end} 页")
        s = input(f"{Fore.YELLOW}📝 输入起始页(回车默认{default_start}): {Style.RESET_ALL}").strip()
        e = input(f"{Fore.YELLOW}📝 输入结束页(回车默认{default_end}): {Style.RESET_ALL}").strip()
        try:
            start_page = int(s) if s else default_start
            end_page = int(e) if e else default_end
            if start_page < 1:
                start_page = 1
            if end_page < start_page:
                end_page = start_page
            return start_page, end_page
        except Exception:
            print(f"{Fore.YELLOW}⚠️ 输入无效，使用默认范围。{Style.RESET_ALL}")
            return default_start, default_end

    def get_output_directory(self) -> str:
        default_output = self.config.get("directories", {}).get("paths", {}).get("output", "output")
        print(f"\n{Fore.CYAN}📂 输出目录配置:")
        print(f"📁 默认输出目录: {default_output}")
        user = input(f"{Fore.YELLOW}🎯 输入输出目录(回车默认): {Style.RESET_ALL}").strip()
        out = user or default_output
        Path(out).mkdir(parents=True, exist_ok=True)
        return str(Path(out).absolute())

    def get_processing_mode(self) -> str:
        print(f"\n{Fore.CYAN}🔧 处理模式:")
        print("  1. 📂 分类模式：创建子文件夹（处理成功/处理失败/问题文件）[3]")
        print("  2. 🔄 原地重命名：保持原位置直接重命名[3]")
        while True:
            c = input(f"{Fore.YELLOW}请选择 (1/2) [默认1]: {Style.RESET_ALL}").strip() or "1"
            if c == "1":
                return "category"
            if c == "2":
                return "rename_only"
            print(f"{Fore.RED}❌ 请输入 1 或 2{Style.RESET_ALL}")

    # ---------------------------
    # Output folders
    # ---------------------------
    def setup_output_folders(self, output_dir: str) -> Dict[str, str]:
        subfolder_names = self.config.get("output_management", {}).get("subfolder_names", {
            "success": "处理成功",
            "failed": "处理失败",
            "problem": "问题文件"
        })
        folders = {
            "success": os.path.join(output_dir, subfolder_names["success"]),
            "failed": os.path.join(output_dir, subfolder_names["failed"]),
            "problem": os.path.join(output_dir, subfolder_names["problem"]),
        }
        for k, p in folders.items():
            try:
                Path(p).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.error(f"创建文件夹失败 {k}: {p} - {e}")
                folders[k] = output_dir
        return folders

    def move_file_to_category(self, source_path: str, target_folder: str, final_filename: str) -> str:
        target_path = os.path.join(target_folder, final_filename)
        counter = 1
        while os.path.exists(target_path):
            stem, ext = os.path.splitext(final_filename)
            target_path = os.path.join(target_folder, f"{stem}_{counter}{ext}")
            counter += 1

        # 同目录 rename，不同目录 move
        src_dir = os.path.dirname(source_path)
        dst_dir = os.path.dirname(target_path)
        if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
            os.rename(source_path, target_path)
        else:
            shutil.move(source_path, target_path)
        return target_path

    # ---------------------------
    # File scanning & PDF extract
    # ---------------------------
    def find_pdf_files(self, directory: str) -> List[str]:
        allowed_ext = self.config.get("directories", {}).get("file_filtering", {}).get("allowed_extensions", [".pdf"])
        pdfs = []
        seen = set()
        for root, _, files in os.walk(directory):
            for fn in files:
                fn_low = fn.lower()
                if any(fn_low.endswith(ext.lower()) for ext in allowed_ext):
                    p = os.path.join(root, fn)
                    key = os.path.normpath(p).lower()
                    if key not in seen:
                        seen.add(key)
                        pdfs.append(p)
        print(f"{Fore.GREEN}✅ 发现 {len(pdfs)} 个PDF文件{Style.RESET_ALL}")
        return pdfs

    def extract_text_from_pdf(self, file_path: str, start_page: int, end_page: int) -> str:
        try:
            doc = fitz.open(file_path)
            parts = []
            actual_end = min(end_page, doc.page_count)
            for page_num in range(start_page - 1, actual_end):
                text = doc[page_num].get_text()
                if text and text.strip():
                    parts.append(text.strip())
            doc.close()
            extracted = "\n".join(parts)

            if not extracted or len(extracted.strip()) < 10:
                return "NEEDS_OCR_PROCESSING"

            max_len = int(self.config.get("processing", {}).get("extraction", {}).get("max_text_length", 10000))
            if len(extracted) > max_len:
                extracted = extracted[:max_len] + "..."
            return extracted
        except Exception as e:
            self.logger.error(f"PDF提取失败: {file_path}: {e}")
            return "NEEDS_OCR_PROCESSING"

    # ---------------------------
    # AI calling: multi-provider
    # ---------------------------
    def _parse_usage_tokens(self, response_json: Dict[str, Any]) -> Tuple[int, int, int]:
        usage = response_json.get("usage") if isinstance(response_json, dict) else None
        if not usage or not isinstance(usage, dict):
            return 0, 0, 0
        in_t = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0
        out_t = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0
        total = usage.get("total_tokens", 0) or (in_t + out_t)
        return int(in_t), int(out_t), int(total)

    def call_ai_service(self, text: str, service_name: str) -> Optional[Dict[str, Any]]:
        api_manager = self.api_managers.get(service_name)
        if not api_manager:
            self.logger.error(f"未找到 {service_name} 的API管理器")
            return None

        # 获取 key(短暂重试)
        api_key_cfg = None
        for _ in range(3):
            api_key_cfg = api_manager.get_key()
            if api_key_cfg:
                break
            time.sleep(0.1)
        if not api_key_cfg:
            self.logger.error(f"{service_name} 没有可用API密钥")
            return None

        service_cfg = self.config["ai_services"]["services"][service_name]
        base_url = service_cfg.get("base_url", "").strip()
        model = service_cfg.get("model", "")
        timeout = int(self.config.get("ai_services", {}).get("api_request_timeout", 30))

        # ---------- 语言策略(来自 renamer.yaml 的 ai_text_policy.language) ----------
        lang_cfg = self.config.get("ai_text_policy", {}).get("language", {}) if isinstance(self.config.get("ai_text_policy", {}), dict) else {}
        mode = str(lang_cfg.get("mode", "keep_original")).strip().lower()  # keep_original | translate
        target = str(lang_cfg.get("target", "")).strip()  # zh/en/ja...
        fields = lang_cfg.get("fields", ["title", "publisher", "journal"])
        if not isinstance(fields, list) or not fields:
            fields = ["title", "publisher", "journal"]

        allowed_fields = {"title", "publisher", "journal"}
        fields = [f for f in fields if isinstance(f, str) and f.strip().lower() in allowed_fields]
        fields = [f.strip().lower() for f in fields] or ["title", "publisher", "journal"]

        if mode not in {"keep_original", "translate"}:
            mode = "keep_original"
        if mode == "translate" and not target:
            # translate 但未提供 target，回退为保持原文
            mode = "keep_original"

        if mode == "keep_original":
            lang_instruction = (
                "语言要求：请保持以下字段的原始语言，不要翻译："
                f"{', '.join(fields)}。"
                "author 字段永远不要翻译，保持原样。"
            )
        else:
            # translate
            lang_instruction = (
                f"语言要求：请将以下字段翻译为 {target} 语言：{', '.join(fields)}。"
                "author 字段永远不要翻译，保持原样。"
                "如果原字段已经是目标语言，可保持不变。"
            )

        # 输入截断（字符数，不是 token 上限）；输出 token 上限来自 global.yaml max_tokens [2]
        max_chars = int(self.config.get("processing", {}).get("extraction", {}).get("prompt_text_max_chars", 2000))
        preview = (text or "")[:max_chars]

        # ---------- Prompt（保持严格 JSON 与类型约束） ----------
        prompt = f"""
请分析以下PDF文档内容，提取关键信息。请严格按照JSON格式回复，包含字段：
- title: 文档标题（不要包含下划线_，可用破折号-）
- author: 作者姓名（不确定则空字符串；注意：author 不要翻译）
- year: 单一4位出版年份（不确定则空字符串）
- type: 文档类型，必须是 book / paper / others / unknown
- journal: 期刊名（paper需要，其他为空字符串）
- publisher: 出版社（book需要，其他为空字符串）

重要要求：
1) type 必须四选一，默认倾向 others
2) 不要输出除JSON之外的任何文字
3) 无法确定字段用 ""（空字符串），不要写“未知/不详”
4) {lang_instruction}

文档内容：
{preview}
""".strip()

        success = False
        result: Optional[Dict[str, Any]] = None

        try:
            # 1) Claude
            if service_name.lower() == "claude" or "anthropic" in base_url:
                headers = {
                    "x-api-key": api_key_cfg["key"],
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
                payload = {
                    "model": model or "claude-3-haiku-20240307",
                    "max_tokens": int(service_cfg.get("max_tokens", 500)),
                    "temperature": float(service_cfg.get("temperature", 0.1)),
                    "messages": [{"role": "user", "content": prompt}],
                }
                resp = self.session.post(base_url, headers=headers, json=payload, timeout=timeout)
                if resp.status_code != 200:
                    raise RuntimeError(f"Claude调用失败: {resp.status_code} {resp.text[:200]}")
                data = resp.json()
                content = ""
                if isinstance(data.get("content"), list) and data["content"]:
                    content = data["content"][0].get("text", "").strip()
                result = self._safe_json_from_model_text(content)

            # 2) Gemini
            elif service_name.lower() == "gemini" or "generativelanguage.googleapis.com" in base_url:
                url = base_url
                if "key=" not in url:
                    joiner = "&" if "?" in url else "?"
                    url = f"{url}{joiner}key={api_key_cfg['key']}"
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": float(service_cfg.get("temperature", 0.1)),
                        "maxOutputTokens": int(service_cfg.get("max_tokens", 500)),
                    },
                }
                resp = self.session.post(url, json=payload, timeout=timeout)
                if resp.status_code != 200:
                    raise RuntimeError(f"Gemini调用失败: {resp.status_code} {resp.text[:200]}")
                data = resp.json()
                content = ""
                try:
                    content = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                except Exception:
                    content = ""
                result = self._safe_json_from_model_text(content)

            # 3) OpenAI-compatible
            else:
                headers = {"Authorization": f"Bearer {api_key_cfg['key']}", "Content-Type": "application/json"}
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": int(service_cfg.get("max_tokens", 2000)),
                    "temperature": float(service_cfg.get("temperature", 0.1)),
                }
                resp = self.session.post(base_url, headers=headers, json=payload, timeout=timeout)
                if resp.status_code != 200:
                    raise RuntimeError(f"{service_name}调用失败: {resp.status_code} {resp.text[:200]}")
                data = resp.json()

                # token统计（保留你原逻辑）[17]
                in_t, out_t, total = self._parse_usage_tokens(data)
                api_manager.record_token_usage(api_key_cfg.get("name", "unknown"), in_t, out_t, total)

                content = data["choices"][0]["message"]["content"].strip()
                result = self._safe_json_from_model_text(content)

            if result:
                result = self._normalize_ai_result(result)
                success = True
                self.stats["api_calls"] += 1

        except Exception as e:
            self.logger.error(f"AI调用异常({service_name}): {e}")
            success = False
            result = None
        finally:
            api_manager.return_key(api_key_cfg, success)

        return result

    def _safe_json_from_model_text(self, content: str) -> Optional[Dict[str, Any]]:
        if not content:
            return None
        c = content.strip()
        # 去掉 ```json 包裹 [3]
        if c.startswith("```"):
            c = re.sub(r"^```[a-zA-Z]*\n", "", c).strip()
            c = c.rstrip("`").strip()
        # 提取 JSON
        if not (c.startswith("{") and c.endswith("}")):
            m = re.search(r"\{.*\}", c, re.DOTALL)
            if m:
                c = m.group(0)
        try:
            obj = json.loads(c)
            return obj if isinstance(obj, dict) else None
        except Exception:
            self.logger.warning(f"AI返回非JSON: {content[:120]}")
            return None

    def _normalize_ai_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        # year: 提取4位年份或置空
        year = str(result.get("year", "")).strip()
        if any(w in year.lower() for w in ["未知", "unknown", "不详", "n/a", "none"]):
            result["year"] = ""
        else:
            m = re.search(r"\b(19|20)\d{2}\b", year)
            result["year"] = m.group(0) if m else ""

        # author/title/journal/publisher：未知置空；title若空视为问题文档
        def clean_text(v: Any) -> str:
            s = str(v or "").strip().replace("_", "-")
            if any(w in s.lower() for w in ["未知", "unknown", "不详", "n/a", "none"]):
                return ""
            return s

        result["author"] = clean_text(result.get("author", ""))
        result["journal"] = clean_text(result.get("journal", ""))
        result["publisher"] = clean_text(result.get("publisher", ""))

        title = clean_text(result.get("title", ""))
        if not title:
            result["title"] = "未知"
            result["is_problem_doc"] = True
        else:
            result["title"] = title
            result["is_problem_doc"] = False

        # type 限定
        t = str(result.get("type", "others")).strip().lower()
        if t not in {"book", "paper", "others", "unknown"}:
            t = "others"
        result["type"] = t
        return result

    # ---------------------------
    # Naming rules (保留你核心命名能力) [2][3]
    # ---------------------------
    def apply_case_rule(self, text: str) -> str:
        if not text:
            return text
        case_style = self.config.get("file_naming", {}).get("filename_rules", {}).get("case_style", "title")
        if case_style == "title":
            return " ".join(w.capitalize() for w in text.split())
        if case_style == "upper":
            return text.upper()
        if case_style == "lower":
            return text.lower()
        return text

    def generate_filename(self, info: Dict[str, Any], doc_type: str, original_stem: Optional[str] = None) -> str:
        # 问题文档：直接保留原文件名并加前缀 [3]
        if info.get("is_problem_doc") and original_stem:
            return f"[待处理]{original_stem}"

        patterns = self.config.get("file_naming", {}).get("naming_patterns", {})
        pattern = patterns.get(doc_type, patterns.get("others", "{title}_{year}"))  # 你配置中默认others [2]

        defaults = self.config.get("file_naming", {}).get("default_values", {
            "title": "",
            "author": "",
            "year": "",
            "publisher": "",
            "journal": "",
            "timestamp": "{datetime}",
        })

        processed = dict(info)
        for k, dv in defaults.items():
            if not processed.get(k):
                if k == "timestamp":
                    processed[k] = datetime.now().strftime("%Y%m%d_%H%M%S")
                else:
                    processed[k] = ""

        # year 兜底：用当前年份
        if not processed.get("year"):
            processed["year"] = datetime.now().strftime("%Y")

        for k in ["title", "author", "journal", "publisher"]:
            if processed.get(k):
                processed[k] = self.apply_case_rule(processed[k])

        try:
            filename = pattern.format(**processed)
        except Exception as e:
            self.logger.warning(f"模板格式错误，使用安全格式: {e}")
            filename = f"{processed.get('author','')}_{processed.get('title','文档')}_{processed.get('year','')}".strip("_")

        filename = self.clean_filename(filename)
        if not filename or filename.strip() in {"", "_", "-"}:
            filename = f"文档_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return self.handle_duplicate_filename(filename)

    def clean_filename(self, filename: str) -> str:
        rules = self.config.get("file_naming", {}).get("filename_rules", {})
        max_length = int(rules.get("max_length", 200))

        has_problem = filename.startswith("[待处理]")
        clean_part = filename[5:] if has_problem else filename

        replacements = {
            ":": "-",
            "：": "-",
            "<": "《",
            ">": "》",
            "|": "_",
            "?": "",
            "*": "",
            "/": "_",
            "\\": "_",
        }
        for old, new in replacements.items():
            clean_part = clean_part.replace(old, new)

        if rules.get("normalize_spaces", True):
            clean_part = re.sub(r"\s+", " ", clean_part)
        if rules.get("trim_whitespace", True):
            clean_part = clean_part.strip()

        clean_part = re.sub(r"_+", "_", clean_part)
        clean_part = re.sub(r"^_|_$", "", clean_part)

        out = f"[待处理]{clean_part}" if has_problem else clean_part
        if len(out) > max_length:
            out = out[:max_length]
        return out

    def handle_duplicate_filename(self, filename: str) -> str:
        base_name = filename[:-4] if filename.lower().endswith(".pdf") else filename
        counter = self.duplicate_tracker[base_name.lower()]
        if counter > 0:
            suffix_t = self.config.get("file_naming", {}).get("special_handling", {}).get("duplicate_suffix", "_{counter}")
            filename = f"{base_name}{suffix_t.format(counter=counter)}"
        self.duplicate_tracker[base_name.lower()] += 1
        return filename

    # ---------------------------
    # Single-file processing
    # ---------------------------
    def process_single_file(
        self,
        file_path: str,
        service_name: str,
        start_page: int,
        end_page: int,
        output_dir: str,
        processing_mode: str
    ) -> Dict[str, Any]:
        thread_name = threading.current_thread().name
        self.stats["thread_stats"][thread_name] += 1

        result = {
            "file_path": file_path,
            "success": False,
            "new_filename": None,
            "error": None,
            "thread": thread_name,
            "timestamp": datetime.now(),
            "ai_info": None,
            "category": "unknown",
            "final_path": None,
        }

        try:
            text = self.extract_text_from_pdf(file_path, start_page, end_page)

            # OCR标记路径：旧版用 NEEDS_OCR_PROCESSING 并加 [需要OCR] 前缀 [3]
            if text == "NEEDS_OCR_PROCESSING":
                ocr_prefix = self.config.get("file_naming", {}).get("special_handling", {}).get("ocr_prefix", "[需要OCR]")
                original = os.path.basename(file_path)
                new_name = f"{ocr_prefix}{original}"
                if processing_mode == "category":
                    final_path = self.move_file_to_category(file_path, self.output_folders["problem"], new_name)
                    result["final_path"] = final_path
                else:
                    # 原地重命名
                    new_path = os.path.join(os.path.dirname(file_path), new_name)
                    if os.path.abspath(new_path) != os.path.abspath(file_path):
                        os.rename(file_path, new_path)
                    result["final_path"] = new_path
                result["success"] = True
                result["new_filename"] = new_name
                result["category"] = "problem"
                result["ai_info"] = {"type": "needs_ocr", "title": "OCR处理"}
                return result

            info = self.call_ai_service(text, service_name)
            if not info:
                result["error"] = "AI分析失败"
                return result

            result["ai_info"] = info
            if info.get("is_problem_doc"):
                self.stats["problem_docs"] += 1

            doc_type = info.get("type", "others")
            original_stem = os.path.splitext(os.path.basename(file_path))[0]
            new_stem = self.generate_filename(info, doc_type, original_stem)
            result["new_filename"] = new_stem

            if processing_mode == "category":
                if info.get("is_problem_doc"):
                    target = self.output_folders["problem"]
                    final_filename = f"[待处理]{os.path.basename(file_path)}"
                    result["category"] = "problem"
                else:
                    target = self.output_folders["success"]
                    final_filename = f"{new_stem}.pdf"
                    result["category"] = "success"
                result["final_path"] = self.move_file_to_category(file_path, target, final_filename)
            else:
                # 原地重命名
                file_dir = os.path.dirname(file_path)
                if info.get("is_problem_doc"):
                    final_filename = f"[待处理]{os.path.basename(file_path)}"
                    result["category"] = "problem"
                else:
                    final_filename = f"{new_stem}.pdf"
                    result["category"] = "success"

                new_path = os.path.join(file_dir, final_filename)
                counter = 1
                while os.path.exists(new_path) and os.path.abspath(new_path) != os.path.abspath(file_path):
                    stem, ext = os.path.splitext(final_filename)
                    new_path = os.path.join(file_dir, f"{stem}_{counter}{ext}")
                    counter += 1
                if os.path.abspath(new_path) != os.path.abspath(file_path):
                    os.rename(file_path, new_path)
                result["final_path"] = new_path

            result["success"] = True
            return result

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            return result

    # ---------------------------
    # Batch processing
    # ---------------------------
    def process_files(
        self,
        input_dir: str,
        service_name: str,
        start_page: int,
        end_page: int,
        output_dir: str,
        processing_mode: str = "category",
    ):
        self.stats["start_time"] = time.time()

        if processing_mode == "category":
            print(f"{Fore.CYAN}📁 正在设置输出文件夹结构...{Style.RESET_ALL}")
            self.output_folders = self.setup_output_folders(output_dir)
        else:
            print(f"{Fore.CYAN}🔄 原地重命名模式：保持文件原位置{Style.RESET_ALL}")
            self.output_folders = None

        pdf_files = self.find_pdf_files(input_dir)
        if not pdf_files:
            print(f"{Fore.YELLOW}📂 未找到PDF文件: {input_dir}{Style.RESET_ALL}")
            return

        perf = self.config.get("processing", {}).get("performance", {})
        configured_workers = int(perf.get("max_workers", 8))
        available_keys = len(self.api_managers[service_name].available_keys)
        max_workers = max(1, min(configured_workers, available_keys))  # 线程数<=key数 [3]

        print(f"\n{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}🚀 开始批量处理{Style.RESET_ALL}")
        print(f"📁 输入目录: {input_dir}")
        print(f"📂 输出目录: {output_dir}")
        print(f"📄 文件数量: {len(pdf_files)}")
        print(f"📋 页面范围: {start_page}-{end_page}")
        print(f"🤖 AI服务: {service_name}")
        print(f"🧵 线程数: {max_workers} (配置:{configured_workers}, 密钥:{available_keys})")
        print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="PDFWorker") as executor:
            futures = {
                executor.submit(self.process_single_file, fp, service_name, start_page, end_page, output_dir, processing_mode): fp
                for fp in pdf_files
            }

            bar = tqdm(
                total=len(pdf_files),
                desc="🔄 处理进度",
                leave=True,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )

            for future in as_completed(futures):
                fp = futures[future]
                self.stats["processed"] += 1
                try:
                    r = future.result(timeout=120)
                    self.processing_log.append(r)
                    if r.get("success"):
                        self.stats["successful"] += 1
                    else:
                        self.stats["failed"] += 1
                        # 分类模式下：失败文件移入 failed
                        if processing_mode == "category" and self.output_folders:
                            try:
                                failed_name = f"[失败]{os.path.basename(fp)}"
                                self.move_file_to_category(fp, self.output_folders["failed"], failed_name)
                            except Exception:
                                pass
                except Exception as e:
                    self.stats["failed"] += 1
                    self.processing_log.append({"file_path": fp, "success": False, "error": str(e), "timestamp": datetime.now()})
                finally:
                    bar.update(1)

            bar.close()

        gc.collect()
        self.show_statistics()

    def show_statistics(self):
        duration = time.time() - self.stats["start_time"] if self.stats["start_time"] else 0
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}📊 处理完成统计{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"📄 处理文件: {self.stats['processed']}")
        print(f"{Fore.GREEN}✅ 成功: {self.stats['successful']}{Style.RESET_ALL}")
        print(f"{Fore.RED}❌ 失败: {self.stats['failed']}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}⚠️ 问题文档: {self.stats['problem_docs']}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}⏱️ 用时: {duration:.2f} 秒{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}🔄 API调用: {self.stats['api_calls']}{Style.RESET_ALL}")

        # 线程统计与资源
        if self.stats["thread_stats"]:
            print(f"{Fore.CYAN}🧵 线程分布:{Style.RESET_ALL}")
            for t, c in self.stats["thread_stats"].items():
                print(f"  {t}: {c} 个文件")

        mem = psutil.Process().memory_info().rss / 1024 / 1024
        cpu = psutil.cpu_percent()
        print(f"{Fore.YELLOW}💾 内存使用: {mem:.1f} MB{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}💻 CPU使用: {cpu:.1f}%{Style.RESET_ALL}")

        # API key/token统计（保留你原本的“key级统计”）[3]
        for svc, mgr in self.api_managers.items():
            ks = mgr.get_stats()
            ts = mgr.get_token_stats()
            if ks:
                print(f"{Fore.CYAN}🔑 {svc} 密钥调用统计:{Style.RESET_ALL}")
                for k, st in ks.items():
                    print(f"  {k}: {st['calls']} 次调用, {st['errors']} 次错误")
            if ts:
                total_tokens = sum(v["total_tokens"] for v in ts.values())
                if total_tokens:
                    print(f"{Fore.CYAN}💰 {svc} Token统计: {total_tokens:,} tokens{Style.RESET_ALL}")

        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

    # ---------------------------
    # Main interactive run
    # ---------------------------
    def run(self, input_dir: Optional[str] = None, output_dir: Optional[str] = None):
        try:
            if not input_dir:
                input_dir = input(f"{Fore.YELLOW}📁 请输入PDF文件目录路径: {Style.RESET_ALL}").strip().strip('"')
            if not input_dir or not os.path.exists(input_dir):
                print(f"{Fore.RED}❌ 目录不存在: {input_dir}{Style.RESET_ALL}")
                return 1

            service_name = self.select_ai_service()
            start_page, end_page = self.get_processing_pages()

            out_dir = output_dir or self.get_output_directory()
            mode = self.get_processing_mode()

            self.process_files(input_dir, service_name, start_page, end_page, out_dir, mode)

            print(f"\n{Fore.CYAN}🎯 处理完成!{Style.RESET_ALL}")
            return 0
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}⚠️ 用户中断{Style.RESET_ALL}")
            return 1
        except Exception as e:
            print(f"{Fore.RED}❌ 程序异常: {e}{Style.RESET_ALL}")
            return 1


class RenamerEngine:
    """对外引擎封装（供 hdt-renamer CLI 调用）"""

    def __init__(self, global_config: str, tool_config: str):
        self.app = DocumentRenamerEnhanced(global_config, tool_config)

    def run_interactive(self, input_dir: Optional[str] = None, output_dir: Optional[str] = None) -> int:
        return self.app.run(input_dir=input_dir, output_dir=output_dir)