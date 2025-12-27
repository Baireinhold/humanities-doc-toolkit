
# Humanities Doc Toolkit（人文文献工具链）

Humanities Doc Toolkit 是一套面向人文学科研究与文献治理的命令行工具链，用于把“杂乱的 PDF/文档文件”整理为可检索、可复核、可持续维护的个人/团队资料库。

它包含三个核心工具，并提供一个统一的交互式主菜单入口：

- **Renamer**：批量提取文献信息并重命名（支持人文常见命名范式；多线程；多密钥轮询）
- **Classifier**：智能分类与归档（默认 dry-run；先生成清单，核对后再 apply 执行移动；强调可复核、低风险）
- **Sorter**：按研究主题智能分拣（两轮筛选：文件名语义初筛 + 内容深度分析；默认 copy，降低误操作风险）

---

## 1. 适用场景

- **研究生/学者**：把下载目录里的大量 PDF 统一命名、分类、形成结构化资料库，为后续写作/综述/笔记做准备。
- **课题组/图书馆/档案工作流**：批量整理大规模文献，追求可审计、可复核、可批处理（日志与清单输出）。
- **需要多模型/多密钥轮询的 AI 文献处理管线**：通过多线程与多 Key 轮询提高吞吐，降低限流导致的失败率。

---

## 2. 工具概览（主菜单与三件套）

### 2.1 `hdt`（统一主菜单：推荐入口）
运行后会显示主菜单，选择 `1/2/3/4/0` 执行对应功能；每个功能结束后会返回主菜单。

首次运行会自动从 `configs/` 复制示例配置到项目根目录生成：

- `global.yaml`
- `renamer.yaml`
- `classifier.yaml`
- `sorter.yaml`

你需要先编辑 `global.yaml` 填入 API Key，才能使用 AI 功能。

### 2.2 Renamer（批量重命名）
功能要点：

- 扫描目录中的 PDF（默认只扫描 `.pdf`，可在配置中调整）
- 提取指定页范围文本（默认 1–10 页，可配置）
- 调用 AI 提取元信息（title/author/year/type/journal/publisher 等）
- 按模板生成文件名，并按模式执行：
  - **分类模式**：输出到“处理成功/处理失败/问题文件”等子目录（适合清洗下载目录）
  - **原地重命名模式**：在原目录直接重命名（适合已整理目录的统一标准化）

适合场景：

- 文献下载目录的清洗与标准化命名
- 为后续 Zotero/Obsidian/向量库/检索系统提供稳定的文件命名与元信息

### 2.3 Classifier（安全归类：dry-run + apply）
这是一个强调“低风险与可复核”的分类器：

- **默认 dry-run**：只生成 JSON/TXT 清单与日志，不移动文件
- 你核对清单后，再执行 **apply**：严格按清单中的 `planned_to` 路径执行移动

推荐工作流：

1. dry-run 生成分类清单（JSON/TXT）
2. 打开 JSON 检查每个文件的 `planned_to` 是否符合预期
3. 确认无误后执行 apply 落盘

适合场景：

- 重要资料库的归档（不允许误操作）
- 团队协作中需要审计与复核的批处理流程

### 2.4 Sorter（研究主题智能分拣）
用于“按研究需求”从大库中筛出与主题高度相关的一部分文献：

- **第一轮**：基于文件名批量评估相关性（高吞吐、低成本）
- **第二轮**：基于内容预览进行深度分析（更精准）
- 支持“仅第一轮快速筛选”模式，适合探索性研究

默认行为是 **copy**：把结果复制到新的输出文件夹，保留原库不动。

适合场景：

- 写论文/开题/综述时，从大库中快速筛出“真正相关”的阅读集
- 主题变动频繁时，快速构建多个主题子库

---

## 3. 安装与一键启动（推荐）

本项目要求 **Python 3.10 或更高版本**。

### 3.1 Windows（推荐：双击一键启动）
- 直接双击：`run_windows.bat`
- 脚本会自动创建 `.venv` 并安装依赖，然后启动主菜单

失败排查：
- 查看 `run_windows.log`
- 常见原因：Python 未安装/版本过低/系统环境变量异常

### 3.2 macOS（推荐：Finder 双击启动）
- 双击：`run.command`
- 脚本会自动选择可用的 Python（3.10+），创建 `.venv`，安装依赖，然后启动主菜单

失败排查：
- 查看 `run_macos_launch.log`
- 若系统提示“无法打开/来源不明”，在系统设置 → 隐私与安全 中选择“仍要打开”
-  macOS（新手一键命令）

### 1）最推荐：在终端一键启动（无需理解任何 Python）
1. 打开 **终端**（Terminal）
2. 进入项目目录（把路径替换成你的项目所在位置）：
```bash
cd "/path/to/humanities-doc-toolkit"
```
3. 直接运行一键脚本：
```bash
bash run.command
```

说明：`run.command` 会写日志到 `run_macos_launch.log`，并自动创建/复用 `.venv` 安装依赖后启动主菜单 [16]。

---

### 2）如果提示“权限不足/不能执行”：一键修复再启动
```bash
cd "/path/to/humanities-doc-toolkit"
chmod +x run.command
bash run.command
```

---

### 3）如果提示 `bash\r` 或类似换行符问题：一键修复（常见于脚本从 Windows 拷贝过来）
```bash
cd "/path/to/humanities-doc-toolkit"
sed -i '' $'s/\r$//' run.command
sed -i '' $'s/\r$//' run.sh
bash run.command
```

### 4）之后再访达里双击run.command运行即可

### 3.3 Linux / WSL（推荐：终端运行）
进入项目根目录后运行：

```bash
bash run.sh
```

WSL 注意事项：
- 从 `/mnt/c`、`/mnt/d` 访问 Windows 挂载目录时，可能遇到权限或换行符问题
- 建议总是用 `bash run.sh`（不依赖可执行位），并确保脚本是 LF 换行

---

## 4. 配置文件说明（必须先配置）

### 4.1 配置文件总览
- `global.yaml`：全局配置（AI 服务、密钥、默认服务、超时等），三个工具共享
- `renamer.yaml / classifier.yaml / sorter.yaml`：各工具差异化策略配置（命名模板、分类规则、分拣阈值等）

安全提示：
- 不要在仓库提交真实 `global.yaml`（包含密钥）
- 建议仅提交 `configs/*.example.yaml` 作为模板

### 4.2 `global.yaml`：AI 服务与密钥
你可以配置多个服务（如 OpenAI-compatible、Claude、Gemini 等），并为每个服务设置多把密钥。一般建议：

- 日常使用：只启用一个主力服务 + 多把 key
- 工业/批处理：启用多个服务或多把 key，以降低限流风险

常用可调项：
- 默认服务选择
- 请求超时
- 模型名称与输出 token 上限
- temperature（越低越稳定）

### 4.3 `renamer.yaml`：用户可自定义项（重点）
- **文本提取页范围**：起始页/结束页
- **prompt 截断长度**：控制发送给模型的最大字符数（影响成本与准确度）
- **最大线程数**：线程数会结合可用 key 数量进行限制（key 不足时会自动降线程）
- **命名模板**：支持按文档类型（book/paper/others/unknown 等）使用不同模板
- **语言策略**：
  - `keep_original`：保留原文
  - `translate`：把指定字段翻译到目标语言  
  注意：作者字段不翻译（避免引用不一致）

### 4.4 `classifier.yaml`：用户可自定义项（重点）
- **分类模式**：预设 / 动态 / 混合（推荐混合）
- **预设分类字典**：可按你的学科与研究方向扩充（中文/英文关键词混合）
- **置信度阈值**：低于阈值默认不移动，并记录为“低置信度”
- **多线程**：可配置线程数；线程越高吞吐越大，但也更容易触发限流
- **PDF 提取引擎**：可选择自动/指定引擎；自动模式会尝试更快更稳的引擎并在失败时回退
- **备份开关**：重要资料库建议开启备份

### 4.5 `sorter.yaml`：用户可自定义项（重点）
- **筛选模式**：fast_first_round / balanced / precise / broad（影响阈值、批大小、是否跑第二轮）
- **阈值**：第一轮更宽松用于召回，第二轮更严格用于精度
- **线程数**：可运行时输入；也可配置默认值
- **输出行为**：默认 copy；如需 move 建议在充分验证后再启用
- **扫描格式**：默认支持多种格式；若某些格式暂不需要，建议从列表删除以提升稳定性与速度

---

## 5. 多线程与大规模处理（工程/工业级建议）

本项目支持多线程与多密钥轮询，适合处理大量文献，但需要正确配置以获得稳定吞吐。

### 5.1 线程数与 API Key 数量
经验原则：

- 如果 AI 服务有限流，线程数过高会导致大量失败、重试与限流
- 建议线程数与可用 key 数量保持合理比例：key 越多，吞吐越高；key 少时应降低线程数以提升稳定性

### 5.2 大规模任务建议
- 先用小样本目录验证命名/分类/分拣策略，再对全库执行
- Classifier 强烈建议先 dry-run 再 apply
- 对于重要资料库，开启备份，并把 output/backup 放在空间充足的磁盘上
- 长任务建议在稳定网络环境运行，必要时使用固定代理或企业网络出口

---

## 6. 常见问题（FAQ）

### 6.1 运行脚本后没有反应/闪退
- Windows：查看 `run_windows.log`
- macOS：查看 `run_macos_launch.log`，或在终端执行 `bash run.command`
- Linux/WSL：使用 `bash run.sh` 并查看 `run_unix.log`

### 6.2 AI 服务不可用/一直失败
请检查 `global.yaml`：
- 服务是否启用
- 是否填入有效 key
- base_url / model 是否正确
- 网络是否可访问 API

---

## 7. 开发与移植建议

### 7.1 作为 Python 包安装（适合开发者/自动化）
在项目根目录：

```bash
pip install -e .
```

安装后可用命令：
- `hdt`
- `hdt-renamer`
- `hdt-classifier`
- `hdt-classifier-apply --log <classification_details_*.json>`
- `hdt-sorter`

### 7.2 面向未来扩展的建议（路线图）
本项目定位为“文献治理基础设施”，后续扩展方向包括：

- **PDF/图像 → Markdown**：接入视觉大模型或本地 OCR/版面分析，对扫描件/古籍等难识别材料输出结构化 Markdown，并保留可校对中间产物。
- **Web UI**：提供任务创建、进度与日志查看、清单审核（尤其适合 Classifier 的 dry-run 审核流程）。
- **自然语言入口**：用自然语言描述任务，由解析组件将意图转换为可执行的命令与配置，并生成可复核的执行计划，实现“自然语言操控本机文件”的安全闭环。

---

## 8. 开源规范（建议遵守）

- 仓库只保留 `configs/*.example.yaml` 模板；真实 `global.yaml` 等配置只存在于本地
- 不提交运行产物：`.venv/`、`logs/`、`output/`、`classified/`、`backup/`、`*.log` 等
- 大规模任务前先在小样本验证，再对全库运行

---

## 9. 许可证

本项目使用 **MIT License**。

---

## 10. 基本命令自查表（Windows / macOS / Linux/WSL）

这一节用于“最小化排错”：当你遇到启动失败、依赖缺失、版本不对时，按顺序执行即可定位问题。

### 10.1 Windows（PowerShell 或 CMD）
检查 Python 与 Launcher：
```bat
py -3.10 --version
python --version
```

进入项目并启动（推荐脚本）：
```bat
cd /d D:\path\to\humanities-doc-toolkit
run_windows.bat
```

手动启动（脚本失败时备用）：
```bat
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -U pip setuptools wheel
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m hdt.cli
```

查看日志：
```bat
type run_windows.log
```

### 10.2 macOS（Terminal）
检查 Python（确保 ≥3.10）：
```bash
python3 --version
python3.12 --version
```

运行可点击脚本（或在终端直接跑）：
```bash
cd /path/to/humanities-doc-toolkit
bash run.command
```

手动启动（脚本失败时备用）：
```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -U pip setuptools wheel
.venv/bin/python -m pip install -e .
.venv/bin/python -m hdt.cli
```

查看日志：
```bash
tail -n 200 run_macos_launch.log
```

### 10.3 Linux / WSL（bash）
检查 Python（确保 ≥3.10）：
```bash
python3 --version
```

推荐启动：
```bash
cd /path/to/humanities-doc-toolkit
bash run.sh
```

WSL 从 Windows 盘进入示例：
```bash
cd /mnt/d/humanities-doc-toolkit
bash run.sh
```

若脚本因换行符报错（出现 `bash\r`），修复后再跑：
```bash
sed -i 's/\r$//' run.sh
bash run.sh
```

查看日志：
```bash
tail -n 200 run_unix.log
```
