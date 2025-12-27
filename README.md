# humanities-doc-toolkit
AI-assisted PDF renaming/classification/sorting pipeline

本仓库是面向人文学科研究的文献处理工具链（CLI 优先）：
- Renamer：按可配置命名规范批量标准化文件名（例如 作者_(标题)_出版社_年份）[2]
- Classifier：针对人文学科的语义分类与文件夹归档（预设/动态/混合）[6]
- Sorter：基于研究需求的自然语言智能分拣（两轮：文件名批量 + 内容深度）[7]

## 安全提示（必须）
请勿提交真实 API Key。你的旧配置文件中存在真实 key 风险，必须改为 example 配置并通过 .gitignore 忽略真实配置 [2][8][5]。

## 安装
python -m pip install .

## 运行（3 个独立命令）
hdt-renamer --help
hdt-classifier --help
hdt-sorter --help

## 配置（分层）
复制示例配置到仓库根目录：
- configs/global.example.yaml -> global.yaml（共享 ai_services；Classifier 的 ai.services 会被兼容映射）[6]
- configs/renamer.example.yaml -> renamer.yaml
- configs/classifier.example.yaml -> classifier.yaml
- configs/sorter.example.yaml -> sorter.yaml

## Web（测试版）
extras/web/ 仅作为测试壳。当前版本 Web 任务管理层会调用 renamer 引擎 [1]，但不作为 v0.1 推荐入口。