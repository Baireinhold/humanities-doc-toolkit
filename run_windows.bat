@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

set "LOG=%~dp0run_windows.log"
echo ================================================================ > "%LOG%"
echo Humanities Doc Toolkit (v0.1.0) - One-click Launcher            >> "%LOG%"
echo Author: Baireinhold  Email: Baireinhold@163.com                 >> "%LOG%"
echo Workdir: %cd%                                                   >> "%LOG%"
echo ================================================================ >> "%LOG%"

echo ================================================================
echo Humanities Doc Toolkit (v0.1.0) - One-click Launcher
echo Author: Baireinhold  Email: Baireinhold@163.com
echo Log: %LOG%
echo ================================================================
echo.

REM 0) 选择 Python>=3.10 (不使用 where python，避免卡住) [16]
set "PYEXE="

py -3.10 -c "import sys; assert sys.version_info>=(3,10), sys.version" >> "%LOG%" 2>&1
if not errorlevel 1 set "PYEXE=py -3.10"

if "%PYEXE%"=="" (
  python -c "import sys; assert sys.version_info>=(3,10), sys.version" >> "%LOG%" 2>&1
  if not errorlevel 1 set "PYEXE=python"
)

if "%PYEXE%"=="" (
  echo [ERROR] 未检测到可用的 Python 3.10+。请安装 Python>=3.10。>> "%LOG%"
  echo [ERROR] 未检测到可用的 Python 3.10+。请安装 Python>=3.10。
  pause
  exit /b 1
)

echo [INFO] 使用解释器: %PYEXE%>> "%LOG%"
echo [INFO] 使用解释器: %PYEXE%

REM 1) 创建 .venv (关键：用同一个解释器创建) [16]
if not exist ".venv\Scripts\python.exe" (
  echo [INFO] 创建虚拟环境 .venv ...>> "%LOG%"
  echo [INFO] 创建虚拟环境 .venv ...
  %PYEXE% -m venv .venv >> "%LOG%" 2>&1
  if errorlevel 1 (
    echo [ERROR] venv 创建失败。>> "%LOG%"
    echo [ERROR] venv 创建失败,请打开 run_windows.log 查看。
    pause
    exit /b 1
  )
)

REM 2) 确保 pip + 升级 pip [16]
".venv\Scripts\python.exe" -m ensurepip --upgrade >> "%LOG%" 2>&1
".venv\Scripts\python.exe" -m pip install -U pip setuptools wheel >> "%LOG%" 2>&1

REM 3) 安装项目(Editable) [16][13]
echo [INFO] 安装/更新依赖(首次可能较慢，取决于网络，可能5分钟以上，请耐心等待)...
".venv\Scripts\python.exe" -m pip install -e . >> "%LOG%" 2>&1
if errorlevel 1 (
  echo [ERROR] pip install -e . 失败,请打开 run_windows.log 查看。>> "%LOG%"
  echo [ERROR] 安装失败,请打开 run_windows.log 查看。
  pause
  exit /b 1
)

REM 4) 复制示例配置到根目录(README 约定) [16][10]
if not exist "global.yaml" if exist "configs\global.example.yaml" copy /Y "configs\global.example.yaml" "global.yaml" >nul
if not exist "renamer.yaml" if exist "configs\renamer.example.yaml" copy /Y "configs\renamer.example.yaml" "renamer.yaml" >nul
if not exist "classifier.yaml" if exist "configs\classifier.example.yaml" copy /Y "configs\classifier.example.yaml" "classifier.yaml" >nul
if not exist "sorter.yaml" if exist "configs\sorter.example.yaml" copy /Y "configs\sorter.example.yaml" "sorter.yaml" >nul

echo.
echo [INFO] 启动总菜单(使用 .venv 的 python):python -m hdt.cli
echo 提示:首次运行请编辑 global.yaml 填入 API key
echo.

REM 5) 用 venv 的 python 启动（统一入口 hdt.cli:main）[13][10][16]
".venv\Scripts\python.exe" -m hdt.cli
set "CODE=%ERRORLEVEL%"

echo ExitCode=%CODE%>> "%LOG%"
echo.
echo [DONE] 结束。ExitCode=%CODE%
echo 详细日志: %LOG%
pause
exit /b %CODE%