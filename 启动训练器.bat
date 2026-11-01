@echo off
chcp 65001 >nul
title Anima LoRA Trainer
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if not exist "..\venv\Scripts\python.exe" (
  echo.
  echo  [错误] 未找到 ..\venv\Scripts\python.exe
  echo.
  echo  请先按 README 的「安装」步骤创建虚拟环境；
  echo  若训练根目录不叫 D:\anima-train，请设置环境变量 ANIMA_BASE 后再启动。
  echo.
  pause
  exit /b 1
)

echo.
echo   Anima LoRA Trainer
echo   ---------------------------------------------
echo   启动后浏览器会自动打开；关闭本窗口即停止服务
echo.

"..\venv\Scripts\python.exe" app.py
if errorlevel 1 pause
