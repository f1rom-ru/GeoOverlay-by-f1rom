@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: .venv folder not found next to this script.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat

echo === Installing PyInstaller ===
python -m pip install pyinstaller
if errorlevel 1 goto fail

echo === Building (5-15 minutes) ===
python -m PyInstaller --noconfirm --clean --onedir --name GeoOverlay --uac-admin ^
  --collect-all geoclip ^
  --collect-all torchvision ^
  --collect-submodules transformers ^
  --copy-metadata transformers ^
  --copy-metadata torch ^
  --copy-metadata torchvision ^
  --copy-metadata tqdm ^
  --copy-metadata regex ^
  --copy-metadata packaging ^
  --copy-metadata filelock ^
  --copy-metadata numpy ^
  --copy-metadata tokenizers ^
  --copy-metadata safetensors ^
  --copy-metadata huggingface_hub ^
  --copy-metadata pyyaml ^
  geo_app.py
if errorlevel 1 goto fail

echo.
echo DONE: dist\GeoOverlay\GeoOverlay.exe
echo Copy the WHOLE dist\GeoOverlay folder, not only the exe.
pause
exit /b 0

:fail
echo.
echo BUILD FAILED. Copy the error text above and send it to Claude.
pause
exit /b 1
