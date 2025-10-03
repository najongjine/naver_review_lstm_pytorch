uv venv venv

.\venv\Scripts\Activate.ps1

uv pip uninstall torch -y
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
