uv venv venv

.\venv\Scripts\Activate.ps1

pip uninstall torch -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
