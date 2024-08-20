#/bin/bash
source .env
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu121
pip install -r requirements.txt
