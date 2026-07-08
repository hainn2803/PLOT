# PLOT

conda create -p /anvil/scratch/x-hnguyen23/env/PLOT python=3.11 -y

python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130

python -m pip install -r requirements.txt