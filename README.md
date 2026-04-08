# Installation

Step 1: Create Environment
Create a new conda environment:

conda create -n GeoNet python=3.8

Step 2: Activate Environment

conda activate GeoNet

Step 3: Install Dependencies

The main packages required:

torch - PyTorch framework (v1.9.0+cu111)

torch-geometric - Graph neural network library (v2.0.3)

torch-scatter - Scatter operations for PyTorch (v2.0.8)

ase - Atomic Simulation Environment

rdkit - Chemical informatics toolkit

wandb - Experiment tracking

pytorch_lightning - Training framework (v1.5.0)

# Install PyTorch with CUDA support

pip install torch==1.9.0+cu111 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu111

# Install PyTorch Geometric and related packages

pip install torch-scatter==2.0.8 torch-sparse==0.6.10 torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html

pip install torch-geometric==2.0.3

# Install additional dependencies

pip install pytorch_lightning==1.5.0

pip install wandb torch-ema ase sympy

pip install opencv-python-headless

conda install yaml -y

# Install framework-specific requirements

pip install -r lightnp_env_requirements.txt

# Running

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=1230 \
  run_ddp.py \
    --datapath ./ \
    --model=Visnorm_shared_LSRMNorm2_2branchSerial \
    --molecule AT_AT_CG_CG \
    --dataset=my_dataset \
    --group_builder rdkit \
    --num_interactions=6 --long_num_layers=2 \
    --lr=0.0004 --rho_criteria=0.001 \
    --dropout=0 --hidden_channels=128 \
    --calculate_meanstd --otfcutoff=4 \
    --short_cutoff_upper=4 --long_cutoff_lower=0 --long_cutoff_upper=9 \
    --early_stop --early_stop_patience=500 \
    --no_broadcast --batch_size=16 \
    --ema_decay=0.999 --dropout=0.1 \
    --wandb --api_key [YOUR_WANDB_API_KEY]
