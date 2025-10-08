### llava environment settings..

#### conda create -n llava python=3.10
#### conda activate llava
#### pip install --extra-index-url https://download.pytorch.org/whl/cu117 torch==2.0.1 torchvision==0.15.2
#### pip install triton==2.0.0 accelerate==0.21.0 peft==0.4.0 transformers==4.36.2 einops==0.6.1 einops-exts==0.0.4 timm==0.9.16 sentencepiece==0.1.99
#### pip install "numpy<2"
#### conda install -c nvidia/label/cuda-11.7.1 cuda-toolkit
#### export CUDA_HOME=$CONDA_PREFIX
#### export PATH=$CUDA_HOME/bin:$PATH
#### export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
#### git clone https://github.com/Dao-AILab/flash-attention.git
#### cd flash-attention
#### git checkout v2.6.3 
#### pip install ninja
#### pip install --no-build-isolation -e .
#### python -c "import flash_attn"
#### cd /data3/jykim/Projects/VLM/LLaVA
#### pip install --no-deps -e .
#### pip install deepspeed==0.11.1
#### python -c "import deepspeed; print(deepspeed.__version__)"
#### python -c "from llava.model import LlavaLlamaForCausalLM"

