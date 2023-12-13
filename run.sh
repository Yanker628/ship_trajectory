module load anaconda/2022.10
# 加载 cuda 11.3
module load cuda/11.7
# 激活 python 虚拟环境
source activate torch110
export PYTHONUNBUFFERED=1
python train.py imgsz = (1920 1080) batch=16 epoch=100 data='dataset/ICSHM_BSCC.yaml' optimizer='Adam' cfg='yolov8x.yaml' weights=yolov8x.pt