#### install dependences:

- I used Anaconda with python3.7
 https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
 
 - cuda10.2 with cudnn 7.6.5
 https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
 
 - dali
 pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali
 
 - apex
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
```
- inplace abn
```
pip install git+https://github.com/mapillary/inplace_abn.git@v1.0.11
```

- bonlime's pytorch tools
```
pip install git+https://github.com/bonlime/pytorch-tools.git@master
```
- my train loop library thunder-hummer
```
cd thunder-hammer
pip install -e .
```

#### dataset:
I used Pavel Pleskov dataset from kaggle. So I expected following folders in dataset folder:
- 512_S1_1
- 512_S2_1
- 512_S2_2
- 512_S3_1
- 512_S3_2
- 512_S4_1
- 512_S4_2
- 512_S5_1
- 512_S5_2
- 512_S5_3
- 512_S5_4
- 512_S6_1
- 512_S6_2
- 512_S7_1
- 512_S7_2
- 512_S7_3
- 512_S7_4
- 512_S8_1
- 512_S8_2
- 512_S8_3
- 512_S8_4
- 512_S8_5
- 512_S9_1
- 512_S9_2
- 512_S9_3
- 512_S9_4
- 512_S9_5
- 512_S10_1
- 512_S10_2
- 512_S10_3
- 512_S10_4

Copy `annotation` folder to dataset folder.

#### adjust path:
To specify path to dataset add paths to `configs/paths.yml` file in dict with pcname as key.


#### run training:

wsl_resnext101 d8 to run on 8 gpus

```
python -m src.main --config configs/rx101_stages_7.yml --trainer.gpus 8
```

wsl_resnext50 8 to run on 8 gpus

```
python -m src.main --config configs/rx50_stages_7.yml --trainer.gpus 8
```
