# DCSG: Data Complement Pseudo Label Refinement and Self-Guided Pre-training for Unsupervised Person Re-identification

We have stored the weight file of the experimental results on Google Cloud Platform:
https://drive.google.com/drive/folders/1-YiIdaGYzXdT2_5Wm-0W21wLBkQZOM6r?usp=drive_link

note:Currently, we are publishing the baseline and the best weight files for testing purposes. The innovative source code of the article will be made available after its acceptance.

## Getting Started
### Installation
```shell
git clone https://github.com/duolaJohn/DCSG.git
cd DCSG
python setup.py develop

### Preparing Datasets
```
cd examples && mkdir data
```

The directory should look like
```
examples/data
├── Market-1501-v15.09.15
├── DukeMTMC-reID
└── MSMT17_V1
```

### Training
We utilize 2 GPUs for training and use 384x128 sized images for Market-1501 and DukeMTMC-reID.
We utilize 4 GPUs for training and use 384x128 sized images for MSMT17.

For Market-1501:
```
CUDA_VISIBLE_DEVICE=0,1 \
python examples/usl.py \
-d market1501 --logs-dir $PATH_FOR_LOGS
```
For DukeMTMC-reID:
```
CUDA_VISIBLE_DEVICE=0,1 \
python examples/usl.py \
-d duke --logs-dir $PATH_FOR_LOGS
```

For MSMT17:
```
CUDA_VISIBLE_DEVICE=0,1,2,3 \
python examples/usl.py \
-d msmt17 --logs-dir $PATH_FOR_LOGS
```


## Testing 

For Market-1501:
```
CUDA_VISIBLE_DEVICE=0\
python examples/test.py \
-d market1501 --resume $PATH_FOR_MODEL
```
For DukeMTMC-reID:
```
CUDA_VISIBLE_DEVICE=0\
python examples/test.py \
-d duke --resume $PATH_FOR_MODEL
```
For MSMT17:
```
CUDA_VISIBLE_DEVICE=0\
python examples/test.py \
-d msmt17 --resume $PATH_FOR_MODEL
```

## Testing with aggregated features

For Market-1501:
```
CUDA_VISIBLE_DEVICE=0\
python examples/test_aggregate.py \
-d market1501 --resume $PATH_FOR_MODEL
```
For DukeMTMC-reID:
```
CUDA_VISIBLE_DEVICE=0\
python examples/test_aggregate.py \
-d duke --resume $PATH_FOR_MODEL
```
For MSMT17:
```
CUDA_VISIBLE_DEVICE=0\
python examples/test_aggregate.py \
-d msmt17 --resume $PATH_FOR_MODEL
```

## Acknowledgement
Some parts of the code is borrowed from [SpCL](https://github.com/yxgeee/SpCL) and [PPLR](https://github.com/yoonkicho/PPLR)

## Citation
If you find this code useful for your research, please consider citing our paper
