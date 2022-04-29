# SOLO-pytorch
An instance segmentation mathod(https://arxiv.org/abs/1912.04488, https://arxiv.org/abs/2003.10152) without mmdet&mmcv.

Backbone of res50 has been tested pass on coco2017-val. For backbone of res18&34, easier datasets such as MVtec60(https://www.mvtec.com/company/research/datasets/mvtec-d2s) are tested pass.

# Request

python3.8.13, PyTorch1.7.1

# How to use
1. build loss
```
python setup.py build_ext --develop
```

2. prepare data as COCO format

3. set configs like data/config.py

4. import SOLOv1/SOLOv2 in the beginning of train.py/eval.py

# TODO:
test video support

finish README.md

# ref

https://github.com/WXinlong/SOLO

https://github.com/OpenFirework/pytorch_solov2
