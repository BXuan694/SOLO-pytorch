# SOLO-pytorch
An instance segmentation mathod(https://arxiv.org/abs/1912.04488, https://arxiv.org/abs/2003.10152) without mmdet&mmcv.

Backbone of res50 has been tested pass on coco2017-val. For backbone of res18&34, easier datasets such as MVtec60(https://www.mvtec.com/company/research/datasets/mvtec-d2s) are tested pass.

# Request

python3.7.11, PyTorch1.9.0, torchvision0.10.0

# How to use
1. build loss
```
python setup.py develop
```
2. set configs like data/config.py

3. use train1.py/eval1.py for SOLO1 and train2.py/eval2.py for SOLO2

# TODO:
test solo1-decouple

test video support

optimize project structure

finish README.md

# ref

https://github.com/WXinlong/SOLO

https://github.com/OpenFirework/pytorch_solov2
