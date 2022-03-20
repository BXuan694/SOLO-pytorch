# SOLO-pytorch
An instance segmentation mathod(https://arxiv.org/abs/1912.04488, https://arxiv.org/abs/2003.10152) without mmdet&mmcv.

res34: tested pass on coco2017-eval and mvtec.

# How to use
1. build loss
```
python setup.py develop
```
2. set configs like data/config_SOLO_r34.py

3. use train1.py/eval1.py for SOLO1 and train2.py/eval2.py for SOLO2

# TODO:
test res18/50

test solo1-decouple

add video support

optimize project structure

finish README.md

# ref

https://github.com/WXinlong/SOLO

https://github.com/OpenFirework/pytorch_solov2
