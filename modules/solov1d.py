import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18, resnet34, resnet50
from .nninit import xavier_init
from .solov1_decouple_head import SOLOv1DecoupleHead

class FPN(nn.Module):
    def __init__(self, 
               in_channels,
               out_channels,
               num_outs,
               start_level=0,
               end_level=-1,
               add_extra_convs=False,
               extra_convs_on_inputs=True,
               relu_before_extra_convs=False,
               no_norm_on_lateral=False,
               upsample_cfg=dict(mode='nearest')):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels  #[64, 128, 256, 512], outs of resnet18&34
        self.out_channels = out_channels  #256
        self.num_ins = len(in_channels)  #4 
        self.num_outs = num_outs        #5 
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()
        if end_level == -1:                  #default -1
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
  
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'
        
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Conv2d(in_channels[i], out_channels, kernel_size=1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:   #default false
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1)
                self.fpn_convs.append(extra_fpn_conv)

    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        used_backbone_levels = len(laterals)
        
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

class SOLOV1(nn.Module):
    def __init__(self, cfg=None, pretrained=None, mode='train', config=None, weights = None):
        super(SOLOV1, self).__init__()
        self.backbone = resnet50(pretrained=True, loadpath=config["backbone"]["path"])

        # this set only support resnet18 and resnet34 backbone, \\xe5\\x8f\\xaf\\xe4\\xbb\\xa5\\xe6\\xa0\\xb9\\xe6\\x8d\\xaesolo\\xe4\\xb8\\xadresent50\\xe7\\x9a\\x84\\xe9\\x85\\x8d\\xe7\\xbd\\xae\\xe8\\xbf\\x9b\\xe8\\xa1\\x8c\\xe6\\x9b\\xb4\\xe6\\x94\\xb9\\xef\\xbc\\x8c\\xe4\\xbd\\xbf\\xe5\\x85\\xb6\\xe6\\x94\\xaf\\xe6\\x8c\\x81resnset50\\xe7\\x9a\\x84\\xe8\\xae\\xad\\xe7\\xbb\\x83\\xef\\xbc\\x8c\\xe4\\xb8\\x8b\\xe5\\x90\\x8c
        self.fpn = FPN(in_channels=[256, 512, 1024, 2048], out_channels=256, start_level=0, num_outs=5, upsample_cfg=dict(mode='nearest'))
    
        # this set only support resnet18 and resnet34 backbone
        self.bbox_head = SOLOv1DecoupleHead(num_classes = config["head"]["num_classes"],
                            in_channels = config["head"]["in_channels"],
                            seg_feat_channels = config["head"]["seg_feat_channels"],
                            stacked_convs = config["head"]["stacked_convs"],
                            strides = config["head"]["strides"],
                            #scale_ranges=((1, 56), (28, 112), (56, 224), (112, 448), (224, 896)),
                            scale_ranges = config["head"]["scale_ranges"],
                            num_grids = config["head"]["num_grids"],
                            cate_down_pos = config["head"]["cate_down_pos"],
                            with_deform = config["head"]["with_deform"],
                            input_shape = (512, 512)
                        )
        
        self.mode = mode
        if self.mode=="test":
            self.nms_pre = config["test_cfg"]['nms_pre']
            self.score_thr = config["test_cfg"]['score_thr']
            self.mask_thr = config["test_cfg"]['mask_thr']
            self.update_thr = config["test_cfg"]['update_thr']
            self.kernel = config["test_cfg"]['kernel']
            self.sigma = config["test_cfg"]['sigma']
            self.max_per_img = config["test_cfg"]['max_per_img']
        self.seg_num_grids = config["head"]['num_grids']
        self.strides = config["head"]['strides']
        self.num_classes = config["head"]['num_classes']

        if self.mode == 'train':
            self.backbone.train(mode=True)
        else:
            self.backbone.train(mode=False)
        
        if pretrained is None and weights is None:
            self.init_weights() #if first train, use this initweight
        elif pretrained is not None and weights is None:
            self.load_weights(path=pretrained)
        elif pretrained is None and weights is not None:
            self.load_weights(weights=weights)

    def init_weights(self):
        # fpn
        if isinstance(self.fpn, nn.Sequential):
            for m in self.fpn:
                m.init_weights()
        else:
            self.fpn.init_weights()

        self.bbox_head.init_weights()
    
    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)

    def load_weights(self, path = None, weights = None):
        if path:
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
        else:
            self.load_state_dict(weights)

 
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img) #[1, 64, 128, 128],[128, 64],[256, 32],[512, 16]
        x = self.fpn(x) # [1, 256, 128, 128],[256, 64],[256, 32],[256, 16],[256, 8]
        return x
    
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs


    def forward(self, img, img_meta, return_loss=True, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):


        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, img_metas)

        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    # \\xe7\\x9f\\xad\\xe8\\xbe\\xb9resize\\xe5\\x88\\xb0448\\xef\\xbc\\x8c\\xe5\\x89\\xa9\\xe4\\xbd\\x99\\xe7\\x9a\\x84\\xe8\\xbe\\xb9pad\\xe5\\x88\\xb0\\xe8\\x83\\xbd\\xe8\\xa2\\xab32\\xe6\\x95\\xb4\\xe9\\x99\\xa4
    '''
    img_metas context
    'filename': 'data/casia-SPT_val/val/JPEGImages/00238.jpg', 
    'ori_shape': (402, 600, 3), 'img_shape': (448, 669, 3), 
    'pad_shape': (448, 672, 3), 'scale_factor': 1.1144278606965174, 'flip': False, 
    'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 
    'std': array([58.395, 57.12 , 57.375], dtype=float32), 'to_rgb': True}}

    '''

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_meta (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError('num of augmentations ({}) != num of image meta ({})'.format(len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1
        assert num_augs == 1
        return self.simple_test(imgs[0], img_metas[0], **kwargs)

    
    def simple_test(self, img, img_meta, rescale=False):
        #test_tensor = torch.ones(1,3,448,512).cuda()
        #x = self.extract_feat(test_tensor)
        x = self.extract_feat(img)
        outs = self.bbox_head(x, eval=True)
        seg_inputs = outs + (img_meta,
        self.nms_pre, self.score_thr,self.mask_thr, self.update_thr, self.kernel, self.sigma, self.max_per_img,
        rescale)
        seg_result = self.bbox_head.get_seg(*seg_inputs)
        return seg_result