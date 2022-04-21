from data.config import cfg, process_funcs_dict
from data.coco import CocoDataset
from data.loader import build_dataloader
# from modules.solov1 import SOLOV1 as solo
# from modules.solov2 import SOLOV2 as solo
from modules.solov1d import SOLOV1 as solo
import time
import torch
import numpy as np

# 梯度均衡
def clip_grads(params_):
    params_ = list(filter(lambda p: p.requires_grad and p.grad is not None, params_))
    if len(params_) > 0:
        return torch.nn.utils.clip_grad.clip_grad_norm_(params_, max_norm=35, norm_type=2)

# 设置新学习率
def set_lr(optimizer_, newLr_):
    for paramGroup_ in optimizer_.param_groups:
        paramGroup_['lr'] = newLr_

# 设置requires_grad为False
def gradinator(x_):
    x_.requires_grad = False
    return x_

# 设置pipline
def build_process_pipeline(pipelineConfgs_):
    assert isinstance(pipelineConfgs_, list)
    process_pipelines = []
    for pConfig_ in pipelineConfgs_:
        assert isinstance(pConfig_, dict) and 'type' in pConfig_
        args = pConfig_.copy()
        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            process_pipelines.append(process_funcs_dict[obj_type](**args))
    return process_pipelines

# 计算warmup学习率
def get_warmup_lr(curIter_, totalIters_, baseLr_, warmupRatio_, warmUpOption='linear'):
    if warmUpOption == 'constant':
        warmupLr = baseLr_ * warmupRatio_ 
    elif warmUpOption == 'linear':
        k = (1 - curIter_ / totalIters_) * (1 - warmupRatio_)
        warmupLr = baseLr_ * (1 - k)
    elif warmUpOption == 'exp':
        k = warmupRatio_**(1 - curIter_ / totalIters_)
        warmupLr = baseLr_ * k
    return warmupLr


def train(globalStartEpoch, totalEpoches):

    # train process pipelines func
    trainTransformsPiplines = build_process_pipeline(cfg.train_pipeline)
    print(trainTransformsPiplines)
    # build datashet
    casiadata = CocoDataset(ann_file=cfg.dataset.train_info,
                            pipeline = trainTransformsPiplines,
                            img_prefix = cfg.dataset.trainimg_prefix,
                            data_root=cfg.dataset.train_prefix)
    torchdataLoader = build_dataloader(casiadata, cfg.imgs_per_gpu, cfg.workers_per_gpu, num_gpus=cfg.num_gpus, shuffle=True)

    if cfg.resume_from is None:
        model = solo(cfg, pretrained=None, mode='train')
        print('cfg.resume_from is None')
    else:
        model = solo(cfg, pretrained=cfg.resume_from, mode='train')
    model = model.cuda()
    model = model.train()

    lrOri = cfg.optimizer['lr']
    lrStages = cfg.lr_config["step"]
    lrList = np.full(totalEpoches, lrOri)
    for ii in range(len(lrStages)):
        lrList[lrStages[ii]:]*=0.1
    print("starting epoch: ", globalStartEpoch)
    print("lr adapting stages: ", end=' ')
    for ii in range(len(lrStages)):
        print(cfg.lr_config["step"][ii], end=" ")
    print("\ntotal training epoches: ", totalEpoches)

    optimizer_config = cfg.optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=optimizer_config['lr'], momentum=optimizer_config['momentum'], weight_decay=optimizer_config['weight_decay'])

    batchSize = cfg.imgs_per_gpu * cfg.num_gpus
    epochSize = len(casiadata) // batchSize  
    # nums of trained epoches, idx of epoch to start
    pastEpoches = globalStartEpoch
    # nums of trained iters, idx of iter to start
    pastIters = (globalStartEpoch-1) * epochSize
    # nums of left epoches
    leftEpoches = totalEpoches - pastEpoches + 1
    # nums of left iters
    leftIters = leftEpoches * epochSize

    print('##### begin train ######')
    currentIter = 0   
 
    for epoch in range(leftEpoches):

        currentEpoch = epoch + pastEpoches
        # 终止训练
        if currentEpoch >= totalEpoches:
            print("Current epoch is larger than setting epoch nums, training stop.")
            return

        # 仅用于打印
        loss_sum = 0.0 
        loss_ins = 0.0 
        loss_cate = 0.0
       
        for j, data in enumerate(torchdataLoader):
            iterStartTime = time.time()

            if cfg.lr_config['warmup'] is not None and pastIters < cfg.lr_config['warmup_iters']:
                cur_lr = get_warmup_lr(pastIters, cfg.lr_config['warmup_iters'],
                                        optimizer_config['lr'], cfg.lr_config['warmup_ratio'],
                                        cfg.lr_config['warmup'])
            else:
                cur_lr = lrList[currentEpoch]
            set_lr(optimizer, cur_lr)

            imgs = gradinator(data['img'].data[0].cuda())
            img_meta = data['img_metas'].data[0]   #图片的一些原始信息
            gt_bboxes = []
            for bbox in data['gt_bboxes'].data[0]:
                bbox = gradinator(bbox.cuda())
                gt_bboxes.append(bbox)
            
            gt_masks = data['gt_masks'].data[0]  #cpu numpy data
            
            gt_labels = []
            for label in data['gt_labels'].data[0]:
                label = gradinator(label.cuda())
                gt_labels.append(label)


            loss = model.forward(img=imgs,
                    img_meta=img_meta,
                    gt_bboxes=gt_bboxes,
                    gt_labels=gt_labels,
                    gt_masks=gt_masks)


            losses = loss['loss_ins'] + loss['loss_cate']
            loss_sum += losses.cpu().item()
            loss_ins += loss['loss_ins'].cpu().item()
            loss_cate += loss['loss_cate'].cpu().item()

            optimizer.zero_grad()
            losses.backward()

            if torch.isfinite(losses).item():
                grad_norm = clip_grads(model.parameters())  #梯度平衡
                optimizer.step()
            else:
                NotImplementedError("loss type error!can't backward!")

            leftIters -= 1
            pastIters += 1
            currentIter += 1

            showIters = 10
            if j%int(showIters) == 0 and j != 0:
                iterLastTime = time.time() - iterStartTime
                left_seconds = iterLastTime * leftIters
                left_minutes = left_seconds / 60.0
                left_hours = left_minutes / 60.0
                left_days = left_hours // 24
                left_hours = left_hours % 24

                out_srt = 'epoch:['+str(currentEpoch)+']/['+str(totalEpoches)+'],' # end of epoch of idx currentEpoch
                out_srt = out_srt + '['+str(j)+']/'+str(epochSize)+'], left_time: ' + str(left_days)+'days '+format(left_hours,'.2f')+'hours,'
                print(out_srt, "loss:", format(loss_sum/showIters,'.4f'), 'loss_ins:', format(loss_ins/showIters,'.4f'), "loss_cate:", format(loss_cate/showIters,'.4f'), "lr:", format(cur_lr,'.8f'))
                loss_sum = 0.0 
                loss_ins = 0.0 
                loss_cate = 0.0

        leftEpoches -= 1

        save_name = "./weights/solo1/" + cfg.name + "_epoch_" + str(currentEpoch) + ".pth"
        model.save_weights(save_name)        

if __name__ == '__main__':
    train(globalStartEpoch=cfg.epoch_iters_start, totalEpoches=cfg.total_epoch)   #设置本次训练的起始epoch
