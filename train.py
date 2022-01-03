from data.config import cfg, process_funcs_dict
from data.coco import CocoDataset
from data.loader import build_dataloader
from modules.solov2 import SOLOV2
import time
import torch

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


def train(globalStartEpoch=1, totalEpoches=36):
    # train process pipelines func
    trainTransformsPiplines = build_process_pipeline(cfg.train_pipeline)
    print(trainTransformsPiplines)
    # build datashet
    casiadata = CocoDataset(ann_file=cfg.dataset.train_info,
                            pipeline = trainTransformsPiplines,
                            img_prefix = cfg.dataset.trainimg_prefix,
                            data_root=cfg.dataset.train_prefix)
    torchdataLoader = build_dataloader(casiadata, cfg.imgs_per_gpu, cfg.workers_per_gpu, num_gpus=cfg.num_gpus, shuffle=True)
    batchSize = cfg.imgs_per_gpu * cfg.num_gpus
    epochSize = len(casiadata) // batchSize  
    
    if cfg.resume_from is None:
        model = SOLOV2(cfg, pretrained=None, mode='train')
        print('cfg.resume_from is None')
    else:
        model = SOLOV2(cfg, pretrained=cfg.resume_from, mode='train')
    
    model = model.cuda()
    model = model.train()

    optimizer_config = cfg.optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=optimizer_config['lr'], momentum=optimizer_config['momentum'], weight_decay=optimizer_config['weight_decay'])

    numStages = len(cfg.lr_config["step"])
    print("starting epoch: ", globalStartEpoch)
    print("lr adapting stages: ", end=' ')
    for ii in range(numStages):
        print(cfg.lr_config["step"][ii], end=" ")
    print("\ntotal training epoches: ", totalEpoches)

    lrOri = cfg.optimizer['lr']
    if globalStartEpoch < cfg.lr_config["step"][0]:
        set_lr(optimizer, lrOri)
    elif globalStartEpoch >= totalEpoches:
        print("Starting epoch is too large, no more training is needed.")
        exit()
    else:
        for ii in range(numStages-1):
            if cfg.lr_config["step"][ii] <= globalStartEpoch < cfg.lr_config["step"][ii+1]:
                set_lr(optimizer, lrOri*(0.1**(ii+1)))
                break
    
    # nums of trained epoches, idx of epoch to start
    pastEpoches = globalStartEpoch
    # nums of trained iters, idx of iter to start
    pastIters = (globalStartEpoch-1) * epochSize

    # nums of left epoches
    leftEpoches = totalEpoches - pastEpoches + 1
    # nums of left iters
    leftIters = leftEpoches * epochSize

    base_lr = optimizer_config['lr']
    cur_lr = base_lr
    print('##### begin train ######')
    currentIter = 0   
 
    for epoch in range(leftEpoches):
        currentEpoch = epoch + pastEpoches

        # 仅用于打印
        loss_sum = 0.0 
        loss_ins = 0.0 
        loss_cate = 0.0

        # 每个epoch更新base_lr
        lrOri = cfg.optimizer['lr']
        if currentEpoch >= totalEpoches:
            raise NotImplementedError("train epoch is done!")
        elif currentEpoch < cfg.lr_config["step"][0]:
            base_lr = lrOri
        else:
            coeff = 0.1
            for ii in range(numStages-1):
                if cfg.lr_config["step"][ii] <= globalStartEpoch < cfg.lr_config["step"][ii+1]:
                    for _ in range(ii):
                        coeff *= 0.1
                    break
            base_lr = lrOri * coeff
        print("base_lr: ", base_lr)
        
        for j, data in enumerate(torchdataLoader):
            if cfg.lr_config['warmup'] is not None and pastIters < cfg.lr_config['warmup_iters']:
                cur_lr = get_warmup_lr(pastIters, cfg.lr_config['warmup_iters'],
                                        optimizer_config['lr'], cfg.lr_config['warmup_ratio'],
                                        cfg.lr_config['warmup'])
            else:
                cur_lr = base_lr
            set_lr(optimizer, cur_lr)
            last_time = time.time()

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
            loss_sum = loss_sum + losses.cpu().item()
            loss_ins = loss_ins + loss['loss_ins'].cpu().item()
            loss_cate = loss_cate + loss['loss_cate'].cpu().item()

            optimizer.zero_grad()
            losses.backward()

            if torch.isfinite(losses).item():
                grad_norm = clip_grads(model.parameters())  #梯度平衡
                optimizer.step()
            else:
                NotImplementedError("loss type error!can't backward!")

            leftIters -= 1
            use_time = time.time() - last_time
            pastIters += 1
            currentIter += 1

            if j%50 == 0 and j != 0:
                left_time = use_time * (leftIters-currentIter)
                left_minut = left_time / 60.0
                left_hours = left_minut / 60.0
                left_day = left_hours // 24
                left_hour = left_hours % 24

                out_srt = 'epoch:[' + str(epoch + pastEpoches) + ']/[' + str(totalEpoches) + '],';
                out_srt = out_srt + '[' + str(j) + ']/' + str(epochSize) + '], left_time:' + str(left_day) + 'days,' + format(left_hour,'.2f') + 'h,'
                print(out_srt, "loss: ", format(loss_sum/50.0,'.4f'), ' loss_ins:', format(loss_ins/50.0,'.4f'), "loss_cate:", format(loss_cate/50.0,'.4f'), "lr:",  format(cur_lr,'.8f'), "base_lr:",  format(base_lr,'.8f'))
                loss_sum = 0.0 
                loss_ins = 0.0 
                loss_cate = 0.0

        leftEpoches -= 1

        save_name = "./weights/solov2_" + cfg.backbone.name + "_epoch_" + str(epoch + pastEpoches) + ".pth"
        model.save_weights(save_name)        

if __name__ == '__main__':
    train(globalStartEpoch=cfg.epoch_iters_start, totalEpoches = cfg.total_epoch)   #设置本次训练的起始epoch
