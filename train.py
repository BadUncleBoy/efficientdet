# original author: signatrix
# adapted from https://github.com/signatrix/efficientdet/blob/master/train.py
# modified by Zylo117

import datetime
import os
import argparse
import traceback

from tensorboardX import SummaryWriter
import numpy as np
from tqdm.autonotebook import tqdm

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CocoDataset, PascalVocDataset, Resizer, Normalizer, Augmenter, collater
from core.EfficientDet import EfficientDetBackbone
from core.loss import FocalLoss, FocalLoss_fcos

from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, init_weights
import config
os.environ["CUDA_VISIBLE_DEVICES"] = '3,4,5,7'
class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        if config.anchor_free_mode:
            self.criterion = FocalLoss_fcos()
        else:
            self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss


def train():
    if config.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    config.saved_path = config.saved_path + '/{0}/'.format(config.dataset_name)
    config.log_path = config.log_path + '/{0}/'.format(config.dataset_name)
    os.makedirs(config.log_path, exist_ok=True)
    os.makedirs(config.saved_path, exist_ok=True)

    training_params = {'batch_size': config.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': config.num_workers}

    val_params = {'batch_size': config.batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': config.num_workers}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    if("coco" in config.dataset_name):
        DS = CocoDataset
    else:
        DS = PascalVocDataset
    training_set = DS(root_dir=os.path.join(config.data_path, config.dataset_name), set=config.train_set,
                               img_size=input_sizes[config.compound_coef], anchor_free_mode=config.anchor_free_mode,
                               transform=transforms.Compose([Normalizer(mean=config.mean, std=config.std),
                                                             Augmenter(),
                                                             Resizer(input_sizes[config.compound_coef])]))
    training_generator = DataLoader(training_set, **training_params)

    val_set = DS(root_dir=os.path.join(config.data_path, config.dataset_name), set=config.val_set,
                               img_size=input_sizes[config.compound_coef], anchor_free_mode=config.anchor_free_mode,
                               transform=transforms.Compose([Normalizer(mean=config.mean, std=config.std),
                                                             Resizer(input_sizes[config.compound_coef])]))
    val_generator = DataLoader(val_set, **val_params)

    model = EfficientDetBackbone(num_classes=len(config.obj_list), compound_coef=config.compound_coef, load_weights=False,
                                 anchor_free_mode=config.anchor_free_mode,
                                 ratios=eval(config.anchors_ratios), scales=eval(config.anchors_scales))
    
    init_weights(model)
    last_step = 0
    # load last weights
    if config.load_weights:
        # 首先使用init_weights来初始化网络参数，然后再restore，
        # 使得网络中未restore的参数可以正常初始化
        
        if config.pret_weight_path.endswith('.pth'):
            weights_path = config.pret_weight_path
        try:
            model_dict = torch.load(weights_path)
            new_dict   = {}
            for k, v in model_dict.items():
                if 'header' not in k:
                    new_dict[k] = v
            ret = model.load_state_dict(new_dict, strict=False)
        except RuntimeError as e:
            print('[Warning] Ignoring {0}'.format(e))
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print('[Info] loaded pretrained weights: {0},'.format(weights_path))

    if config.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if config.num_gpus > 1 and config.batch_size // config.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    writer = SummaryWriter(config.log_path + '/{0}/'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=config.debug)

    if config.num_gpus > 0:
        model = model.cuda()
        if config.num_gpus > 1:
            model = CustomDataParallel(model, config.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    if config.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), config.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.patience, verbose=True, factor=config.factor, min_lr=config.min_lr)

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(training_generator)

    try:
        for epoch in range(config.num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    imgs = data['img']
                    annot = data['annot']
                    # print(annot)

                    if config.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    optimizer.zero_grad()
                    cls_loss, reg_loss = model(imgs, annot, obj_list=config.obj_list)
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                            step, epoch, config.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                            reg_loss.item(), loss.item()))
                    writer.add_scalars('Loss', {'train':loss}, step)
                    writer.add_scalars('Regression_loss', {'train':reg_loss}, step)
                    writer.add_scalars('Classfication_loss', {'train':cls_loss}, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1

                    if step % config.save_interval == 0 and step > 0:
                        save_checkpoint(model, 'efficientdet-d{0}_{1}_{2}.pth'.format(config.compound_coef,epoch,step))

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue
            scheduler.step(np.mean(epoch_loss))

            if epoch % config.val_interval == 0:
                
                model.eval()
                loss_regression_ls = []
                loss_classification_ls = []
                for iter, data in enumerate(val_generator):
                    with torch.no_grad():
                        imgs = data['img']
                        annot = data['annot']

                        if config.num_gpus == 1:
                            imgs = imgs.cuda()
                            annot = annot.cuda()

                        cls_loss, reg_loss = model(imgs, annot, obj_list=config.obj_list)
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss = cls_loss + reg_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss_classification_ls.append(cls_loss.item())
                        loss_regression_ls.append(reg_loss.item())

                cls_loss = np.mean(loss_classification_ls)
                reg_loss = np.mean(loss_regression_ls)
                loss = cls_loss + reg_loss

                print(
                    'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                        epoch, config.num_epochs, cls_loss, reg_loss, loss))
                writer.add_scalars('Loss', {'val':loss}, step)
                writer.add_scalars('Regression_loss', {'val':reg_loss}, step)
                writer.add_scalars('Classfication_loss', {'val':cls_loss}, step)

                if loss + config.es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch

                    save_checkpoint(model, 'efficientdet-d{0}_{1}_{2}_best_loss.pth'.format(config.compound_coef, epoch, step))

                model.train()
                           
                # Early stopping
                if epoch - best_epoch > config.es_patience > 0:
                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                    break
    except KeyboardInterrupt:
        save_checkpoint(model, 'efficientdet-d{0}_{1}_{2}.pth'.format(config.compound_coef, epoch, step))
        writer.close()
    writer.close()


def save_checkpoint(model, name):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(config.saved_path, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(config.saved_path, name))


if __name__ == '__main__':
    train()
