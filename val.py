# Author: Zylo117

import numpy as np
import os
import config
import torch
from tqdm import tqdm

from pycocotools.coco import COCO


from core.EfficientDet import EfficientDetBackbone
from utils.utils import parse_gt_rec
from utils.eval_utils import evaluate_coco, evaluate_voc, _eval_coco


compound_coef = config.compound_coef
use_cuda = config.eval_use_cuda
gpu = config.eval_gpu
use_float16 = config.eval_use_float16

weights_path = config.eval_weight_path

num_classes  = len(config.obj_list)

anchor_scales = config.anchors_scales
anchors_ratios = config.anchors_ratios

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]


def coco_eval(model):
    SET_NAME = config.val_set
    data_path = config.data_path
    dataset_name = config.dataset_name
    VAL_GT = '{0}/{1}/annotations/instances_{2}.json'.format(data_path, dataset_name, SET_NAME)
    VAL_IMGS_DIR = '{0}/{1}/{2}/'.format(data_path, dataset_name, SET_NAME)
    MAX_IMAGES = 10000
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]
    input_size = input_sizes[config.compound_coef] if config.force_input_size is None else config.force_input_size
    
    results_json_path = ".{0}_results.json".format(SET_NAME)
    evaluate_coco(VAL_IMGS_DIR, results_json_path, image_ids, coco_gt, model, input_size, config)

    _eval_coco(coco_gt, image_ids, results_json_path)
    if os.path.exists(results_json_path):
        os.remove(results_json_path)

def voc_eval(model):
    val_data_path = config.data_path + "/" + config.dataset_name + "/val.txt"
    gt_dict, img_pathes = parse_gt_rec(val_data_path)
    
    results = evaluate_voc(gt_dict, img_pathes, model, input_sizes[config.compound_coef], config)
    for i, each in enumerate(results):
        print("class:{:15s}Precision:{:.3f}\tRecall:{:.3f}\tAP:{:.3f}".format(config.obj_list[i], each[0], each[1],each[2]))

if __name__ == '__main__':
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=num_classes,
                                     anchor_free_mode=config.anchor_free_mode,
                                     ratios=eval(anchors_ratios), scales=eval(anchor_scales))
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model.cuda(gpu)

        if use_float16:
            model.half()
    
    if(config.dataset_name == "voc"):
        voc_eval(model)
    else:
        coco_eval(model)
