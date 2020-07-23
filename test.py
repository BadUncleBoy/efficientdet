# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import config
import torch
from torch.backends import cudnn
from matplotlib import colors

from core.EfficientDet import EfficientDetBackbone
import cv2
import numpy as np

from core.others import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box


compound_coef = config.compound_coef
force_input_size = config.force_input_size  # set None to use default size
img_path = config.img_path
weight_path = config.weight_path

anchor_ratios = config.anchors_ratios
anchor_scales = config.anchors_scales

threshold = config.threshold
iou_threshold = config.iou_threshold

use_cuda = True
use_float16 = False

obj_list = config.obj_list
print("test signal image in:{0}\nimg_path:{1}\nweight_path:{2}".format(config.dataset_name, img_path, weight_path))

color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

if use_cuda:
    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
else:
    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             anchor_free_mode=config.anchor_free_mode,
                             ratios=eval(anchor_ratios), scales=eval(anchor_scales))
try:
    model.load_state_dict(torch.load(weight_path))
except Exception as e:
    print("nnn")
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

with torch.no_grad():
    print("start predicating...")
    features, regression, classification, anchors = model(x)

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    # view
    # start =0
    # for each in [8, 16, 32,64,128]:
    #     ll = (input_size//each) **2
    #     ss=classification[:,start:start+ll,:]
    #     tt
    #     start += ll
    #     n=torch.argmax(ss,dim=-1)
    #     print(np.array(n.view(input_size//each, input_size//each).cpu()))
    out = postprocess(x,
                      anchors, regression, classification,
                      regressBoxes, clipBoxes,
                      threshold, iou_threshold,
                      anchor_free_mode=config.anchor_free_mode)
    
def display(preds, imgs, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            print(x1,y1,x2,y2,obj,score)
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])


        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            predicated_path = 'demo_jpg/{0}_infer.jpg'.format(config.dataset_name)
            cv2.imwrite(predicated_path, imgs[i])
            print("write predicated result in:{0}".format(predicated_path))


out = invert_affine(framed_metas, out)
print("predicating finished")
print(out)
display(out, ori_imgs, imshow=False, imwrite=True)
