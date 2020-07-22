import torch
import torch.nn as nn
import cv2
import numpy as np



class FocalLoss_fcos(nn.Module):
    def __init__(self):

        super(FocalLoss_fcos, self).__init__()

    def forward(self, classifications, regressions, uselss_inf ,annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        regressions = regressions.sigmoid()
        for j in range(batch_size):
            annotation = annotations[j]
            anno_bbox  = annotation[:, :4]
            anno_cls   = annotation[:, 5:]
            effective_indices = annotation[:,4] == 1. 
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]


            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            regression     = torch.clamp(regression, 1e-4, 1.0 - 1e-4)
            
            
            # img have object
            if effective_indices.sum() >0:

                # compute the loss for classification
                alpha_factor = torch.ones_like(anno_cls) * alpha
                if torch.cuda.is_available():
                    alpha_factor = alpha_factor.cuda()

                alpha_factor = torch.where(torch.eq(anno_cls, 1.), alpha_factor, 1. - alpha_factor)
                focal_weight = torch.where(torch.eq(anno_cls, 1.), 1. - classification, classification)
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                bce = -(anno_cls * torch.log(classification) + (1.0 - anno_cls) * torch.log(1.0 - classification))
                cls_loss = focal_weight * bce
                
                classification_losses.append(cls_loss.sum() / cls_loss.shape[0])
                
                regression_diff_true = torch.abs(anno_bbox[effective_indices] - regression[effective_indices])
                ineffective_indices = effective_indices = annotation[:,4] != 1. 
                regression_diff_false = torch.abs(regression[ineffective_indices] - 0)
                
                beta = 0.7
                # regression_loss = torch.where(
                #         torch.le(regression_diff_true, 1.0 / 9.0),
                #         0.5 * 9.0 * torch.pow(regression_diff_true, 2),
                #         regression_diff_true - 0.5 / 9.0
                #     )
                
                regression_losses.append(beta * regression_diff_true.mean() + (1 - beta) * regression_diff_false.mean())
            
            # img doesnot have object
            else:
                if torch.cuda.is_available():
                    
                    alpha_factor = torch.ones_like(classification) * alpha
                    alpha_factor = alpha_factor.cuda()
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                    
                    bce = -(torch.log(1.0 - classification))
                    
                    cls_loss = focal_weight * bce
                    
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    classification_losses.append(cls_loss.sum())
                else:
                    
                    alpha_factor = torch.ones_like(classification) * alpha
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                    
                    bce = -(torch.log(1.0 - classification))
                    
                    cls_loss = focal_weight * bce
                    
                    regression_losses.append(torch.tensor(0).to(dtype))
                    classification_losses.append(cls_loss.sum())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
               torch.stack(regression_losses).mean(dim=0, keepdim=True)
            
