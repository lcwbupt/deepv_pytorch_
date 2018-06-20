# -*- coding: utf-8 -*-
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import IoG, decode_new
import sys


class RepulsionLoss(nn.Module):

    def __init__(self, use_gpu=True, sigma=0.):
        super(RepulsionLoss, self).__init__()
        self.use_gpu = use_gpu
        self.variance = cfg['variance']
        self.sigma = sigma
        
        
    # TODO 
    def smoothln(self, x, sigma=0.):        
        # −ln(1−x), for x <= sigma
        # (x-sigma)/(1-sigma)-ln(1-sigma), for x > sigma
        mask_1 = x<=sigma
        mask_2 = x>sigma
        tensor1 = torch.masked_select(x, mask_1)
        tensor1 = -torch.log(1-tensor1+1e-10)
        tensor2 = torch.masked_select(x, mask_2)
        tensor2 = (tensor2-sigma)/(1-sigma)-torch.log(1-sigma)
        return torch.sum(tensor1) + torch.sum(tensor2)
    
    def smoothl1(self, y_pred, y_true):
#         sigma_squared = smooth ** 2
#         regression_diff = y_pred - y_true
#         regression_diff = tf.abs(regression_diff)
#         return tf.where(
#             tf.less(regression_diff, 1.0 / sigma_squared),
#             0.5 * sigma_squared * tf.pow(regression_diff, 2),
#             regression_diff - 0.5 / sigma_squared
#         )
        regression_diff = y_pred - y_true
        regression_diff = torch.abs(regression_diff)
        mask_1 = regression_diff<1.0
        mask_2 = regression_diff>=1.0
        tensor1 = torch.masked_select(regression_diff, mask_1)
        tensor1 = 0.5*tensor1*tensor1
        tensor2 = torch.masked_select(regression_diff, mask_2)
        tensor2 = tensor2 - 0.5
        return torch.sum(tensor1) + torch.sum(tensor2)
        

#     def forward(self, loc_data, ground_data, prior_data):
          
#         decoded_boxes = decode_new(loc_data, Variable(prior_data.data, requires_grad=False), self.variance)
#         iog = IoG(ground_data, decoded_boxes)
# #         sigma = 1
# #         loss = torch.sum(-torch.log(1-iog+1e-10))  
#         # sigma = 0
#         loss = torch.sum(iog)          
#         return loss

    def forward(self, loc_data, ground_data, prior_data):
        decoded_boxes = decode_new(loc_data, Variable(prior_data.data, requires_grad=False), self.variance)
        iog = IoG(ground_data, decoded_boxes)
        mask_1 = x<=sigma
        mask_2 = x>sigma
        tensor1 = torch.masked_select(x, mask_1)
        tensor1 = -torch.log(1-tensor1+1e-10)
        tensor2 = torch.masked_select(x, mask_2)
        tensor2 = (tensor2-sigma)/(1-sigma)-torch.log(1-sigma)
        loss = torch.sum(tensor1) + torch.sum(tensor2) 
        
        return loss
