# Distilling the Knowledge in a Neural Network
# paper: https://arxiv.org/pdf/1503.02531.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftTarget(nn.Module):
    def __init__(self, T):
        super(SoftTarget, self).__init__()
        self.T = T      # temperature of softmax
    
    def forward(self, student_out, teacher_out):
        loss = F.kl_div(F.log_softmax(student_out/self.T, dim=1),
                        F.softmax(teacher_out/self.T, dim=1),
                        reduction='batchmean') * self.T **2
        return loss