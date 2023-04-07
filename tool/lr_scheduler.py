import logging
import math

import torch
from torch.optim.lr_scheduler import LambdaLR

class Transformer_LRScheduler(torch.optim.lr_scheduler.LambdaLR):
    """
        This corresponds to increasing the learning rate linearly for the firstwarmup_stepstraining steps,
        and decreasing it thereafter proportionally to the inverse square root of the step number.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        def lr_lambda(step):
            ratio = float(step) / float(warmup_steps)
            coef = ratio if ratio < 1.0 else math.sqrt(1 / ratio)
            coef = 1e-8 if coef <= 0 else coef
            return coef

        super(Transformer_LRScheduler, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)
