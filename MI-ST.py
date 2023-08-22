import torch
from .AdversarialInputBase import AdversarialInputAttacker
from typing import Callable, List, Iterable
from attacks.utils import *
from .utils import cosine_similarity
from torch import nn, Tensor
import random
from torchvision import transforms
import numpy as np
import math
from scipy import stats as st

class ZeroCSEAttacker(MI_CommonWeakness):
    def __init__(self,
                 *args,
                 loss_weight: float = 1,
                 cosine_similarity_weight: float = 1,
                 **kwargs):
        super(ZeroCSEAttacker, self).__init__(*args, **kwargs)
        self.loss_weight = loss_weight
        self.cosine_similarity_weight = cosine_similarity_weight

    def get_all_gradient(self, x: Tensor, y: Tensor) -> Tensor:
        result = []
        tmp = 0 ##
        for model in self.models:
            N = x.shape[0] ##
            x.requires_grad_(True)
            pre = model(x.to(model.device)).to(x.device) ##
            loss = self.criterion(pre, y)
            tmp += loss ##
            x.grad = None
            loss.backward() ##
            result.append(x.grad/torch.norm(x.grad.reshape(N, -1), p=2, dim=1).view(N,1,1,1)) ##
            x.grad = None
            x.requires_grad_(False)
        result = torch.stack(result)
        result = torch.mean(result, dim=0)
        print('result:' + str(tmp/6))##
        return result * self.inner_step_size

    def attack(self, x, y, ):
        N = x.shape[0]
        original_x = x.clone()
        inner_momentum = torch.zeros_like(x)
        self.outer_momentum = torch.zeros_like(x)
        if self.random_start:
            x = self.perturb(x)

        for _ in range(self.total_step):
            # --------------------------------------------------------------------------------#
            # pre step
            self.first_term = self.get_all_gradient(x, y)
            # first step
            self.begin_attack(x.clone().detach())
            # x.requires_grad = True
            # logit = 0
            # for model in self.models:
            #     logit += model(x.to(model.device)).to(x.device)
            # loss = self.criterion(logit, y)
            # loss.backward()
            # grad = x.grad
            # if self.TI:
            #     grad = self.ti(grad)
            # x.requires_grad = False
            # if self.targerted_attack:
            #     x += self.reverse_step_size * grad.sign()
            # else:
            #     x -= self.reverse_step_size * grad.sign()
            #     # x -= self.reverse_step_size * grad / torch.norm(grad.reshape(N, -1), p=2, dim=1).view(N, 1, 1, 1)
            # x = clamp(x)
            # x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)
            # --------------------------------------------------------------------------------#
            # second step
            x.grad = None
            # self.begin_attack(x.clone().detach())
            for model in self.models:
                x.requires_grad = True
                aug_x = self.aug_policy(x)
                loss = self.criterion(model(aug_x.to(model.device)), y.to(model.device))
                loss.backward()
                grad = x.grad
                self.grad_record.append(grad)
                x.requires_grad = False
                # update
                if self.TI:
                    grad = self.ti(grad)
                if self.targerted_attack:
                    inner_momentum = self.mu * inner_momentum - grad / torch.norm(grad.reshape(N, -1), p=2, dim=1).view(
                        N, 1, 1, 1)
                    x += self.inner_step_size * inner_momentum
                else:
                    inner_momentum = self.mu * inner_momentum + grad / torch.norm(grad.reshape(N, -1), p=2, dim=1).view(
                        N, 1, 1, 1)
                    x += self.inner_step_size * inner_momentum
                x = clamp(x)
                x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)
            x = self.end_attack(x)
            x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)
        return x

    @torch.no_grad()
    def end_attack(self, now: torch.tensor, ksi=16 / 255 / 5):
        '''
        theta: original_patch
        theta_hat: now patch in optimizer
        theta = theta + ksi*(theta_hat - theta), so:
        theta =(1-ksi )theta + ksi* theta_hat
        '''
        patch = now
        update = (patch - self.original)
        cos_grad = update - self.first_term  # cosine similarity only # 【改动1】
        fake_grad = self.loss_weight * (self.first_term) / self.inner_step_size + self.cosine_similarity_weight * 2 * (cos_grad) / (self.inner_step_size ** 2) # 【改动2】
        self.outer_momentum = self.mu * self.outer_momentum + fake_grad / torch.norm(fake_grad, p=1)
        patch.mul_(0)
        patch.add_(self.original)
        patch.add_(ksi * self.outer_momentum.sign())
        patch = clamp(patch)
        del self.grad_record
        del self.original
        return patch
