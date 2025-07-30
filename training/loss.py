# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence
from typing import Callable

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------

@persistence.persistent_class
class MonotonicEDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.lambda_mean = (-2) * self.P_mean
        self.lambda_std = 2 * self.P_std
        self.sigma_data = sigma_data
        self.argmax_lambda = torch.tensor(-3.30778)
        self.weight_lambda_max = ((torch.exp(-self.argmax_lambda) + self.sigma_data ** 2) /
                                  (torch.exp(-self.argmax_lambda) * self.sigma_data ** 2))
        
    def normal_pdf(self, x, mu=0, sigma=1):
        return (1 / (sigma * (2 * torch.pi) ** 0.5)) * torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    # argmax_lambda w(lambda) = -3.30778
    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        lambda_ = (-1) * (sigma ** 2).log()
        weight = torch.where(
            lambda_ < self.argmax_lambda,
            (self.weight_lambda_max * torch.exp(-self.argmax_lambda) * 
            self.normal_pdf(self.argmax_lambda, self.lambda_mean, self.lambda_std) / 
            (torch.exp(-lambda_) * self.normal_pdf(lambda_, self.lambda_mean, self.lambda_std)) 
            ),
            (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        )
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------

@persistence.persistent_class
class ULoss:
    def __init__(self, u: Callable[[torch.Tensor], torch.Tensor],
                 alpha: Callable[[torch.Tensor], torch.Tensor],
                 sigma: Callable[[torch.Tensor], torch.Tensor],
                 t_min: float = 5e-3,
                 t_max: float = 1 - 5e-3):
        self.u = u
        self.alpha = alpha
        self.sigma = sigma
        self.t_min = t_min
        self.t_max = t_max

    def __call__(self, net, images, labels=None, augment_pipe=None):
        t = torch.rand([images.shape[0]], device=images.device) * (self.t_max - self.t_min) + self.t_min
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        eps = torch.randn_like(y) * self.sigma(t).to(torch.float32).reshape(-1, 1, 1, 1)
        D_yn = net(y * self.alpha(t) + eps, t, labels, augment_labels=augment_labels)
        loss = (D_yn - eps * self.u(t))**2
        return loss
#----------------------------------------------------------------------------

@persistence.persistent_class
class ConstantULoss(ULoss):
    def __init__(self):
        t_min = torch.tensor(5e-3)
        t_max = torch.tensor(1 - 5e-3)

        # constant u to prevent imaginary values
        self.u_constant = max(-self._d_lambda(t_min), -self._d_lambda(t_max)) + 1e-3

        super().__init__(self._u_func, self._alpha_func, self._sigma_func, t_min, t_max)

    def _alpha_func(self, t):
        return torch.cos(t * torch.pi / 2)
    
    def _sigma_func(self, t):
        return torch.sin(t * torch.pi / 2)
    
    def _d_lambda(self, t):
        # https://www.wolframalpha.com/input?i2d=true&i=Divide%5Bd%2Cdt%5D2+*+log%5C%2840%29Divide%5Bcos%5C%2840%29t*Divide%5Bpi%2C2%5D%5C%2841%29%2Csin%5C%2840%29t*Divide%5Bpi%2C2%5D%5C%2841%29%5D%5C%2841%29
        return -torch.pi / (torch.cos(t * torch.pi / 2) * torch.sin(t * torch.pi / 2))
    
    def _u_func(self, t):
        # constant u to prevent imaginary values
        return torch.full_like(t, self.u_constant)
#----------------------------------------------------------------------------