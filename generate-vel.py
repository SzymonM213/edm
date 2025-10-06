# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from er_sde_solver import ER_SDE_Solver
from torch_utils import distributed as dist

@torch.no_grad()
def eta_constant(t):
    return torch.zeros_like(t)

@torch.no_grad()
def eta_discrete(u, alpha_s, alpha_t, sigma_s, sigma_t):
    gamma_s = ((alpha_s / sigma_s) * (sigma_t / alpha_t))**2

    numerator = gamma_s - 1
    assert u**2 + 1 - gamma_s >= 0, "negative number squared"
    denominator = torch.sqrt(gamma_s * u**2) - torch.sqrt(u**2 + 1 - gamma_s)
    return numerator / denominator

@torch.no_grad()
def eta_continous(net, s):
    lambda_s_prime = net.d_lambda(s)
    eta_s = net.u_constant - torch.sqrt(net.u_constant**2 + lambda_s_prime)
    return eta_s


#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).
def our_sampler(
    net,
    latents,
    class_labels = None,
    randn_like=torch.randn_like,
    num_steps = 5, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,):

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t_min = torch.tensor(5e-3)
    t_max = torch.tensor(1 - 5e-3)

    z_t = latents.to(device)
    ts = torch.linspace(t_max, t_min, steps=num_steps + 1, device=device)

    for i in tqdm.trange(num_steps, desc="Sampling"):
        t = ts[i].unsqueeze(0)
        s = ts[i + 1].unsqueeze(0)

        alpha_t = net.alpha(t)
        sigma_t = net.sigma(t)
        alpha_s = net.alpha(s)
        sigma_s = net.sigma(s)

        u_s = net.u(s) * 12
        u_t = net.u(t)

        # 1. eta = 1
        # 2. eta z pliku not_main.pdf
        # 3. eta z pliku Pokarowski_Heidelberg2025.pdf
        # eta_s = eta_constant(t)
        # eta_s = eta_discrete(u_s, alpha_t, alpha_s, sigma_s, sigma_t)
        eta_s = torch.tensor(0).to(device)
        # eta_s = eta_continous(net, s)

        eps_scaled = net(z_t, t, class_labels)
        eps_pred = z_t - eps_scaled / u_t.reshape(-1,1,1,1)

        coeff_eps = sigma_s * torch.sqrt(1 - eta_s**2) - alpha_s * (sigma_t / alpha_t)
        coeff_z = alpha_s / alpha_t

        eps = torch.randn_like(z_t) if i < num_steps - 1 else 0

        z_t = coeff_eps.reshape(-1,1,1,1) * eps_pred + coeff_z.reshape(-1,1,1,1) * z_t + sigma_s.reshape(-1,1,1,1) * eta_s.reshape(-1,1,1,1) * eps
    return z_t

@torch.no_grad()
def vel_ode_sampler(
    net,
    latents,
    class_labels=None,
    randn_like=torch.randn_like,
    num_steps=18,
    t_min=None,
    t_max=None,
    method: str = "heun",  # "euler" or "heun"
):
    """
    Continuous-time probability-flow ODE sampler for a network that predicts x_t − ε_t (scaled by u(t)).

    Assumptions:
    - net(x, t, labels) ≈ u(t) * (x_t − ε_t)
    - u(t) is exposed via net.u(t)
    - alpha(t), sigma(t) are exposed via net.alpha(t), net.sigma(t)

    ODE drift: dz/dt = (alpha'(t)/alpha(t)) * z + [sigma'(t) − (alpha'(t)/alpha(t)) * sigma(t)] * ε̂(z,t)
    where ε̂ is recovered from the x−ε predictor: ε̂ = z − (net(x,t)/u(t)).
    """
    assert method in ("euler", "heun")

    device = latents.device
    dtype64 = torch.float64

    # Time range.
    t_min = net.t_min.to(device=device, dtype=dtype64) if t_min is None else torch.as_tensor(t_min, device=device, dtype=dtype64)
    t_max = net.t_max.to(device=device, dtype=dtype64) if t_max is None else torch.as_tensor(t_max, device=device, dtype=dtype64)

    # Discretize from t_max -> t_min (descending time).
    ts = torch.linspace(t_max, t_min, steps=num_steps + 1, device=device, dtype=dtype64)

    # State.
    z = latents.to(dtype64)

    for i in range(num_steps):
        t = ts[i].unsqueeze(0)      # [1]
        s = ts[i + 1].unsqueeze(0)  # [1]

        # Schedules and their derivatives.
        alpha_t = net.alpha(t).to(dtype64)
        sigma_t = net.sigma(t).to(dtype64)
        alpha_p = net.d_alpha(t).to(dtype64)
        sigma_p = net.d_sigma(t).to(dtype64)

        # Recover ε̂ from x−ε prediction.
        out = net(z.to(torch.float32), t.to(torch.float32), class_labels).to(dtype64)  # ≈ u(t)*(z−ε)
        u_t = net.u(t).to(dtype64).reshape(-1, 1, 1, 1)
        eps_hat = z - out / u_t

        # Drift f(z,t).
        a_ratio = (alpha_p / alpha_t).reshape(-1, 1, 1, 1)
        coeff_eps = (sigma_p - (alpha_p / alpha_t) * sigma_t).reshape(-1, 1, 1, 1)
        f_t = a_ratio * z + coeff_eps * eps_hat

        h = (s - t).reshape(1, 1, 1, 1)  # negative step
        z_euler = z + h * f_t

        if method == "euler" or i == num_steps - 1:
            z = z_euler
        else:
            # Heun (predictor-corrector).
            alpha_s = net.alpha(s).to(dtype64)
            sigma_s = net.sigma(s).to(dtype64)
            alpha_p_s = net.d_alpha(s).to(dtype64)
            sigma_p_s = net.d_sigma(s).to(dtype64)

            out_s = net(z_euler.to(torch.float32), s.to(torch.float32), class_labels).to(dtype64)
            u_s = net.u(s).to(dtype64).reshape(-1, 1, 1, 1)
            eps_hat_s = z_euler - out_s / u_s

            a_ratio_s = (alpha_p_s / alpha_s).reshape(-1, 1, 1, 1)
            coeff_eps_s = (sigma_p_s - (alpha_p_s / alpha_s) * sigma_s).reshape(-1, 1, 1, 1)
            f_s = a_ratio_s * z_euler + coeff_eps_s * eps_hat_s

            z = z + h * 0.5 * (f_t + f_s)

    return z.to(torch.float32)

@torch.no_grad()
def vel_sde_sampler(
    net,
    latents,
    class_labels=None,
    randn_like=torch.randn_like,
    num_steps=18,
    t_min=None,
    t_max=None,
):
    """
    Euler–Maruyama integrator for the SDE:

        dZ_t = [ (α'_t/α_t) Z_t + σ_t (η_t^2 − λ'_t)/2 · ε_t(Z_t) ] dt + σ_t η_t dW_t,

    where the network predicts x_t − ε_t scaled by u(t): net(x,t) ≈ u(t)·(x_t − ε_t).
    We recover ε̂_t = z − net(z,t)/u(t) and use it in the drift. η_t is obtained from the
    model if available; otherwise we fall back to a constant schedule of 1.0 and allow
    the user to globally scale it with the `eta` argument (eta=0 turns off noise and
    removes the η_t^2 contribution from the drift as well).
    """
    device = latents.device
    dtype64 = torch.float64

    # Time range.
    t_min = net.t_min.to(device=device, dtype=dtype64) if t_min is None else torch.as_tensor(t_min, device=device, dtype=dtype64)
    t_max = net.t_max.to(device=device, dtype=dtype64) if t_max is None else torch.as_tensor(t_max, device=device, dtype=dtype64)

    ts = torch.linspace(t_max, t_min, steps=num_steps + 1, device=device, dtype=dtype64)

    # Fallback finite-difference for derivatives if the net doesn't expose them.
    def central_diff(fn, t, eps=1e-4):
        tp = torch.clamp(t + eps, min=float(t_min), max=float(t_max))
        tm = torch.clamp(t - eps, min=float(t_min), max=float(t_max))
        if torch.allclose(tp, tm):
            tp = torch.clamp(t + 2*eps, min=float(t_min), max=float(t_max))
            tm = t
        return (fn(tp) - fn(tm)) / (tp - tm)

    z = latents.to(dtype64)

    c = max((-net.d_lambda(t))**(1/2) / net.u(t) for t in ts[:-1])
    print(f"Scaling factor for u(t): {c}")

    for i in range(num_steps):
        t = ts[i].unsqueeze(0)
        s = ts[i + 1].unsqueeze(0)
        h = (s - t).reshape(-1, 1, 1, 1)  # negative step

        # Schedules and required derivatives.
        alpha_t = net.alpha(t).to(dtype64)                         # α_t
        sigma_t = net.sigma(t).to(dtype64)                         # σ_t
        alpha_p = (net.d_alpha(t).to(dtype64)
                   if hasattr(net, 'd_alpha') else central_diff(net.alpha, t).to(dtype64))  # α'_t
        # λ'_t: required by the drift term.
        if hasattr(net, 'd_lambda'):
            lambda_p = net.d_lambda(t).to(dtype64)
        else:
            # If not available, assume 0 (no λ dynamics).
            lambda_p = torch.zeros_like(alpha_t)

        # Recover ε̂_t from x−ε predictor scaled by u(t).
        out = net(z.to(torch.float32), t.to(torch.float32), class_labels).to(dtype64)
        u_t = net.u(t).to(dtype64).reshape(-1, 1, 1, 1)
        eps_hat = z - out / u_t

        # η_t schedule from the model if available; otherwise default to 1.0.
        # Heuristic when only u(t) and λ'_t are present.
        # u_now = u_t.reshape(-1)  # already dtype64
        u_t *= c
        assert u_t**2 + lambda_p.reshape(-1) >= 0, "u(t)^2 + λ'(t) must be non-negative to derive η_t"
        eta_t_sched = u_t - torch.sqrt(torch.clamp(u_t**2 + lambda_p.reshape(-1), min=0))
        eta_t_sched = eta_t_sched.reshape(-1, 1, 1, 1)
        # print(f"eta_t_sched: {eta_t_sched}")

        eta_eff = eta_t_sched.reshape(-1, 1, 1, 1)

        # Drift according to the provided SDE.
        a_ratio = (alpha_p / alpha_t).reshape(-1, 1, 1, 1)
        drift_coeff_eps = (sigma_t * (eta_eff.reshape(-1) ** 2 - lambda_p) / 2).reshape(-1, 1, 1, 1)
        f_bar = a_ratio * z + drift_coeff_eps * eps_hat

        # Diffusion amplitude.
        g_bar = (sigma_t.reshape(-1, 1, 1, 1) * eta_eff)

        # Euler–Maruyama step.
        noise = randn_like(z) if i < num_steps - 1 else 0
        z = z + h * f_bar + g_bar * torch.sqrt(torch.abs(h)) * noise

    return z.to(torch.float32)
# Heun (predictor-corrector) SDE sampler for velocity models
def vel_sde_sampler_heun(
    net,
    latents,
    class_labels=None,
    randn_like=torch.randn_like,
    num_steps=100,
    t_min=None,
    t_max=None,
):
    """
    Heun (predictor-corrector) integrator for the SDE:

        dZ_t = [ (α'_t/α_t) Z_t + σ_t (η_t^2 − λ'_t)/2 · ε_t(Z_t) ] dt + σ_t η_t dW_t,

    where the network predicts x_t − ε_t scaled by u(t): net(x,t) ≈ u(t)·(x_t − ε_t).
    We recover ε̂_t = z − net(z,t)/u(t) and use it in the drift. η_t is obtained from the
    model if available; otherwise we fall back to a constant schedule of 1.0 and allow
    the user to globally scale it with the `eta` argument (eta=0 turns off noise and
    removes the η_t^2 contribution from the drift as well).
    """
    device = latents.device
    dtype64 = torch.float64

    # Time range.
    t_min_ = net.t_min.to(device=device, dtype=dtype64) if t_min is None else torch.as_tensor(t_min, device=device, dtype=dtype64)
    t_max_ = net.t_max.to(device=device, dtype=dtype64) if t_max is None else torch.as_tensor(t_max, device=device, dtype=dtype64)

    ts = torch.linspace(t_max_, t_min_, steps=num_steps + 1, device=device, dtype=dtype64)

    z = latents.to(dtype64)

    for i in range(num_steps):
        t = ts[i].unsqueeze(0)
        s = ts[i + 1].unsqueeze(0)
        h = (s - t).reshape(-1, 1, 1, 1)  # negative step

        # Schedules and required derivatives at t
        alpha_t = net.alpha(t).to(dtype64)
        sigma_t = net.sigma(t).to(dtype64)
        alpha_p = net.d_alpha(t).to(dtype64)
        if hasattr(net, 'd_lambda'):
            lambda_p = net.d_lambda(t).to(dtype64)
        else:
            lambda_p = torch.zeros_like(alpha_t)

        out = net(z.to(torch.float32), t.to(torch.float32), class_labels).to(dtype64)
        u_t = net.u(t).to(dtype64).reshape(-1, 1, 1, 1)
        eps_hat = z - out / u_t

        u_t_eff = u_t * 12
        assert u_t_eff**2 + lambda_p.reshape(-1) >= 0, "u(t)^2 + λ'(t) must be non-negative to derive η_t"
        eta_t_sched = u_t_eff - torch.sqrt(torch.clamp(u_t_eff**2 + lambda_p.reshape(-1), min=0))
        eta_t_sched = eta_t_sched.reshape(-1, 1, 1, 1)
        eta_eff = eta_t_sched.reshape(-1, 1, 1, 1)

        a_ratio = (alpha_p / alpha_t).reshape(-1, 1, 1, 1)
        drift_coeff_eps = (sigma_t * (eta_eff.reshape(-1) ** 2 - lambda_p) / 2).reshape(-1, 1, 1, 1)
        f_bar = a_ratio * z + drift_coeff_eps * eps_hat
        g_bar = (sigma_t.reshape(-1, 1, 1, 1) * eta_eff)

        noise = randn_like(z) if i < num_steps - 1 else 0
        z_euler = z + h * f_bar + g_bar * torch.sqrt(torch.abs(h)) * noise

        if i == num_steps - 1:
            z = z_euler
        else:
            # Heun (predictor-corrector) step
            # Schedules and required derivatives at s
            alpha_s = net.alpha(s).to(dtype64)
            sigma_s = net.sigma(s).to(dtype64)
            alpha_p_s = net.d_alpha(s).to(dtype64)
            if hasattr(net, 'd_lambda'):
                lambda_p_s = net.d_lambda(s).to(dtype64)
            else:
                lambda_p_s = torch.zeros_like(alpha_s)

            out_s = net(z_euler.to(torch.float32), s.to(torch.float32), class_labels).to(dtype64)
            u_s = net.u(s).to(dtype64).reshape(-1, 1, 1, 1)
            eps_hat_s = z_euler - out_s / u_s

            u_s_eff = u_s * 12
            assert u_s_eff**2 + lambda_p_s.reshape(-1) >= 0, "u(s)^2 + λ'(s) must be non-negative to derive η_s"
            eta_s_sched = u_s_eff - torch.sqrt(torch.clamp(u_s_eff**2 + lambda_p_s.reshape(-1), min=0))
            eta_s_sched = eta_s_sched.reshape(-1, 1, 1, 1)
            eta_eff_s = eta_s_sched.reshape(-1, 1, 1, 1)

            a_ratio_s = (alpha_p_s / alpha_s).reshape(-1, 1, 1, 1)
            drift_coeff_eps_s = (sigma_s * (eta_eff_s.reshape(-1) ** 2 - lambda_p_s) / 2).reshape(-1, 1, 1, 1)
            f_bar_s = a_ratio_s * z_euler + drift_coeff_eps_s * eps_hat_s
            g_bar_s = (sigma_s.reshape(-1, 1, 1, 1) * eta_eff_s)

            # Average drift and diffusion
            f_bar_avg = 0.5 * (f_bar + f_bar_s)
            g_bar_avg = 0.5 * (g_bar + g_bar_s)

            z = z + h * f_bar_avg + g_bar_avg * torch.sqrt(torch.abs(h)) * noise
            # z = z + h * f_bar_avg

    return z.to(torch.float32)
    
def discrete_sampler(
    net,
    latents,
    class_labels=None,
    randn_like=torch.randn_like,
    num_steps=100,
    t_min=None,
    t_max=None,
):
    device = latents.device
    dtype64 = torch.float64

    t_min_ = net.t_min.to(device=device, dtype=dtype64) if t_min is None else torch.as_tensor(t_min, device=device, dtype=dtype64)
    t_max_ = net.t_max.to(device=device, dtype=dtype64) if t_max is None else torch.as_tensor(t_max, device=device, dtype=dtype64)

    ts = torch.linspace(t_max_, t_min_, steps=num_steps + 1, device=device, dtype=dtype64)

    z = latents.to(dtype64)

    eta_coeff = max((((net.alpha(ts[i+1]) / net.sigma(ts[i+1])) * (net.sigma(ts[i]) / net.alpha(ts[i])))**2 - 1)**(1/2) 
                    / net.u(ts[i+1]) for i in range(len(ts)-1))


    for i in range(num_steps):
        t = ts[i].unsqueeze(0)
        s = ts[i + 1].unsqueeze(0)

        alpha_t = net.alpha(t)
        sigma_t = net.sigma(t)
        alpha_s = net.alpha(s)
        sigma_s = net.sigma(s)

        u_s = net.u(s) * eta_coeff
        u_t = net.u(t)

        eps_scaled = net(z, t, class_labels)
        eps_pred = z - eps_scaled / u_t.reshape(-1, 1, 1, 1)

        if i < num_steps:
            eta_s = eta_discrete(u_s, alpha_t, alpha_s, sigma_s, sigma_t)
            # eta_s = torch.tensor(0).to(device)
            coeff_eps = sigma_s * torch.sqrt(1 - eta_s**2) - alpha_s * (sigma_t / alpha_t)
            coeff_z = alpha_s / alpha_t

            eps = randn_like(z) if i < num_steps - 1 else 0
            z = coeff_eps.reshape(-1,1,1,1) * eps_pred + coeff_z.reshape(-1,1,1,1) * z + sigma_s.reshape(-1,1,1,1) * eta_s.reshape(-1,1,1,1) * eps
        else:
            z = alpha_s / alpha_t * z + (sigma_s - alpha_s * sigma_t / alpha_t) * eps_pred

    return z

def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.

def ablation_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)
@click.option('--eta', 'eta',              help='Stochasticity for continuous-time sampling (0=ODE, >0=SDE)',       type=click.FloatRange(min=0), default=0.0, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun|taylor',                          type=click.Choice(['euler', 'heun', 'taylor']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))

def main(network_pkl, outdir, subdirs, seeds, class_idx, max_batch_size, device=torch.device('cuda'), **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    solver = ER_SDE_Solver(sde_type='ve', model_prediction_type='x_start')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
        if all(hasattr(net, attr) for attr in ('alpha', 'sigma', 'u')):
            ct_allowed = {'num_steps', 't_min', 't_max'}
            ct_kwargs = {k: v for k, v in sampler_kwargs.items() if k in ct_allowed}
            # images = vel_sde_sampler_heun(net, latents, class_labels, randn_like=rnd.randn_like, **ct_kwargs)
            # images = discrete_sampler(net, latents, class_labels, randn_like=rnd.randn_like, **ct_kwargs)
            images = vel_sde_sampler(net, latents, class_labels, randn_like=rnd.randn_like, **ct_kwargs)

            # # Use uvel_heun from ER_SDE_Solver
            # num_steps = sampler_kwargs.get('num_steps', 18)
            # t_min = getattr(net, 't_min', torch.tensor(5e-3))
            # t_max = getattr(net, 't_max', torch.tensor(1 - 5e-3))
            # device = latents.device
            # dtype = torch.float64
            # ts = torch.linspace(t_max, t_min, steps=num_steps + 1, device=device, dtype=dtype)
            # alphas = net.alpha(ts)
            # sigmas = net.sigma(ts)
            # images = solver.vp_3_order_taylor(net, latents.to(dtype), alphas.to(dtype), sigmas.to(dtype), ts.to(dtype))
        else:
            sampler_fn = ablation_sampler if have_ablation_kwargs else edm_sampler
            images = sampler_fn(net, latents, class_labels, randn_like=rnd.randn_like, **sampler_kwargs)

        # Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
