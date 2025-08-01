# your_sampler.py

import torch
import numpy as np
import tqdm
import pickle
import PIL.Image
import dnnlib
import os

@torch.no_grad()
def sample_images_from_model(
    network_pkl,
    dest_path,
    seed=0,
    gridw=8,
    gridh=8,
    num_steps=1000,
    device=torch.device('cuda')
):
    batch_size = gridw * gridh
    torch.manual_seed(seed)

    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)


    z_t = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    ts = torch.linspace(1.0, 0.0, steps=num_steps + 1, device=device)

    # reverse sampling
    for i in tqdm.trange(num_steps, desc="Sampling"):
        t = ts[i].unsqueeze(0)
        s = ts[i + 1].unsqueeze(0)

        alpha_t = net._alpha(t)
        sigma_t = net._sigma(t)
        alpha_s = net._alpha(s)
        sigma_s = net._sigma(s)

        if sigma_s < 1e-4:
            sigma_s = torch.clamp(sigma_s, min=1e-4)
        print(f'alpha_t: {alpha_t}, sigma_t: {sigma_t}, alpha_s: {alpha_s}, sigma_s: {sigma_s}')

        gamma_s = (alpha_s**2 / sigma_s**2) * (sigma_t**2 / alpha_t**2)
        print("gamma_s", gamma_s)
        
        u = net.u_constant 

        # first option doesnt work dunno why
        numerator = gamma_s - 1
        denominator = torch.sqrt(gamma_s * u**2) - torch.sqrt(u**2 + 1 - gamma_s)
        print(f'numerator: {torch.sqrt(gamma_s * u**2)}, denominator: {u**2 + 1 - gamma_s}')
        eta_s = numerator / denominator
        print(f'eta_s: {eta_s}')
        # eta_s = 0eta = torch.zeros_like(t)
        # eta_s = torch.zeros_like(t)
        eta_s = torch.sqrt(1 - 1 / gamma_s)
        print(f'eta_s: {eta_s}')
        # Predicted noise

        eps_scaled = net(z_t, t)
        eps_pred = eps_scaled / net.u_constant.reshape(-1, 1, 1, 1)

        coeff_eps = sigma_s * torch.sqrt(1 - eta_s**2) - alpha_s * (sigma_t / alpha_t)
        coeff_z = alpha_s / alpha_t


        eps = torch.randn_like(z_t) if i < num_steps - 1 else 0

        z_t = coeff_eps.reshape(-1,1,1,1) * eps_pred + coeff_z.reshape(-1,1,1,1) * z_t + sigma_s.reshape(-1,1,1,1) * eta_s.reshape(-1,1,1,1) * eps
   
    save_image_grid(z_t, 'last.png', gridw, gridh)
    t_final = ts[-1].unsqueeze(0)  

    alpha_final = net._alpha(t_final).reshape(-1,1,1,1)
    sigma_final = net._sigma(t_final).reshape(-1,1,1,1)

    # print(f'Final timestep: alpha={alpha_final}, sigma={sigma_final}')

    # eps_pred_final = net(z_t, t_final) / net.u_constant.reshape(-1,1,1,1)
    # print(f'Finnal : { net(z_t, t_final)}')
    # x_final = (z_t - sigma_final * eps_pred_final) / (alpha_final + 1e-8) 
    # x_final = x_final.clamp(-1, 1)

    # save_image_grid(x_final, dest_path, gridw, gridh)

    print(f'Saved output to {dest_path}')


def save_image_grid(images, path, gridw, gridh):
    images = (images * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    C, H, W = images.shape[1:]
    images = images.view(gridh, gridw, C, H, W).permute(0, 3, 1, 4, 2).reshape(gridh * H, gridw * W, C)
    image = PIL.Image.fromarray(images.cpu().numpy(), 'RGB')
    image.save(path)


def main():
    network_path = 'model/ueps.pkl'  
    output_path = 'output.png'
    sample_images_from_model(
        network_pkl=network_path,
        dest_path=output_path,
        seed=42,
        gridw=8,
        gridh=8,
        num_steps=1,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

if __name__ == "__main__":
    main()
