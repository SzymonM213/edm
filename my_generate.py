# your_sampler.py

import torch
import numpy as np
import tqdm
import pickle
import PIL.Image
import dnnlib
import os

# eta = 1 
@torch.no_grad()
def without_normalization(net, num_steps, ts, z_t, class_labels) :
    for i in tqdm.trange(num_steps, desc="Sampling"):
        t = ts[i].unsqueeze(0)
        s = ts[i + 1].unsqueeze(0)

        alpha_t = net._alpha(t)
        sigma_t = net._sigma(t)
        alpha_s = net._alpha(s)
        sigma_s = net._sigma(s)

        gamma_s = (alpha_s**2 / sigma_s**2) * (sigma_t**2 / alpha_t**2)
        # eta_s = torch.sqrt(1 - 1 / gamma_s)
        eta_s = torch.zeros_like(t)

        eps_scaled = net(z_t, t, class_labels)
        eps_pred = eps_scaled / net.u_constant.reshape(-1, 1, 1, 1)

        coeff_eps = sigma_s * torch.sqrt(1 - eta_s**2) - alpha_s * (sigma_t / alpha_t)
        coeff_z = alpha_s / alpha_t

        eps = torch.randn_like(z_t) if i < num_steps - 1 else 0

        z_t = coeff_eps.reshape(-1,1,1,1) * eps_pred + coeff_z.reshape(-1,1,1,1) * z_t + sigma_s.reshape(-1,1,1,1) * eta_s.reshape(-1,1,1,1) * eps
    return z_t

# eta equals this formula at the botton of the_main.pdf
@torch.no_grad()   
def normalization(net, num_steps, ts, z_t, class_labels):
    alpha_ts = net._alpha(ts)
    sigma_ts = net._sigma(ts)

    alpha_next = net._alpha(ts[1:])
    sigma_next = net._sigma(ts[1:])

    gamma_all = (alpha_next**2 / sigma_next**2) * (sigma_ts[:-1]**2 / alpha_ts[:-1]**2)
    sqrt_gamma_minus_1 = torch.sqrt(torch.clamp(gamma_all - 1, min=0.0))

    u = net.u_constant
    scaling_factor = torch.max(sqrt_gamma_minus_1 / u)

    if scaling_factor > 1:
        u = u * scaling_factor

    for i in tqdm.trange(num_steps, desc="Sampling"):
        t = ts[i].unsqueeze(0)
        s = ts[i + 1].unsqueeze(0)

        alpha_t = net._alpha(t)
        sigma_t = net._sigma(t)
        alpha_s = net._alpha(s)
        sigma_s = net._sigma(s)

        gamma_s = (alpha_s**2 / sigma_s**2) * (sigma_t**2 / alpha_t**2)

        numerator = gamma_s - 1
        denominator = torch.sqrt(gamma_s * u**2) - torch.sqrt(u**2 + 1 - gamma_s)
        eta_s = numerator / denominator

        eps_scaled = net(z_t, t, class_labels)
        eps_pred = eps_scaled / net.u_constant.reshape(-1, 1, 1, 1)

        coeff_eps = sigma_s * torch.sqrt(1 - eta_s**2) - alpha_s * (sigma_t / alpha_t)
        coeff_z = alpha_s / alpha_t

        eps = torch.randn_like(z_t) if i < num_steps - 1 else 0

        z_t = coeff_eps.reshape(-1,1,1,1) * eps_pred + coeff_z.reshape(-1,1,1,1) * z_t + sigma_s.reshape(-1,1,1,1) * eta_s.reshape(-1,1,1,1) * eps
    return z_t

# NIE DZIAŁA JESZCZE
@torch.no_grad()
def continous(net, num_steps, ts, z_t, class_labels):
    lambda_prime_all = -net._d_lambda(ts[1:])
    u = net.u_constant
    sqrt_expr = torch.sqrt(u**2 + lambda_prime_all)
    scaling_factor = torch.max(sqrt_expr / u)

    if scaling_factor > 1:
        u = u * scaling_factor

    for i in tqdm.trange(num_steps, desc="Sampling"):
        t = ts[i].unsqueeze(0)
        s = ts[i + 1].unsqueeze(0)

        alpha_t = net._alpha(t)
        alpha_s = net._alpha(s)

        lambda_s_prime = net._d_lambda(s)
        eta_s = u - torch.sqrt(u**2 + lambda_s_prime)

        eps_scaled = net(z_t, t, class_labels)
        eps_pred = eps_scaled / net.u_constant.reshape(-1, 1, 1, 1)

        coeff_eps = sigma_s * torch.sqrt(1 - eta_s**2) - alpha_s * (sigma_t / alpha_t)
        coeff_z = alpha_s / alpha_t

        eps = torch.randn_like(z_t) if i < num_steps - 1 else 0

        z_t = (
            coeff_eps.reshape(-1, 1, 1, 1) * eps_pred +
            coeff_z.reshape(-1, 1, 1, 1) * z_t +
            sigma_s.reshape(-1, 1, 1, 1) * eta_s.reshape(-1, 1, 1, 1) * eps
        )
    return z_t


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
    t_min = torch.tensor(5e-3)
    t_max = torch.tensor(1 - 5e-3)

    # JESZCZE NIE WIEM KTÓRY TO JELEŃ, SZUKAM JESZCZE 
    class_index = 3
    batch_size = gridw * gridh
    torch.manual_seed(seed)

    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)

    #class_labels[:, :] = 0
    #class_labels[:, class_idx] = 1

    # chciałeś po bożemu, to masz 
    class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]

    z_t = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    ts = torch.linspace(t_max, t_min, steps=num_steps + 1, device=device)

    z_1 = normalization(net, num_steps, ts, z_t, class_labels) 
    # z_2 = without_normalization(net, num_steps, ts, z_t, class_labels) 
    # z_3 = continous(net, num_steps, ts, z_t, class_labels) 

    save_image_grid(z_1, 'normalization.png', gridw, gridh)
    # save_image_grid(z_2, 'without_normalization.png', gridw, gridh)
    # save_image_grid(z_3, 'continous.png', gridw, gridh)

    # PYTANIE 
    # Na końcu dostaję z_t_min czyli muszę z tego odczytać x_0. 
    # Czyli powinien to obliczyć za pomoca z_t_min co nie ?

    # t_final = ts[-1].unsqueeze(0)  

    # alpha_final = net._alpha(t_final).reshape(-1,1,1,1)
    # sigma_final = net._sigma(t_final).reshape(-1,1,1,1)

    # print(f'Final timestep: alpha={alpha_final}, sigma={sigma_final}')

    # eps_pred_final = net(z_t, t_final) / net.u_constant.reshape(-1,1,1,1)
    # print(f'Finnal : { net(z_t, t_final)}')
    # x_final = (z_t - sigma_final * eps_pred_final) / (alpha_final + 1e-8) 
    # x_final = x_final.clamp(-1, 1)

    # save_image_grid(x_final, dest_path, gridw, gridh)



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
        seed=2137,
        gridw=8,
        gridh=8,
        num_steps=5,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

if __name__ == "__main__":
    main()
