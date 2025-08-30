import torch
import math

# Extended Reverse-time SDE solver for Diffusion Model
# ER-SDE-solver

# noise scale fuction
def customized_func(sigma, func_type=7, eta=0):
    """
    We provide several feasible special solutions.
    You can customize the specific solution you want as long as
    it satisfies the constraint equation in the paper.
    How to customize this noise scale fuction, see appendix A.8 in the paper.
    """
    if func_type == 1:  # ODE
        return sigma
    elif func_type == 2:  # Original SDE
        return sigma ** 2
    elif func_type == 3:  # SDE_1: sigma * (np.exp(-1 / sigma) + 10)
        return sigma * (torch.exp(sigma ** (eta - 1) / (eta - 1)) + 10)
    elif func_type == 4:  # SDE_2
        return sigma ** 1.5
    elif func_type == 5:  # SDE_3
        return sigma ** 2.5
    elif func_type == 6:  # SDE_4
        return sigma ** 0.9 * torch.log10(1 + 100 * sigma ** 1.5)
    elif func_type == 7:  # SDE_5
        return sigma * (torch.exp(sigma ** 0.3) + 10)

class ER_SDE_Solver:
    def uvel_3_order_taylor(
            self,
            model,
            x,
            sigmas,
            times,
            eta_fn=None,
            progress=False,
            **kwargs,
    ):
        """
        Taylor Method, 3rd order for UPrecondVel SDE.
        sigmas: [sigma_max, ..., sigma_min, 0]
        times: same shape as sigmas
        """
        num_steps = len(sigmas) - 1
        indices = range(num_steps)
        nums_intergrate = 100.0
        nums_indices = torch.arange(nums_intergrate, dtype=torch.float64, device=x.device)
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        old_eps = None
        old_d_eps = None
        for i in indices:
            t = times[i]
            t_next = times[i+1]
            sigma = sigmas[i]
            sigma_next = sigmas[i+1]
            h = sigma_next - sigma
            # Schedules and derivatives
            alpha_t = model.alpha(t)
            sigma_t = model.sigma(t)
            d_alpha_t = model.d_alpha(t)
            lambda_p = model.d_lambda(t)
            u_t = model.u(t)
            # eta_t = eta_fn(t) if eta_fn is not None else torch.ones_like(t, dtype=torch.float64, device=x.device)
            u_t_scaled = u_t * 13
            assert u_t_scaled**2 - model.d_lambda(t) >= 0, "u_t_scaled**2 must be larger than d_lambda(t) to avoid NaN"
            eta_t = u_t_scaled - torch.sqrt(u_t_scaled**2 - model.d_lambda(t))

            # Network prediction and eps recovery
            out = model(x, t, **kwargs)
            eps_hat = x - out / u_t

            # Drift term
            drift = (d_alpha_t / alpha_t) * x + (sigma_t * (eta_t ** 2 - lambda_p) / 2) * eps_hat
            # Diffusion amplitude
            diffusion = sigma_t * eta_t
            noise = torch.randn_like(x) * torch.sqrt(self.numerical_clip(sigma_next**2 - sigma**2))

            if old_eps is None or sigma_next == 0:
                x = x + h * drift + noise
                old_eps = eps_hat
            elif old_eps is not None and old_d_eps is None:
                # 2nd order
                sigma_indices = sigma_next + nums_indices / nums_intergrate * (sigma - sigma_next)
                s_int = torch.sum(1.0 * (sigma - sigma_next) / nums_intergrate)
                d_eps = (eps_hat - old_eps) / (sigma - sigmas[i-1])
                x = x + h * drift + (sigma_next - sigma + s_int) * d_eps + noise
                old_eps = eps_hat
                old_d_eps = d_eps
            else:
                # 3rd order
                sigma_indices = sigma_next + nums_indices / nums_intergrate * (sigma - sigma_next)
                s_int = torch.sum(1.0 * (sigma - sigma_next) / nums_intergrate)
                s_d_int = torch.sum((sigma_indices - sigma) * (sigma - sigma_next) / nums_intergrate)
                d_eps = (eps_hat - old_eps) / (sigma - sigmas[i-1])
                dd_eps = 2 * (d_eps - old_d_eps) / (sigma - sigmas[i-2])
                x = x + h * drift + (sigma_next - sigma + s_int) * d_eps \
                    + ((sigma_next - sigma) ** 2 / 2 + s_d_int) * dd_eps + noise
                old_eps = eps_hat
                old_d_eps = d_eps
        return x

    def uvel_heun(
            self,
            model,
            x,
            sigmas,
            times,
            eta_fn=None,
            progress=False,
            **kwargs,
    ):
        """
        Heun's method (2nd order) for UPrecondVel SDE.
        sigmas: [sigma_max, ..., sigma_min, 0]
        times: same shape as sigmas
        """
        num_steps = len(sigmas) - 1
        indices = range(num_steps)
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        for i in indices:
            t = times[i]
            t_next = times[i+1]
            sigma = sigmas[i]
            sigma_next = sigmas[i+1]
            h = sigma_next - sigma
            # Schedules and derivatives
            alpha_t = model.alpha(t)
            sigma_t = model.sigma(t)
            d_alpha_t = model.d_alpha(t)
            lambda_p = model.d_lambda(t)
            u_t = model.u(t)
            # eta_t = eta_fn(t) if eta_fn is not None else torch.ones_like(t, dtype=torch.float64, device=x.device)
            u_t_scaled = u_t * 13
            eta_t = u_t_scaled - torch.sqrt(u_t_scaled**2 - model.d_lambda(t))

            # Network prediction and eps recovery
            out = model(x, t, **kwargs)
            eps_hat = x - out / u_t
            drift = (d_alpha_t / alpha_t) * x + (sigma_t * (eta_t ** 2 - lambda_p) / 2) * eps_hat
            diffusion = sigma_t * eta_t
            noise = torch.randn_like(x) * torch.sqrt(self.numerical_clip(sigma_next**2 - sigma**2))

            x_euler = x + h * drift + noise

            # Heun correction
            t_next_val = t_next
            alpha_tn = model.alpha(t_next_val)
            sigma_tn = model.sigma(t_next_val)
            d_alpha_tn = model.d_alpha(t_next_val)
            lambda_pn = model.d_lambda(t_next_val)
            u_tn = model.u(t_next_val)
            eta_tn = eta_fn(t_next_val) if eta_fn is not None else torch.ones_like(t_next_val, dtype=torch.float64, device=x.device)
            outn = model(x_euler, t_next_val, **kwargs)
            eps_hatn = x_euler - outn / u_tn
            drift_n = (d_alpha_tn / alpha_tn) * x_euler + (sigma_tn * (eta_tn ** 2 - lambda_pn) / 2) * eps_hatn

            x = x + h * 0.5 * (drift + drift_n) + noise
        return x
    def __init__(
            self,
            sde_type = 've',
            model_prediction_type = 'x_start'
    ):  
        """
        Only ve and x_start are support in this version.
        EDM can be seen as a special VE-type, so we directly use it as VE.
        The remaining types will be added later.
        """
        assert model_prediction_type in ['noise', 'x_start', 'v']
        assert sde_type in ['ve', 'vp']
        self.sde_type = sde_type
        self.model_prediction_type = model_prediction_type

    def ve_xstart_1_order(
            self,
            model,
            x,
            sigmas,
            times,
            fn_sigma = customized_func,
            progress = False,
            **kwargs,
    ):
        """
        Euler Method, 1-order ER-SDE Solver.
        Support ve-type and model which predicts the data x_0.
        sigmas: index[0, 1, ..., N], sigmas[0] = sigma_max, sigma[N - 1] = sigma_min, sigma[N] = 0
        """
        assert self.model_prediction_type == 'x_start'
        assert self.sde_type == 've'

        num_steps = len(sigmas) - 1
        indices = range(num_steps)
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        for i in indices:
            x0 = model(x, times[i], **kwargs)
            r_fn = fn_sigma(sigmas[i + 1]) / fn_sigma(sigmas[i])
            noise = torch.randn_like(x) * torch.sqrt(self.numerical_clip(sigmas[i + 1]**2 - sigmas[i]**2 * r_fn**2))
            x = r_fn * x + (1 - r_fn) * x0 + noise
        return x

    def ve_start_2_order_taylor(
            self,
            model,
            x,
            sigmas,
            times,
            fn_sigma = customized_func,
            progress = True,
            **kwargs,
    ):
        """
        Taylor Method, 2-order ER-SDE Solver.
        Support ve-type and model which predicts the data x_0.
        sigmas: index [0, 1, ...,   N], sigmas[0] = sigma_max, sigma[N-1] = sigma_min, sigmas[N] = 0
        """
        assert self.model_prediction_type == 'x_start'
        assert self.sde_type == 've'

        num_steps = len(sigmas) - 1
        indices = range(num_steps)

        nums_intergrate = 100.0
        nums_indices = torch.arange(nums_intergrate, dtype=torch.float64, device=x.device)
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices) 
           
        old_x0 = None
        for i in indices:
            x0 = model(x, times[i], **kwargs)
            r_fn = fn_sigma(sigmas[i + 1]) / fn_sigma(sigmas[i])
            noise = torch.randn_like(x) * torch.sqrt(self.numerical_clip(sigmas[i + 1]**2 - sigmas[i]**2 * r_fn**2)) 
            if old_x0 == None or sigmas[i + 1]==0:
                x = r_fn * x + (1 - r_fn) * x0 + noise
            else:
                sigma_indices = sigmas[i + 1] + nums_indices/ nums_intergrate*(sigmas[i] - sigmas[i + 1])
                s_int = torch.sum(1.0 / fn_sigma(sigma_indices) * (sigmas[i] - sigmas[i + 1]) / nums_intergrate)
                d_x0 = (x0 - old_x0)/(sigmas[i] - sigmas[i - 1])
                x = r_fn * x + (1 - r_fn) * x0 + (sigmas[i + 1] - sigmas[i] + s_int * fn_sigma(sigmas[i + 1])) * d_x0 + noise
            old_x0 = x0
        return x

    def ve_start_3_order_taylor(
            self,
            model,
            x,
            sigmas,
            times,
            fn_sigma = customized_func,
            progress = False,
            **kwargs,
    ):
        """
        Taylor Method, 3-order ER-SDE Solver.
        Support ve-type and model which predicts the data x_0.
        sigmas: index [0, 1, ...,   N], sigmas[0] = sigma_max, sigma[N-1] = sigma_min, sigmas[N] = 0
        """
        assert self.model_prediction_type == 'x_start'
        assert self.sde_type == 've'

        num_steps = len(sigmas) - 1
        indices = range(num_steps)

        nums_intergrate = 100.0
        nums_indices = torch.arange(nums_intergrate, dtype=torch.float64, device=x.device)
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices) 
           
        old_x0 = None
        old_d_x0 = None
        for i in indices:
            x0 = model(x, times[i], **kwargs)
            r_fn = fn_sigma(sigmas[i + 1]) / fn_sigma(sigmas[i])
            noise = torch.randn_like(x) * torch.sqrt(self.numerical_clip(sigmas[i + 1]**2 - sigmas[i]**2 * r_fn**2)) 
            if old_x0 == None or sigmas[i + 1]==0:
                x = r_fn * x + (1 - r_fn) * x0 + noise
                old_x0 = x0
            elif (old_x0!= None) and (old_d_x0 == None):
                sigma_indices = sigmas[i + 1] + nums_indices/ nums_intergrate*(sigmas[i] - sigmas[i + 1])
                s_int = torch.sum(1.0 / fn_sigma(sigma_indices) * (sigmas[i] - sigmas[i + 1]) / nums_intergrate)
                d_x0 = (x0 - old_x0)/(sigmas[i] - sigmas[i - 1])
                x = r_fn * x + (1 - r_fn) * x0 + (sigmas[i + 1] - sigmas[i] + s_int * fn_sigma(sigmas[i + 1])) * d_x0 + noise

                old_x0 = x0
                old_d_x0 = d_x0
            else:
                sigma_indices = sigmas[i + 1] + nums_indices/ nums_intergrate*(sigmas[i] - sigmas[i + 1])
                s_int = torch.sum(1.0 / fn_sigma(sigma_indices) * (sigmas[i] - sigmas[i + 1]) / nums_intergrate)
                s_d_int = torch.sum((sigma_indices - sigmas[i])/ fn_sigma(sigma_indices) * (sigmas[i] - sigmas[i + 1]) / nums_intergrate)
                d_x0 = (x0 - old_x0)/(sigmas[i] - sigmas[i - 1])
                dd_x0 = 2 * (d_x0 - old_d_x0)/(sigmas[i] - sigmas[i - 2])
                x = r_fn * x + (1 - r_fn) * x0 + (sigmas[i + 1] - sigmas[i] + s_int * fn_sigma(sigmas[i + 1])) * d_x0 \
                    + ((sigmas[i + 1] - sigmas[i])**2/2 + s_d_int * fn_sigma(sigmas[i + 1])) * dd_x0 + noise
                old_x0 = x0
                old_d_x0 = d_x0
        return x
      
    def numerical_clip(self, x, eps = 1e-6):
        """
        Correct some numerical errors.
        Preventing negative numbers due to computer calculation accuracy errors.
        """
        if torch.abs(x) < eps:
            return torch.tensor(0.0).to(torch.float64).to(x.device)
        else:
            return x







