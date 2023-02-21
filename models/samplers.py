# used for sde diffusion model

import torch
import math
import torch.nn as nn
import functools
import numpy as np

# the forward process in SN_model
@torch.no_grad()
def forward_process(x,
                    marginal_prob_std,
                    batch_size=64,
                    num_steps=10,
                    device='cpu',
                    eps=1e-3,
                    end_t=1.,
                    only_final=True):
    sampling_list = []
    time_steps = torch.linspace(eps, end_t, num_steps, device=device)
    if only_final:
        return x + marginal_prob_std(time_steps[-1]) * torch.randn_like(x)
    else:
        for time_step in time_steps:
            sampling_list.append(x + marginal_prob_std(time_step) * torch.randn_like(x))
        return torch.stack(sampling_list)

#@title Define the Euler-Maruyama sampler (double click to expand or collapse)

## The number of sampling steps.
# num_steps =  500#@param {'type':'integer'}
from mimetypes import init

#@title Define the Euler-Maruyama sampler (double click to expand or collapse)

## The number of sampling steps.

def Euler_Maruyama_sampler(score_model,
                           marginal_prob_std,
                           diffusion_coeff,
                           dim=2,
                           batch_size=64,
                           num_steps=100,
                           device='cpu',
                           eps=1e-3,
                           start_t=1.,
                           save_times=8,
                           only_final=True,
                           init_x=None,
                           y=None):
    """Generate samples from score-based models with the Euler-Maruyama solver.
    Args:
        score_model: A PyTorch model that represents the time-dependent score-based model.
        marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
        diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
        batch_size: The number of samplers to generate by calling this function once.
        num_steps: The number of sampling steps.
        Equivalent to the number of discretized time steps.
        device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
        eps: The smallest time step for numerical stability.

    Returns:
        Samples.
    """
    sampling_list = []
    t = torch.ones(batch_size, device=device)
    # set t=1 to approximate the marginal distribution, init_x
    if init_x is None:
        init_x = torch.randn(batch_size, dim, device=device) \
            * marginal_prob_std(t)[:, None]
    time_steps = torch.linspace(start_t, eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    i = 0
    with torch.no_grad():
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step).to(device)
            mean_x = x + (g**2)[:, None] * score_model(x, batch_time_step, y) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None] * torch.randn_like(x)
            if not only_final and i%(num_steps//save_times)==0:
                sampling_list.append(x)
            i+=1
    # Do not include any noise in the last sampling step.
    return mean_x if only_final else torch.stack(sampling_list)


#@title Define the ODE sampler (double click to expand or collapse)
from scipy import integrate

## The error tolerance for the black-box ODE solver
error_tolerance = 1e-5 #@param {'type': 'number'}
def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                batch_size=64,
                atol=error_tolerance,
                rtol=error_tolerance,
                device='cuda',
                x=None,
                z=None,
                eps=1e-3,
                start_t=1.):
    """Generate samples from score-based models with black-box ODE solvers.
    Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation
        of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    x: The latent code that governs the final sample. If None, we start from p_1;
        otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
    """
    t = torch.ones(batch_size, device=device)
    # Create the latent code
    if x is None:
        init_x = torch.randn(batch_size, 20, device=device) \
            * marginal_prob_std(t)[:, None, None, None]
    else:
        init_x = x

    shape = init_x.shape

    def score_eval_wrapper(sample, time_steps, z=None):
        """A wrapper of the score-based model for use by the ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))
        with torch.no_grad():
            score = score_model(sample, time_steps, z)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones((shape[0],)) * t
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps, z)

    # Run the black-box ODE solver.
    res = integrate.solve_ivp(ode_func, (start_t, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')
    print(f"Number of function evaluations: {res.nfev}")
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

    return x