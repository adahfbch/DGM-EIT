import matplotlib.pyplot as plt
import numpy as np
import torch
from dival.measure import PSNR, SSIM
from dataset_paper import ys_test, xs_test
import os

def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

def Re_sigma(x, reco_eit):
    Re_up = torch.nn.L1Loss()(x, reco_eit)
    t = torch.zeros(x.shape).cpu()
    Re_down = torch.nn.L1Loss()(x, t)
    Re_sigma = Re_up / Re_down
    return Re_sigma

def DR(x, reco_eit):
    DR_up = torch.max(reco_eit) - torch.min(reco_eit)
    DR_down = torch.max(x) - torch.min(x)
    DR = DR_up / DR_down
    return DR

import torch
import numpy as np
from dival.measure import PSNR, SSIM

def calculate_metrics(xs, reco, metric_funcs):
    """
    Calculate and return results for all indicators.
    :param xs: ground truth
    :param reco: reconstruction
    :param metric_funcs: Dictionary containing all indicator functions to be calculated
    :return: Dictionary with results for all indicators
    """
    results = {}
    for name, func in metric_funcs.items():
        results[name] = func(xs, reco)
    return results

def load_and_evaluate(ys_test, xs_test, vae_path, fcn_path):
    """
    加载模型并对测试数据集进行评估。
    :param ys_test:
    :param xs_test: 
    :param vae_path: VAE model path。
    :param fcn_path: FCN model pat。
    """
    device = torch.device('cpu')
    vae = torch.load(vae_path, map_location=device)
    fcn = torch.load(fcn_path, map_location=device)
    vae.eval()
    fcn.eval()

    metrics = {
        'MSE': lambda x, y: torch.nn.MSELoss()(x.squeeze(), y.squeeze()),
        'RE': Re_sigma,
        'AE': lambda x, y: torch.nn.L1Loss()(x.squeeze(), y.squeeze()),
        'DR': DR,
        'SSIM': lambda x, y: SSIM(x.numpy().squeeze(), y.numpy().squeeze()),
        'PSNR': PSNR
    }
    results = {name: [] for name in metrics.keys()}

    for y, x in zip(ys_test, xs_test):
        z = fcn(y)
        reco = vae.decode(z)
        metric_results = calculate_metrics(x.data.squeeze().cpu(), reco.data.squeeze().cpu(), metrics)
        for name, result in metric_results.items():
            results[name].append(result)

    # calculate 
    stats = {name: {'mean': np.mean(vals), 'std': np.std(vals, ddof=1), 'var': np.var(vals)}
             for name, vals in results.items()}
    
    return stats

stats = load_and_evaluate(ys_test, xs_test, "vae_model.pt", "fcn_model.pt")
print(stats)