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

vae = torch.load("vae_model.pt", map_location=torch.device('cpu')) # change to your own vae model path
fcn = torch.load("fcn_model.pt", map_location=torch.device('cpu')) # change to your own fcn model path
vae.eval()
fcn.eval()

# fig,ax = plt.subplots()
# ax.imshow(xs_test[13].squeeze(),vmin=0,vmax=1.5)
# plt.show()

Mse,Re,Ae,Dr,Ssim, Psnr = [],[],[],[],[],[]

for idx in range(len(ys_test)):
    z = fcn(ys_test[idx])
    reco_eit1 = vae.decode(z)

    mse = torch.nn.MSELoss()(xs_test[idx].data.squeeze(), reco_eit1.data.squeeze())
    re = Re_sigma(xs_test[idx].data.squeeze().cpu(), reco_eit1.data.squeeze().cpu())
    ae = torch.nn.L1Loss()(xs_test[idx].data.squeeze(), reco_eit1.data.squeeze())
    dr = DR(xs_test[idx].data.squeeze(), reco_eit1.data.squeeze())
    ssim = SSIM(xs_test[idx].data.numpy().squeeze(), reco_eit1.data.numpy().squeeze())
    psnr = PSNR(xs_test[idx].data.squeeze(), reco_eit1.data.squeeze())

    Mse.append(mse)
    Re.append(re)
    Ae.append(ae)
    Dr.append(dr)
    Ssim.append(ssim)
    Psnr.append(psnr)


mse_mean = np.mean(Mse)
re_mean = np.mean(Re)
ae_mean = np.mean(Ae)
dr_mean = np.mean(Dr)
ssim_mean = np.mean(Ssim)
psnr_mean = np.mean(Psnr)


mse_std = np.std(Mse, ddof=1)
re_std = np.std(Re, ddof=1)
ae_std = np.std(Ae, ddof=1)
dr_std = np.std(Dr, ddof=1)
ssim_std = np.std(Ssim, ddof=1)
psnr_std = np.std(Psnr, ddof=1)

mse_var = np.var(Mse)
re_var = np.var(Re)
ae_var = np.var(Ae)
dr_var = np.var(Dr)
ssim_var = np.var(Ssim)
psnr_var = np.var(Psnr)
print(0)