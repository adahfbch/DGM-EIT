"""
Stage3: FCN + decoder, FCN from the second stage but decoder from first stage
"""
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from dival.measure import PSNR, SSIM
from dataset_paper import ys_test, xs_test

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

def samples(dataset,sample_size):
    """
    :return: sampler(index)
    """
    sampler = torch.utils.data.sampler.SubsetRandomSampler(
        np.random.choice(range(len(dataset)), sample_size))
    return sampler

# device = torch.device('cuda')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# pymodel
vae = torch.load("vae_model.pt", map_location=torch.device('cpu')) # change to your own vae model path
fcn = torch.load("fcn_model.pt", map_location=torch.device('cpu')) # change to your own fcn model path
vae.eval()
fcn.eval()
# vae_fcn_writer = SummaryWriter("vae_2/vae_fcn_")

# sampler
index = samples(xs_test, 4).indices

# mse = torch.nn.MSELoss()(xs_test3[index[0]].data.squeeze(), reco_eit.data.squeeze())
# re_sigma = Re_sigma(xs_test3[index[0]].data.squeeze(), reco_eit.data.squeeze())
# SSIM= compare_ssim(xs_test3[index[0]].data.squeeze(), reco_eit.data.squeeze())
# DR = DR(xs_test3[index[0]].data.squeeze(), reco_eit.data.squeeze())
# AE = torch.nn.L1Loss()(xs_test3[index[0]].data.squeeze(), reco_eit.data.squeeze())


# plot
# pair 1
ax0 = plt.subplot(2, 4, 1)
z = fcn(ys_test[index[0]])
# z = torch.normal(size=[1, 1, 1, 16], mean=0, std=1.0).cpu()
reco_eit1 = vae.decode(z)

mse = torch.nn.MSELoss()(xs_test[index[0]].data.squeeze(), reco_eit1.data.squeeze())
re_sigma = Re_sigma(xs_test[index[0]].data.squeeze().cpu(), reco_eit1.data.squeeze().cpu())
AE = torch.nn.L1Loss()(xs_test[index[0]].data.squeeze(), reco_eit1.data.squeeze())
dr = DR(xs_test[index[0]].data.squeeze(), reco_eit1.data.squeeze())
ssim= SSIM(xs_test[index[0]].data.numpy().squeeze(), reco_eit1.data.numpy().squeeze())

reco_eit11 = plt.imshow(reco_eit1.data.squeeze())
colorbar(reco_eit11)
plt.clim(0, 1.50)
plt.subplots_adjust(wspace =0.2, hspace =0.05)
plt.title("MSE: {:.5f} \n RE: {:.5f} \n AE:{:5f} \n DR:{:5f} \n SSIM:{:5f}".format(
    mse.item(), re_sigma.item(),AE.item(),dr.item(),ssim.item()),
          fontsize=5)
plt.axis('off')


ax1 = plt.subplot(2, 4, 5)
xs1 = plt.imshow(xs_test[index[0]].squeeze())
colorbar(xs1)
plt.clim(0, 1.50)
plt.subplots_adjust(wspace =0.2, hspace =0)
plt.axis('off')


# pair 2
plt.subplot(2, 4, 2)
z = fcn(ys_test[index[1]])
reco_eit2 = vae.decode(z)

mse1 = torch.nn.MSELoss()(xs_test[index[1]].data.squeeze(), reco_eit2.data.squeeze())
re_sigma1 = Re_sigma(xs_test[index[1]].data.squeeze(), reco_eit2.data.squeeze())
AE1 = torch.nn.L1Loss()(xs_test[index[1]].data.squeeze(), reco_eit2.data.squeeze())
dr1 = DR(xs_test[index[1]].data.squeeze(), reco_eit2.data.squeeze())
ssim1= SSIM(xs_test[index[1]].data.numpy().squeeze(), reco_eit2.data.numpy().squeeze())

reco_eit22 = plt.imshow(reco_eit2.data.squeeze())
colorbar(reco_eit22)
plt.clim(0, 1.50)
plt.subplots_adjust(wspace =0.2, hspace =0)
plt.title("MSE: {:.5f} \n RE_sigma: {:.5f} \n AE:{:5f} \n DR:{:5f} \n SSIM:{:5f}".format(
    mse1.item(), re_sigma1.item(),AE1.item(),dr1.item(),ssim1.item()), fontsize=5)
plt.axis('off')


plt.subplot(2, 4, 6)
xs2 = plt.imshow(xs_test[index[1]].squeeze())
colorbar(xs2)
plt.clim(0, 1.50)
plt.subplots_adjust(wspace =0.2, hspace =0)
plt.axis('off')


# pair 3
plt.subplot(2, 4, 3)
z = fcn(ys_test[index[2]])
reco_eit3 = vae.decode(z)

mse2 = torch.nn.MSELoss()(xs_test[index[2]].data.squeeze(), reco_eit3.data.squeeze())
re_sigma2 = Re_sigma(xs_test[index[2]].data.squeeze(), reco_eit3.data.squeeze())
AE2 = torch.nn.L1Loss()(xs_test[index[2]].data.squeeze(), reco_eit3.data.squeeze())
dr2 = DR(xs_test[index[2]].data.squeeze(), reco_eit3.data.squeeze())
ssim2= SSIM(xs_test[index[2]].data.numpy().squeeze(), reco_eit3.data.numpy().squeeze())

reco_eit33 = plt.imshow(reco_eit3.data.squeeze())
colorbar(reco_eit33)
plt.clim(0, 1.50)
plt.subplots_adjust(wspace =0.2, hspace =0)
plt.title("MSE: {:.5f} \n RE_sigma: {:.5f} \n AE:{:5f} \n DR:{:5f} \n SSIM:{:5f}".format(
    mse2.item(), re_sigma2.item(),AE2.item(),dr2.item(),ssim2.item()), fontsize=5)
plt.axis('off')


plt.subplot(2, 4, 7)
xs3 = plt.imshow(xs_test[index[2]].squeeze())
colorbar(xs3)
plt.clim(0, 1.50)
plt.subplots_adjust(wspace =0.2, hspace =0)
plt.axis('off')


# pair 4
plt.subplot(2, 4, 4)
z = fcn(ys_test[index[3]])
reco_eit4 = vae.decode(z)

mse3 = torch.nn.MSELoss()(xs_test[index[3]].data.squeeze(), reco_eit4.data.squeeze())
re_sigma3 = Re_sigma(xs_test[index[3]].data.squeeze(), reco_eit4.data.squeeze())
AE3 = torch.nn.L1Loss()(xs_test[index[3]].data.squeeze(), reco_eit4.data.squeeze())
dr3 = DR(xs_test[index[3]].data.squeeze(), reco_eit4.data.squeeze())
ssim3= SSIM(xs_test[index[3]].data.numpy().squeeze(), reco_eit4.data.numpy().squeeze())

reco_eit44 = plt.imshow(reco_eit4.data.squeeze())
colorbar(reco_eit44)
plt.clim(0, 1.50)
plt.subplots_adjust(wspace =0.2, hspace =0)
plt.title("MSE: {:.5f} \n RE_sigma: {:.5f} \n AE:{:5f} \n DR:{:5f} \n SSIM:{:5f}".format(
    mse3.item(), re_sigma3.item(),AE3.item(),dr3.item(),ssim3.item()), fontsize=5)
plt.axis('off')


plt.subplot(2, 4, 8)
xs4 = plt.imshow(xs_test[index[3]].squeeze())
colorbar(xs4)
plt.clim(0, 1.50)
plt.subplots_adjust(wspace =0.2, hspace =0)
plt.axis('off')

plt.tight_layout()
plt.show()
